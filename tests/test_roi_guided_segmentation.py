"""
Unit tests for Phase 4A.4 detector-guided ROI segmentation coordinate mapping,
mask reconstruction, association, quality checks, and bounding-box fallback.
"""

import os
import sys
import unittest
from unittest.mock import MagicMock
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from experiments.roi_guided_segmentation import (
    ROISegConfig,
    ROIGuidedSegmenter,
    expand_and_clip_box,
    letterbox_to_square,
    unletterbox_mask_to_crop,
    unletterbox_box_to_frame,
    place_crop_mask_in_frame,
    box_to_bool_mask,
    evaluate_roi_mask_quality,
)


class TestROICoordinateAndMaskMapping(unittest.TestCase):
    def test_expand_and_clip_box_boundaries(self):
        cx1, cy1, cx2, cy2 = expand_and_clip_box((5.0, 4.0, 55.0, 34.0), img_w=640, img_h=360, pad_ratio=0.4, min_pad_px=24)
        self.assertEqual(cx1, 0)
        self.assertEqual(cy1, 0)
        self.assertEqual(cx2, 79)  # 55 + 24
        self.assertEqual(cy2, 58)  # 34 + 24

        cx1, cy1, cx2, cy2 = expand_and_clip_box((580.0, 310.0, 638.0, 358.0), img_w=640, img_h=360, pad_ratio=0.5, min_pad_px=24)
        self.assertEqual(cx2, 640)
        self.assertEqual(cy2, 360)
        self.assertLess(cx1, 580)
        self.assertLess(cy1, 310)

    def test_letterbox_and_unletterbox_mask_roundtrip_wide_crop(self):
        # Wide rectangular crop (200x50) -> square 640 -> project mask back to crop & frame
        crop_h, crop_w = 50, 200
        crop_bgr = np.zeros((crop_h, crop_w, 3), dtype=np.uint8)
        canvas, meta = letterbox_to_square(crop_bgr, target_size=640)
        self.assertEqual(canvas.shape, (640, 640, 3))
        self.assertEqual(meta["new_w"], 640)
        self.assertEqual(meta["new_h"], 160)  # aspect 4:1 preserved

        # Draw known rectangle inside active area of canvas matching [40:140, 10:40] in crop
        gt_crop = np.zeros((crop_h, crop_w), dtype=bool)
        gt_crop[10:40, 40:140] = True

        scale = meta["scale"]
        cx1_c = int(round(40 * scale)) + meta["left"]
        cx2_c = int(round(140 * scale)) + meta["left"]
        cy1_c = int(round(10 * scale)) + meta["top"]
        cy2_c = int(round(40 * scale)) + meta["top"]
        canvas_prob = np.zeros((640, 640), dtype=np.float32)
        canvas_prob[cy1_c:cy2_c, cx1_c:cx2_c] = 1.0

        restored_crop = unletterbox_mask_to_crop(canvas_prob, meta, bin_thresh=0.5)
        self.assertEqual(restored_crop.shape, (crop_h, crop_w))
        inter = np.logical_and(gt_crop, restored_crop).sum()
        union = np.logical_or(gt_crop, restored_crop).sum()
        self.assertGreater(inter / union, 0.97)

        # Box roundtrip check
        mapped_box = unletterbox_box_to_frame(np.array([cx1_c, cy1_c, cx2_c, cy2_c]), (100, 200, 300, 250), meta)
        self.assertAlmostEqual(mapped_box[0], 140.0, delta=0.5)
        self.assertAlmostEqual(mapped_box[1], 210.0, delta=0.5)
        self.assertAlmostEqual(mapped_box[2], 240.0, delta=0.5)
        self.assertAlmostEqual(mapped_box[3], 240.0, delta=0.5)

        # Full-frame placement check
        full_mask = place_crop_mask_in_frame(restored_crop, (100, 200, 300, 250), img_h=360, img_w=640)
        self.assertEqual(full_mask.shape, (360, 640))
        self.assertTrue(full_mask[225, 190])
        self.assertFalse(full_mask[10, 10])


class TestROIMaskQualityAndFallback(unittest.TestCase):
    def setUp(self):
        self.cfg = ROISegConfig()
        self.det_box = (100.0, 100.0, 200.0, 200.0)
        self.crop_xyxy = (60, 60, 240, 240)  # 180x180 crop; det_box occupies [40:140, 40:140]

    def test_accepts_clean_mask_inside_detector_box(self):
        m = np.zeros((180, 180), dtype=bool)
        m[50:130, 50:130] = True  # 6400 px inside 10000 px box (ratio=0.64, inside=1.0)
        qa = evaluate_roi_mask_quality(self.det_box, self.crop_xyxy, [m], [0.82], self.cfg)
        self.assertTrue(qa["accepted"])
        self.assertFalse(qa["fallback_used"])
        self.assertEqual(qa["rejection_reasons"], [])

    def test_rejects_empty_prediction(self):
        qa = evaluate_roi_mask_quality(self.det_box, self.crop_xyxy, [], [], self.cfg)
        self.assertFalse(qa["accepted"])
        self.assertTrue(qa["fallback_used"])
        self.assertIn("empty_mask_prediction", qa["rejection_reasons"])

    def test_rejects_excessive_spill_outside_detector_box(self):
        m = np.zeros((180, 180), dtype=bool)
        m[5:175, 5:175] = True  # Spills across almost entire crop (ratio=2.89 > 1.35)
        qa = evaluate_roi_mask_quality(self.det_box, self.crop_xyxy, [m], [0.75], self.cfg)
        self.assertFalse(qa["accepted"])
        self.assertTrue(qa["fallback_used"])
        self.assertIn("excessive_area_ratio", qa["rejection_reasons"])

    def test_rejects_crop_boundary_contact(self):
        m = np.zeros((180, 180), dtype=bool)
        # Flush against left crop edge (x=0..49) spanning y=0..179 -> high crop boundary contact + spill
        m[:, 0:50] = True
        qa = evaluate_roi_mask_quality(self.det_box, self.crop_xyxy, [m], [0.65], self.cfg)
        self.assertFalse(qa["accepted"])
        self.assertIn("excessive_crop_boundary_contact", qa["rejection_reasons"])
        self.assertIn("outside_detector_box_spill", qa["rejection_reasons"])

    def test_rejects_fragmentation_and_undersize(self):
        m = np.zeros((180, 180), dtype=bool)
        for k in range(6):
            r = 50 + k * 10
            m[r : r + 3, 60:64] = True  # 6 disconnected components (>=4 px each)
        qa = evaluate_roi_mask_quality(self.det_box, self.crop_xyxy, [m], [0.60], self.cfg)
        self.assertFalse(qa["accepted"])
        self.assertIn("excessive_fragmentation", qa["rejection_reasons"])
        self.assertIn("undersized_vs_det_box", qa["rejection_reasons"])

    def test_rejects_ambiguous_competing_candidates(self):
        m1 = np.zeros((180, 180), dtype=bool)
        m1[45:135, 45:90] = True  # Left half of det box
        m2 = np.zeros((180, 180), dtype=bool)
        m2[45:135, 92:136] = True  # Right half of det box (nearly equal IoU with det box)
        qa = evaluate_roi_mask_quality(self.det_box, self.crop_xyxy, [m1, m2], [0.70, 0.68], self.cfg)
        self.assertFalse(qa["accepted"])
        self.assertTrue(qa["ambiguous_association"])
        self.assertIn("ambiguous_roi_association", qa["rejection_reasons"])

    def test_segmenter_preserves_detection_on_fallback(self):
        import torch
        seg_sys = ROIGuidedSegmenter.__new__(ROIGuidedSegmenter)
        seg_sys.cfg = ROISegConfig()
        seg_sys.device = "cpu"

        mock_det_res = MagicMock()
        mock_det_res.boxes.__len__.return_value = 1
        mock_det_res.boxes.xyxy = torch.tensor([[100.0, 100.0, 200.0, 200.0]])
        mock_det_res.boxes.conf = torch.tensor([0.88])
        seg_sys.det_model = MagicMock()
        seg_sys.det_model.predict.return_value = [mock_det_res]

        # Segmentation model returns zero detections inside ROI
        mock_seg_res = MagicMock()
        mock_seg_res.boxes = None
        mock_seg_res.masks = None
        seg_sys.seg_model = MagicMock()
        seg_sys.seg_model.predict.return_value = [mock_seg_res]

        frame = np.zeros((360, 640, 3), dtype=np.uint8)
        out = seg_sys.process_frame(frame)
        self.assertEqual(out["summary"]["num_proposals"], 1)
        self.assertEqual(out["summary"]["accepted_masks"], 0)
        self.assertEqual(out["summary"]["bbox_fallbacks"], 1)
        self.assertEqual(out["summary"]["detections_preserved"], 1)
        p0 = out["proposals"][0]
        self.assertEqual(p0["final_representation"], "bbox_fallback")
        expected_box = box_to_bool_mask((100.0, 100.0, 200.0, 200.0), 360, 640)
        self.assertTrue(np.array_equal(p0["final_mask"], expected_box))
        self.assertTrue(np.array_equal(out["union_effective_mask"], expected_box))
        self.assertEqual(int(out["union_accepted_mask"].sum()), 0)


if __name__ == "__main__":
    unittest.main()
