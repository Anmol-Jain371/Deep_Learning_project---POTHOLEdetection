"""
Automated unit tests for Phase 4A.5 MobileSAM prompt-guided segmentation coordinate
unpadding/mapping, multi-mask candidate selection, quality rejection, and fallback.
"""

import os
import sys
import unittest
from unittest.mock import MagicMock
import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from experiments.prompt_guided_segmentation import (
    PromptSegConfig,
    MobileSAMBoxPromptAdapter,
    unpad_and_resize_sam_logits,
    assess_sam_candidates_for_box,
)
from experiments.roi_guided_segmentation import box_to_bool_mask


class TestSAMCoordinateMappingAndSelection(unittest.TestCase):
    def test_unpad_and_resize_sam_logits_roundtrip_16_9(self):
        # Source 360x640 -> gain = min(1024/360, 1024/640) = 1.6 -> active 576x1024 inside 1024x1024
        src_h, src_w = 360, 640
        gt = np.zeros((src_h, src_w), dtype=bool)
        gt[120:240, 200:440] = True

        # Construct synthetic 256x256 logits mapped to top-left [0:144, 0:256] (576/4=144, 1024/4=256)
        logits_256 = torch.full((1, 256, 256), -10.0, dtype=torch.float32)
        # In 256 space, 1.6/4 = 0.4 scale: y in [48:96], x in [80:176]
        logits_256[0, 48:96, 80:176] = 10.0

        out_masks = unpad_and_resize_sam_logits(logits_256, src_h=src_h, src_w=src_w, input_size=1024, mask_threshold=0.0)
        self.assertEqual(out_masks.shape, (1, src_h, src_w))
        inter = np.logical_and(gt, out_masks[0]).sum()
        union = np.logical_or(gt, out_masks[0]).sum()
        self.assertGreater(inter / union, 0.97)

    def test_multimask_selects_best_geometrically_valid_hypothesis(self):
        cfg = PromptSegConfig()
        H, W = 360, 640
        det_box = (200.0, 100.0, 360.0, 220.0)

        # Cand 0: valid object mask inside box (score 0.80)
        m0 = np.zeros((H, W), dtype=bool)
        m0[115:205, 220:340] = True
        # Cand 1: whole-road explosion even if higher SAM score (score 0.92) -> MUST be rejected
        m1 = np.ones((H, W), dtype=bool)
        # Cand 2: tiny corner fragment (score 0.65)
        m2 = np.zeros((H, W), dtype=bool)
        m2[105:115, 205:215] = True

        cands = np.stack([m0, m1, m2], axis=0)
        qa = assess_sam_candidates_for_box(det_box, cands, [0.80, 0.92, 0.65], W, H, cfg)
        self.assertTrue(qa["accepted"])
        self.assertFalse(qa["fallback_used"])
        self.assertEqual(qa["best_candidate_idx"], 0)

    def test_rejects_empty_and_spilling_candidates_to_fallback(self):
        cfg = PromptSegConfig()
        H, W = 360, 640
        det_box = (200.0, 100.0, 300.0, 180.0)

        qa_emp = assess_sam_candidates_for_box(det_box, np.zeros((0, H, W), dtype=bool), [], W, H, cfg)
        self.assertFalse(qa_emp["accepted"])
        self.assertTrue(qa_emp["fallback_used"])
        self.assertIn("empty_mask_prediction", qa_emp["rejection_reasons"])

        m_big = np.ones((H, W), dtype=bool)
        qa_big = assess_sam_candidates_for_box(det_box, m_big[None], [0.95], W, H, cfg)
        self.assertFalse(qa_big["accepted"])
        self.assertTrue(qa_big["fallback_used"])
        self.assertIn("excessive_area_ratio", qa_big["rejection_reasons"])

    def test_adapter_preserves_detection_on_empty_or_rejected_sam_mask(self):
        adapter = MobileSAMBoxPromptAdapter.__new__(MobileSAMBoxPromptAdapter)
        adapter.cfg = PromptSegConfig(mode="full_frame_embed")
        adapter.device = "cpu"
        adapter.startup_load_ms = 0.0

        mock_det = MagicMock()
        mock_det.boxes.__len__.return_value = 1
        mock_det.boxes.xyxy = torch.tensor([[100.0, 80.0, 220.0, 180.0]])
        mock_det.boxes.conf = torch.tensor([0.84])
        adapter.det_model = MagicMock()
        adapter.det_model.predict.return_value = [mock_det]

        mock_sam = MagicMock()
        mock_sam.model.mask_threshold = 0.0
        mock_sam.dataset = [(None, [np.zeros((360, 640, 3), dtype=np.uint8)], None)]
        mock_sam.preprocess.return_value = torch.zeros((1, 3, 1024, 1024))
        mock_sam.get_im_features.return_value = torch.zeros((1, 256, 64, 64))
        mock_sam._prepare_prompts.return_value = (torch.zeros((1, 4)), None, None, None)
        # Negative logits everywhere -> empty mask after threshold
        mock_sam._inference_features.return_value = (
            torch.full((3, 256, 256), -10.0),
            torch.tensor([0.1, 0.2, 0.15]),
        )
        adapter.sam_predictor = mock_sam

        frame = np.zeros((360, 640, 3), dtype=np.uint8)
        out = adapter.process_frame(frame)
        self.assertEqual(out["summary"]["num_proposals"], 1)
        self.assertEqual(out["summary"]["accepted_masks"], 0)
        self.assertEqual(out["summary"]["bbox_fallbacks"], 1)
        self.assertEqual(out["summary"]["detections_preserved"], 1)
        expected_box = box_to_bool_mask((100.0, 80.0, 220.0, 180.0), 360, 640)
        self.assertTrue(np.array_equal(out["union_effective_mask"], expected_box))


if __name__ == "__main__":
    unittest.main()
