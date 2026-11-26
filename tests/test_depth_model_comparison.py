import unittest
import cv2
import numpy as np
import torch

from experiments.depth_model_comparison import (
    DepthAnythingV2SmallAdapter,
    MiDaSSmallAdapter,
)


class TestDepthModelAdapters(unittest.TestCase):
    def test_both_adapters_contract_polarity_and_resolution(self):
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        cap = cv2.VideoCapture("roadvision-dashboard/public/videos/raw_survey.mp4")
        ok, img = cap.read()
        cap.release()
        self.assertTrue(ok and img is not None)
        H, W = img.shape[:2]

        for adapter in (
            MiDaSSmallAdapter(device=dev),
            DepthAnythingV2SmallAdapter(device=dev, input_size=518),
        ):
            adapter.warmup(H, W)
            pred = adapter.predict(img)
            self.assertEqual(pred.depth_map.shape, (H, W))
            self.assertEqual(pred.depth_map.dtype, np.float32)
            self.assertEqual(pred.valid_mask.shape, (H, W))
            self.assertTrue(np.all(pred.valid_mask))
            self.assertEqual(pred.polarity, "inverse_depth_higher_closer")
            # Near road bottom must exceed far horizon top (higher = closer)
            top_med = float(np.median(pred.depth_map[: int(0.15 * H), :]))
            bot_med = float(np.median(pred.depth_map[int(0.85 * H) :, :]))
            self.assertGreater(bot_med, top_med, f"{adapter.model_id} failed bottom>top polarity check")


if __name__ == "__main__":
    unittest.main()
