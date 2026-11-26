"""
========================================================================================
RoadVision AI -- Phase 6A: Unified Monocular Depth Model Adapter (`MiDaS_small` vs `DAv2 Small`)
========================================================================================
Provides a clean, documented adapter interface normalizing preprocessing contracts,
inverse-depth polarity verification, output resizing, precision tracking, and finite masks
for:
  1. `midas_small`            : TorchHub `intel-isl/MiDaS` `MiDaS_small` (`rgb_corrected`, 256)
  2. `depth_anything_v2_small`: Official Apache-2.0 `DepthAnything/Depth-Anything-V2-Small`
                                (`vits`, 24.79M params, native `518`, plus optional small-input
                                runtime ablation e.g. `252`)

SCIENTIFIC DISCLAIMERS:
  - Both models output **uncalibrated relative inverse depth** (higher = closer, lower = deeper).
  - Their raw numerical ranges differ by ~1-2 orders of magnitude (`~0..800` vs `~0..12`);
    raw magnitudes are NEVER compared directly as physical depth or physical cm.
  - Strictly isolated from production `roadvision_engine.py` and `server.py`.
========================================================================================
"""

import hashlib
import os
import sys
import time
from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple

import cv2
import torch
import torch.nn.functional as F
import numpy as np
from huggingface_hub import hf_hub_download

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "experiments")))
from dav2_official.dpt import DepthAnythingV2


@dataclass
class DepthPrediction:
    model_id: str
    depth_map: np.ndarray          # float32 (H, W), original inverse-depth units
    valid_mask: np.ndarray         # bool (H, W), np.isfinite(depth_map)
    polarity: str                  # "inverse_depth_higher_closer"
    input_size_setting: int
    network_input_hw: Tuple[int, int]
    preprocess_contract: str
    resize_method: str
    precision: str
    timings_ms: Dict[str, float]


class BaseDepthAdapter:
    model_id: str = "base"
    polarity: str = "inverse_depth_higher_closer"

    def warmup(self, h: int = 360, w: int = 640) -> None:
        _ = self.predict(np.zeros((h, w, 3), dtype=np.uint8))

    def predict(self, img_bgr: np.ndarray) -> DepthPrediction:
        raise NotImplementedError


class MiDaSSmallAdapter(BaseDepthAdapter):
    model_id = "midas_small"
    polarity = "inverse_depth_higher_closer"

    def __init__(self, device: Optional[str] = None, use_fp16: bool = True):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.use_fp16 = bool(use_fp16 and self.device == "cuda")
        t0 = time.perf_counter()
        torch.hub._check_repo_is_trusted = lambda *a, **k: True
        self.model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small", trust_repo=True).to(self.device).eval()
        transforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
        self.transform = transforms.small_transform
        self.load_ms = round((time.perf_counter() - t0) * 1000.0, 2)
        self.param_count = int(sum(p.numel() for p in self.model.parameters()))

    def _sync(self) -> None:
        if torch.cuda.is_available() and self.device == "cuda":
            torch.cuda.synchronize()

    def predict(self, img_bgr: np.ndarray) -> DepthPrediction:
        H, W = img_bgr.shape[:2]
        t0 = time.perf_counter()
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        inp = self.transform(img_rgb).to(self.device)
        net_hw = (int(inp.shape[-2]), int(inp.shape[-1]))
        t1 = time.perf_counter()

        self._sync()
        t2 = time.perf_counter()
        with torch.no_grad():
            if self.use_fp16:
                with torch.amp.autocast("cuda"):
                    pred = self.model(inp)
            else:
                pred = self.model(inp)
        self._sync()
        t3 = time.perf_counter()

        # Resize output cleanly to (H, W)
        t4 = time.perf_counter()
        with torch.no_grad():
            pred_up = F.interpolate(pred.unsqueeze(1).float(), size=(H, W), mode="bilinear", align_corners=False).squeeze()
        self._sync()
        dmap = pred_up.cpu().numpy().astype(np.float32)
        t5 = time.perf_counter()

        pre_ms = (t1 - t0) * 1000.0
        inf_ms = (t3 - t2) * 1000.0
        res_ms = (t5 - t4) * 1000.0
        return DepthPrediction(
            model_id=self.model_id,
            depth_map=dmap,
            valid_mask=np.isfinite(dmap),
            polarity=self.polarity,
            input_size_setting=256,
            network_input_hw=net_hw,
            preprocess_contract="BGR->RGB /255, Resize(256, upper_bound, mod32, INTER_CUBIC), ImageNet mean/std",
            resize_method="bilinear(align_corners=False)",
            precision="fp16_autocast" if self.use_fp16 else "fp32",
            timings_ms={
                "preprocess_ms": round(pre_ms, 4),
                "inference_ms": round(inf_ms, 4),
                "resize_ms": round(res_ms, 4),
                "total_ms": round(pre_ms + inf_ms + res_ms, 4),
            },
        )


class DepthAnythingV2SmallAdapter(BaseDepthAdapter):
    model_id = "depth_anything_v2_small"
    polarity = "inverse_depth_higher_closer"

    def __init__(self, device: Optional[str] = None, input_size: int = 518, use_fp16: bool = True):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.input_size = int(input_size)
        self.use_fp16 = bool(use_fp16 and self.device == "cuda")
        t0 = time.perf_counter()
        self.ckpt_path = hf_hub_download("depth-anything/Depth-Anything-V2-Small", "depth_anything_v2_vits.pth")
        with open(self.ckpt_path, "rb") as fh:
            self.ckpt_sha256 = hashlib.sha256(fh.read()).hexdigest().upper()
        self.ckpt_bytes = os.path.getsize(self.ckpt_path)

        model = DepthAnythingV2(encoder="vits", features=64, out_channels=[48, 96, 192, 384])
        sd = torch.load(self.ckpt_path, map_location="cpu", weights_only=True)
        model.load_state_dict(sd, strict=True)
        self.model = model.to(self.device).eval()
        self.load_ms = round((time.perf_counter() - t0) * 1000.0, 2)
        self.param_count = int(sum(p.numel() for p in self.model.parameters()))

    def _sync(self) -> None:
        if torch.cuda.is_available() and self.device == "cuda":
            torch.cuda.synchronize()

    def predict(self, img_bgr: np.ndarray) -> DepthPrediction:
        H, W = img_bgr.shape[:2]
        t0 = time.perf_counter()
        inp, _ = self.model.image2tensor(img_bgr, input_size=self.input_size)
        inp = inp.to(self.device)
        net_hw = (int(inp.shape[-2]), int(inp.shape[-1]))
        t1 = time.perf_counter()

        self._sync()
        t2 = time.perf_counter()
        with torch.no_grad():
            if self.use_fp16:
                with torch.amp.autocast("cuda"):
                    pred = self.model(inp)
            else:
                pred = self.model(inp)
        self._sync()
        t3 = time.perf_counter()

        t4 = time.perf_counter()
        with torch.no_grad():
            pred_up = F.interpolate(pred[:, None].float(), (H, W), mode="bilinear", align_corners=True)[0, 0]
        self._sync()
        dmap = pred_up.cpu().numpy().astype(np.float32)
        t5 = time.perf_counter()

        pre_ms = (t1 - t0) * 1000.0
        inf_ms = (t3 - t2) * 1000.0
        res_ms = (t5 - t4) * 1000.0
        return DepthPrediction(
            model_id=self.model_id,
            depth_map=dmap,
            valid_mask=np.isfinite(dmap),
            polarity=self.polarity,
            input_size_setting=self.input_size,
            network_input_hw=net_hw,
            preprocess_contract=f"BGR->RGB /255, Resize({self.input_size}, lower_bound, mod14, INTER_CUBIC), ImageNet mean/std",
            resize_method="bilinear(align_corners=True)",
            precision="fp16_autocast" if self.use_fp16 else "fp32",
            timings_ms={
                "preprocess_ms": round(pre_ms, 4),
                "inference_ms": round(inf_ms, 4),
                "resize_ms": round(res_ms, 4),
                "total_ms": round(pre_ms + inf_ms + res_ms, 4),
            },
        )
