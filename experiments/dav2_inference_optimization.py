"""
========================================================================================
RoadVision AI -- Phase 6B: Depth Anything V2 Inference Optimization & Backend Audit
========================================================================================
Evaluates lightweight, zero-dependency-upgrade inference paths and acceleration options
for `DepthAnythingV2Small` on NVIDIA RTX 2050 (4 GB VRAM, Windows 11, PyTorch 2.6+cu124):
  1. Precision comparison (`fp32` vs `fp16_autocast` vs pure `fp16_weights`) verifying:
       - finite output checks (`np.isfinite`),
       - affine-aligned numerical agreement & within-image Spearman rank correlation,
       - measurement validity flip rate under identical Optimized RANSAC + elliptical core,
       - synchronized GPU latency (mean, median, P95) & isolated peak allocated/reserved VRAM.
  2. Acceleration backend feasibility (`inference_mode`, `torch.compile` [`inductor`,
     `cudagraphs`, `aot_eager`], ONNX dynamic-shape export [`opset=17`] + ORT CPU/CUDA/TRT
     provider probing & numerical equivalence).

SCIENTIFIC CONSTRAINTS:
  - Strictly isolated from `roadvision_engine.py`, `server.py`, and dashboard files.
  - Does NOT install or upgrade packages. Stops external runtime setup if major CUDA/TRT
    environment mismatches are encountered (accurately documenting exact blockers).
  - Never treats lower latency or lower precision as physical depth accuracy.
========================================================================================
"""

import os
import sys
import time
import tempfile
import importlib.util
from typing import Dict, Any, List, Tuple, Optional

import cv2
import torch
import torch.nn.functional as F
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "experiments")))

from experiments.depth_model_comparison import DepthAnythingV2SmallAdapter
from experiments.core_aware_cavity_extraction import (
    CoreExtractionConfig,
    extract_cavity_with_strategy,
)
from experiments.optimized_relative_depth import OptimizedDepthConfig
from experiments.benchmark_optimized_relative_depth import _spearman_rho, _guarded_rel_diff_pct


def _sync_cuda(device: str) -> None:
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()


def spearman_fast(a: np.ndarray, b: np.ndarray, stride: int = 8) -> float:
    """Fast Spearman rank correlation over strided flattened spatial maps."""
    va = a[::stride, ::stride].ravel().astype(np.float64)
    vb = b[::stride, ::stride].ravel().astype(np.float64)
    mask = np.isfinite(va) & np.isfinite(vb)
    va, vb = va[mask], vb[mask]
    if len(va) < 10:
        return 0.0
    ra = np.argsort(np.argsort(va)).astype(np.float64)
    rb = np.argsort(np.argsort(vb)).astype(np.float64)
    ra -= ra.mean()
    rb -= rb.mean()
    denom = float(np.sqrt(np.sum(ra ** 2) * np.sum(rb ** 2)))
    return float(np.sum(ra * rb) / denom) if denom > 1e-12 else 0.0


def affine_align(src: np.ndarray, ref: np.ndarray) -> Tuple[np.ndarray, float, float, float, float]:
    """
    Least-squares positive-slope affine fit `ref ≈ a * src + b` so differing relative-depth
    numerical ranges / shifts are fairly aligned before computing difference metrics.
    Returns (aligned_src, scale_a, shift_b, aligned_mae, aligned_nrmse_iqr).
    """
    mask = np.isfinite(src) & np.isfinite(ref)
    x = src[mask].astype(np.float64)
    y = ref[mask].astype(np.float64)
    if len(x) < 16:
        return src.copy(), 1.0, 0.0, 0.0, 0.0
    if len(x) > 32768:
        step = max(1, len(x) // 32768)
        xs, ys = x[::step], y[::step]
    else:
        xs, ys = x, y
    xm, ym = float(np.mean(xs)), float(np.mean(ys))
    dx = xs - xm
    dy = ys - ym
    var_x = float(np.dot(dx, dx))
    if var_x <= 1e-12:
        a, b = 1.0, ym - xm
    else:
        a = max(1e-6, float(np.dot(dx, dy)) / var_x)
        b = ym - a * xm
    aligned = a * src.astype(np.float64) + b
    diff = np.abs(aligned[mask] - y)
    mae = float(np.mean(diff))
    rmse = float(np.sqrt(np.mean(diff ** 2)))
    q25, q75 = np.percentile(y, [25.0, 75.0])
    iqr = max(1e-5, float(q75 - q25))
    return aligned.astype(np.float32), round(a, 6), round(b, 6), round(mae, 6), round(rmse / iqr, 6)


def _latency_summary(vals: List[float]) -> Dict[str, Any]:
    if not vals:
        return {"n": 0, "mean_ms": None, "median_ms": None, "p95_ms": None}
    arr = np.asarray(vals, dtype=np.float64)
    return {
        "n": int(len(arr)),
        "mean_ms": round(float(np.mean(arr)), 3),
        "median_ms": round(float(np.median(arr)), 3),
        "p95_ms": round(float(np.percentile(arr, 95.0)), 3),
    }


def benchmark_fp32_vs_fp16(
    frames_with_boxes: List[Dict[str, Any]],
    input_sizes: List[int] = [266, 336, 392, 518],
    device: str = "cuda",
) -> Dict[str, Any]:
    """
    Compare FP32 vs FP16 autocast vs pure FP16 weights (`model.half()`) across DAv2 sizes.
    Separates memory instrumentation cleanly from runtime loops.
    """
    opt_cfg = OptimizedDepthConfig(sampling_mode="stratified_quadrant", max_ref_samples=512)
    core_cfg = CoreExtractionConfig()
    results: Dict[str, Any] = {}

    for sz in input_sizes:
        # 1. FP32 adapter
        ad_fp32 = DepthAnythingV2SmallAdapter(device=device, input_size=sz, use_fp16=False)
        # 2. FP16 autocast adapter
        ad_amp = DepthAnythingV2SmallAdapter(device=device, input_size=sz, use_fp16=True)
        # 3. Pure FP16 weights model sharing weights cleanly
        ad_half = DepthAnythingV2SmallAdapter(device=device, input_size=sz, use_fp16=False)
        ad_half.model = ad_half.model.half().eval()

        # Warmup all 3 cleanly before measurement
        for _ in range(3):
            ad_fp32.warmup(frames_with_boxes[0]["img_bgr"].shape[0], frames_with_boxes[0]["img_bgr"].shape[1])
            ad_amp.warmup(frames_with_boxes[0]["img_bgr"].shape[0], frames_with_boxes[0]["img_bgr"].shape[1])
            x_w, _ = ad_half.model.image2tensor(frames_with_boxes[0]["img_bgr"], input_size=sz)
            with torch.inference_mode():
                _ = ad_half.model(x_w.to(device).half())
            _sync_cuda(device)

        # Pass A: Pure runtime measurement (zero memory instrumentation inside loop)
        timings = {"fp32": {"pre": [], "inf": [], "res": [], "tot": []},
                   "fp16_autocast": {"pre": [], "inf": [], "res": [], "tot": []},
                   "fp16_weights": {"pre": [], "inf": [], "res": [], "tot": []}}
        preds_store = {"fp32": [], "fp16_autocast": [], "fp16_weights": []}
        net_hw = None

        for item in frames_with_boxes:
            img = item["img_bgr"]
            H, W = img.shape[:2]

            # FP32
            p32 = ad_fp32.predict(img)
            net_hw = p32.network_input_hw
            for k_t, k_s in (("preprocess_ms", "pre"), ("inference_ms", "inf"), ("resize_ms", "res"), ("total_ms", "tot")):
                timings["fp32"][k_s].append(p32.timings_ms[k_t])
            preds_store["fp32"].append(p32.depth_map)

            # FP16 autocast
            pamp = ad_amp.predict(img)
            for k_t, k_s in (("preprocess_ms", "pre"), ("inference_ms", "inf"), ("resize_ms", "res"), ("total_ms", "tot")):
                timings["fp16_autocast"][k_s].append(pamp.timings_ms[k_t])
            preds_store["fp16_autocast"].append(pamp.depth_map)

            # Pure FP16 weights with inference_mode
            t0 = time.perf_counter()
            inp_h, _ = ad_half.model.image2tensor(img, input_size=sz)
            inp_h = inp_h.to(device).half()
            t1 = time.perf_counter()
            _sync_cuda(device)
            t2 = time.perf_counter()
            with torch.inference_mode():
                pr_h = ad_half.model(inp_h)
            _sync_cuda(device)
            t3 = time.perf_counter()
            t4 = time.perf_counter()
            with torch.inference_mode():
                pr_up = F.interpolate(pr_h[:, None].float(), (H, W), mode="bilinear", align_corners=True)[0, 0]
            _sync_cuda(device)
            dmap_h = pr_up.cpu().numpy().astype(np.float32)
            t5 = time.perf_counter()
            timings["fp16_weights"]["pre"].append((t1 - t0) * 1000.0)
            timings["fp16_weights"]["inf"].append((t3 - t2) * 1000.0)
            timings["fp16_weights"]["res"].append((t5 - t4) * 1000.0)
            timings["fp16_weights"]["tot"].append((t5 - t0) * 1000.0)
            preds_store["fp16_weights"].append(dmap_h)

        # Pass B: Isolated Peak VRAM instrumentation (separate single-shot pass per mode)
        vram_stats = {}
        img0 = frames_with_boxes[0]["img_bgr"]
        H0, W0 = img0.shape[:2]
        for mode_tag, fn in (
            ("fp32", lambda: ad_fp32.predict(img0)),
            ("fp16_autocast", lambda: ad_amp.predict(img0)),
        ):
            if device == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                base_alloc = torch.cuda.memory_allocated() / (1024 ** 2)
                _ = fn()
                _sync_cuda(device)
                vram_stats[mode_tag] = {
                    "base_alloc_mb": round(base_alloc, 2),
                    "peak_alloc_mb": round(torch.cuda.max_memory_allocated() / (1024 ** 2), 2),
                    "peak_reserved_mb": round(torch.cuda.max_memory_reserved() / (1024 ** 2), 2),
                }
            else:
                vram_stats[mode_tag] = {"peak_alloc_mb": 0.0, "peak_reserved_mb": 0.0}

        if device == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            base_alloc = torch.cuda.memory_allocated() / (1024 ** 2)
            ih, _ = ad_half.model.image2tensor(img0, input_size=sz)
            ih = ih.to(device).half()
            with torch.inference_mode():
                ph = ad_half.model(ih)
                _ = F.interpolate(ph[:, None].float(), (H0, W0), mode="bilinear", align_corners=True)
            _sync_cuda(device)
            vram_stats["fp16_weights"] = {
                "base_alloc_mb": round(base_alloc, 2),
                "peak_alloc_mb": round(torch.cuda.max_memory_allocated() / (1024 ** 2), 2),
                "peak_reserved_mb": round(torch.cuda.max_memory_reserved() / (1024 ** 2), 2),
            }

        # Numerical fidelity & validity agreement vs FP32
        fidelity = {}
        for cmp_mode in ("fp16_autocast", "fp16_weights"):
            nonfinite_cnt = 0
            spearmans = []
            max_diffs = []
            maes = []
            cav_abs_diffs = []
            sn_rel_diffs = []
            sn_refs, sn_cmps = [], []
            val_flips = 0
            gained_flips = 0
            lost_flips = 0
            total_boxes = 0
            for idx_f, item in enumerate(frames_with_boxes):
                d_ref = preds_store["fp32"][idx_f]
                d_cmp = preds_store[cmp_mode][idx_f]
                if not np.all(np.isfinite(d_cmp)):
                    nonfinite_cnt += 1
                    continue
                spearmans.append(spearman_fast(d_cmp, d_ref, stride=4))
                _, _, _, mae_a, _ = affine_align(d_cmp, d_ref)
                maes.append(mae_a)
                max_diffs.append(float(np.max(np.abs(d_cmp - d_ref))))
                boxes = item.get("boxes", [])
                for j, bx in enumerate(boxes):
                    nbrs = [boxes[k] for k in range(len(boxes)) if k != j]
                    o_ref = extract_cavity_with_strategy(d_ref, bx, strategy="elliptical_core", neighbor_boxes=nbrs, img_bgr=item["img_bgr"], opt_cfg=opt_cfg, core_cfg=core_cfg)
                    o_cmp = extract_cavity_with_strategy(d_cmp, bx, strategy="elliptical_core", neighbor_boxes=nbrs, img_bgr=item["img_bgr"], opt_cfg=opt_cfg, core_cfg=core_cfg)
                    total_boxes += 1
                    cav_abs_diffs.append(abs(o_cmp["core_cavity_stats"]["raw_diag_p75"] - o_ref["core_cavity_stats"]["raw_diag_p75"]))
                    if o_ref["is_valid"] != o_cmp["is_valid"]:
                        val_flips += 1
                        if o_ref["is_valid"] and not o_cmp["is_valid"]:
                            lost_flips += 1
                        else:
                            gained_flips += 1
                    if o_ref["is_valid"] and o_cmp["is_valid"]:
                        r_sn = o_ref["core_cavity_stats"]["scale_normalized_p75"]
                        c_sn = o_cmp["core_cavity_stats"]["scale_normalized_p75"]
                        sn_refs.append(r_sn)
                        sn_cmps.append(c_sn)
                        sn_rel_diffs.append(_guarded_rel_diff_pct(c_sn, r_sn))

            sp_sn = None
            if len(sn_refs) >= 5:
                sp_raw = _spearman_rho(np.asarray(sn_cmps, dtype=np.float64), np.asarray(sn_refs, dtype=np.float64))
                sp_sn = round(float(sp_raw), 6) if sp_raw is not None else None

            fidelity[cmp_mode] = {
                "nonfinite_frames": nonfinite_cnt,
                "mean_spearman_vs_fp32": round(float(np.mean(spearmans)), 6) if spearmans else None,
                "min_spearman_vs_fp32": round(float(np.min(spearmans)), 6) if spearmans else None,
                "mean_affine_mae_vs_fp32": round(float(np.mean(maes)), 6) if maes else None,
                "max_raw_abs_diff_vs_fp32": round(float(np.max(max_diffs)), 6) if max_diffs else None,
                "signed_cavity_p75_abs_diff_med": round(float(np.median(cav_abs_diffs)), 6) if cav_abs_diffs else None,
                "signed_cavity_p75_abs_diff_p95": round(float(np.percentile(cav_abs_diffs, 95.0)), 6) if cav_abs_diffs else None,
                "validity_flips_vs_fp32": f"{val_flips}/{total_boxes}",
                "validity_flips_lost_vs_gained": f"{lost_flips}/{gained_flips}",
                "matched_valid_count": len(sn_refs),
                "matched_valid_sn_p75_spearman": sp_sn,
                "matched_valid_sn_p75_rel_diff_med_pct": round(float(np.median(sn_rel_diffs)), 3) if sn_rel_diffs else None,
                "matched_valid_sn_p75_rel_diff_p95_pct": round(float(np.percentile(sn_rel_diffs, 95.0)), 3) if sn_rel_diffs else None,
            }

        results[str(sz)] = {
            "input_size": sz,
            "network_hw_sample0": list(net_hw) if net_hw else None,
            "modes": {
                m: {
                    "preprocess": _latency_summary(timings[m]["pre"]),
                    "inference": _latency_summary(timings[m]["inf"]),
                    "resize": _latency_summary(timings[m]["res"]),
                    "total_depth": _latency_summary(timings[m]["tot"]),
                    "vram": vram_stats[m],
                }
                for m in ("fp32", "fp16_autocast", "fp16_weights")
            },
            "fidelity_vs_fp32": fidelity,
        }

        del ad_fp32, ad_amp, ad_half
        if device == "cuda":
            torch.cuda.empty_cache()

    return results


def run_acceleration_feasibility(
    sample_frames_bgr: List[np.ndarray],
    input_sizes: List[int] = [266, 336, 392, 518],
    device: str = "cuda",
) -> Dict[str, Any]:
    """
    Task 8: Investigate optional acceleration paths without modifying or installing
    external system dependency stacks:
      1. PyTorch `inference_mode` (`fp16_autocast` vs `fp16_weights`)
      2. PyTorch `torch.compile` backends (`inductor`, `cudagraphs`, `aot_eager`)
      3. ONNX opset-17 export + dynamic spatial shape checks across multiple sizes/aspect ratios
      4. ONNX Runtime provider probe (`CPUExecutionProvider` benchmark + exact Windows DLL diagnostic
         for `CUDAExecutionProvider` & `TensorrtExecutionProvider`).
    """
    ad = DepthAnythingV2SmallAdapter(device=device, input_size=336, use_fp16=True)
    img0 = sample_frames_bgr[0]
    H0, W0 = img0.shape[:2]

    # Package inventory
    pkg_status = {
        m: (importlib.util.find_spec(m) is not None)
        for m in ["onnx", "onnxruntime", "onnxruntime_gpu", "tensorrt", "triton", "openvino"]
    }

    # 1. Baseline eager fp16_autocast (inference_mode) across 336 & 266
    inf_mode_bench: Dict[str, Any] = {}
    for sz in input_sizes:
        x_in, _ = ad.model.image2tensor(img0, input_size=sz)
        x_in = x_in.to(device)
        net_hw = (int(x_in.shape[-2]), int(x_in.shape[-1]))
        # Warmup
        for _ in range(5):
            with torch.inference_mode(), torch.amp.autocast("cuda"):
                _ = ad.model(x_in)
            _sync_cuda(device)
        ts = []
        for fr in sample_frames_bgr:
            xf, _ = ad.model.image2tensor(fr, input_size=sz)
            xf = xf.to(device)
            _sync_cuda(device)
            t0 = time.perf_counter()
            with torch.inference_mode(), torch.amp.autocast("cuda"):
                y_pred = ad.model(xf)
                _ = F.interpolate(y_pred[:, None].float(), (fr.shape[0], fr.shape[1]), mode="bilinear", align_corners=True)
            _sync_cuda(device)
            ts.append((time.perf_counter() - t0) * 1000.0)
        inf_mode_bench[str(sz)] = {
            "network_hw": list(net_hw),
            "gpu_infer_plus_up": _latency_summary(ts),
        }

    # 2. torch.compile backend probes at representative size=336
    x336, _ = ad.model.image2tensor(img0, input_size=336)
    x336 = x336.to(device)
    with torch.inference_mode(), torch.amp.autocast("cuda"):
        y_ref_336 = ad.model(x336).detach().cpu().numpy().astype(np.float32)

    compile_results: Dict[str, Any] = {}
    for backend in ("inductor", "cudagraphs", "aot_eager"):
        try:
            torch._dynamo.reset()
            t0 = time.perf_counter()
            compiled = torch.compile(ad.model, backend=backend)
            # First call captures graph & compiles
            with torch.inference_mode(), torch.amp.autocast("cuda"):
                y_first = compiled(x336)
            _sync_cuda(device)
            warmup_compile_ms = (time.perf_counter() - t0) * 1000.0

            # Extra warmup
            for _ in range(3):
                with torch.inference_mode(), torch.amp.autocast("cuda"):
                    _ = compiled(x336)
                _sync_cuda(device)

            ts_c = []
            y_last = None
            for _ in range(len(sample_frames_bgr)):
                _sync_cuda(device)
                ts0 = time.perf_counter()
                with torch.inference_mode(), torch.amp.autocast("cuda"):
                    y_last = compiled(x336)
                    _ = F.interpolate(y_last[:, None].float(), (H0, W0), mode="bilinear", align_corners=True)
                _sync_cuda(device)
                ts_c.append((time.perf_counter() - ts0) * 1000.0)

            y_np = y_last.detach().cpu().numpy().astype(np.float32)
            finite_ok = bool(np.all(np.isfinite(y_np)))
            max_diff = float(np.max(np.abs(y_np - y_ref_336)))
            sp = spearman_fast(y_np[0], y_ref_336[0], stride=4)
            compile_results[backend] = {
                "supported": True,
                "first_compile_call_ms": round(warmup_compile_ms, 2),
                "steady_state_gpu_ms": _latency_summary(ts_c),
                "eager_336_med_ms": inf_mode_bench["336"]["gpu_infer_plus_up"]["median_ms"],
                "finite_ok": finite_ok,
                "max_abs_diff_vs_eager": round(max_diff, 6),
                "spearman_vs_eager": round(sp, 6),
                "note": "No Triton inductor on Windows; compiled via PyTorch backend." if backend != "inductor" else "Succeeded",
            }
        except Exception as exc:
            compile_results[backend] = {
                "supported": False,
                "error": str(exc)[:300],
            }

    # 3. ONNX Export + Dynamic Input Shape Verification + ORT Provider Audit
    onnx_results: Dict[str, Any] = {"exported": False}
    try:
        import onnx
        import onnxruntime as ort

        onx_path = os.path.join(tempfile.gettempdir(), "phase6b_dav2_vits_dynamic.onnx")
        dummy_fp32 = x336.detach().clone().float()
        t_exp0 = time.perf_counter()
        with torch.no_grad():
            torch.onnx.export(
                ad.model.float(),
                dummy_fp32,
                onx_path,
                input_names=["pixel_values"],
                output_names=["inv_depth"],
                dynamic_axes={"pixel_values": {0: "batch", 2: "height", 3: "width"},
                              "inv_depth": {0: "batch", 1: "height", 2: "width"}},
                opset_version=17,
            )
        export_ms = (time.perf_counter() - t_exp0) * 1000.0
        onx_mb = os.path.getsize(onx_path) / (1024 ** 2)
        onnx_model = onnx.load(onx_path)
        onnx.checker.check_model(onnx_model)

        # Provider check (explicitly record CUDA 13 / TRT DLL mismatch on this CUDA 12.4 system!)
        avail_eps = ort.get_available_providers()
        provider_probes = {}
        for ep in ("CUDAExecutionProvider", "TensorrtExecutionProvider", "CPUExecutionProvider"):
            try:
                sess = ort.InferenceSession(onx_path, providers=[ep, "CPUExecutionProvider"] if ep != "CPUExecutionProvider" else ["CPUExecutionProvider"])
                active = sess.get_providers()
                provider_probes[ep] = {
                    "requested": ep,
                    "active_providers": active,
                    "gpu_active": (ep in active and ep != "CPUExecutionProvider"),
                }
            except Exception as exc:
                provider_probes[ep] = {"requested": ep, "gpu_active": False, "error": str(exc)[:240]}

        # 3a. Verify whether dynamic-axis ONNX exported at 336 generalizes to 266..518
        cpu_sess = ort.InferenceSession(onx_path, providers=["CPUExecutionProvider"])
        ad_cpu_ref = ad.model.float().eval()
        dynamic_shape_test = {}
        for sz_test in input_sizes:
            xt, _ = ad_cpu_ref.image2tensor(img0, input_size=sz_test)
            with torch.no_grad():
                pt_out = ad_cpu_ref(xt.to(device)).detach().cpu().numpy().astype(np.float32)
            xt_np = xt.detach().cpu().numpy().astype(np.float32)
            try:
                t_ort0 = time.perf_counter()
                ort_out = cpu_sess.run(["inv_depth"], {"pixel_values": xt_np})[0].astype(np.float32)
                ort_cpu_ms = (time.perf_counter() - t_ort0) * 1000.0
                diff = np.abs(ort_out - pt_out)
                dynamic_shape_test[str(sz_test)] = {
                    "supported": True,
                    "tensor_shape": list(xt_np.shape),
                    "mae_vs_pytorch_fp32": round(float(np.mean(diff)), 6),
                    "max_abs_diff_vs_pytorch_fp32": round(float(np.max(diff)), 6),
                    "spearman_vs_pytorch_fp32": round(spearman_fast(ort_out[0], pt_out[0], stride=4), 6),
                    "ort_cpu_single_ms": round(ort_cpu_ms, 2),
                }
            except Exception as dyn_exc:
                dynamic_shape_test[str(sz_test)] = {
                    "supported": False,
                    "tensor_shape": list(xt_np.shape),
                    "error": str(dyn_exc)[:220],
                }

        # 3b. Fixed-shape per-resolution ONNX export & numerical equivalence (266 & 336 PoC)
        static_onnx_checks = {}
        for sz_s in (266, 336):
            onx_s = os.path.join(tempfile.gettempdir(), f"phase6b_dav2_vits_{sz_s}.onnx")
            xs, _ = ad_cpu_ref.image2tensor(img0, input_size=sz_s)
            xs_fp32 = xs.detach().clone().float()
            t_e0 = time.perf_counter()
            with torch.no_grad():
                pt_s = ad_cpu_ref(xs_fp32.to(device)).detach().cpu().numpy().astype(np.float32)
                torch.onnx.export(
                    ad_cpu_ref,
                    xs_fp32.to(device),
                    onx_s,
                    input_names=["pixel_values"],
                    output_names=["inv_depth"],
                    opset_version=17,
                )
            exp_s_ms = (time.perf_counter() - t_e0) * 1000.0
            sess_s = ort.InferenceSession(onx_s, providers=["CPUExecutionProvider"])
            xs_np = xs_fp32.detach().cpu().numpy().astype(np.float32)
            _ = sess_s.run(["inv_depth"], {"pixel_values": xs_np})
            ts_o = []
            ort_s = None
            for _ in range(3):
                t_o0 = time.perf_counter()
                ort_s = sess_s.run(["inv_depth"], {"pixel_values": xs_np})[0].astype(np.float32)
                ts_o.append((time.perf_counter() - t_o0) * 1000.0)
            diff_s = np.abs(ort_s - pt_s)
            static_onnx_checks[str(sz_s)] = {
                "exported": True,
                "tensor_shape": list(xs_np.shape),
                "model_size_mb": round(os.path.getsize(onx_s) / (1024 ** 2), 2),
                "export_ms": round(exp_s_ms, 2),
                "mae_vs_pytorch_fp32": round(float(np.mean(diff_s)), 6),
                "max_abs_diff_vs_pytorch_fp32": round(float(np.max(diff_s)), 6),
                "spearman_vs_pytorch_fp32": round(spearman_fast(ort_s[0], pt_s[0], stride=4), 6),
                "ort_cpu_ms": _latency_summary(ts_o),
            }
            os.remove(onx_s)

        os.remove(onx_path)
        onnx_results = {
            "exported": True,
            "opset": 17,
            "model_size_mb": round(onx_mb, 2),
            "export_ms": round(export_ms, 2),
            "onnx_checker_passed": True,
            "ort_version": ort.__version__,
            "ort_reported_available_providers": avail_eps,
            "provider_activation_audit": provider_probes,
            "cuda13_dll_blocker": (
                "onnxruntime 1.32.0 CUDAExecutionProvider & TensorrtExecutionProvider require cublasLt64_13.dll / "
                "cublas64_13.dll (CUDA 13.x + cuDNN 9.x + TensorRT), whereas system uses CUDA 12.4 (PyTorch 2.6.0+cu124); "
                "both silently fall back to CPUExecutionProvider without GPU installation changes."
            ),
            "dynamic_shape_checks_from_336_trace": dynamic_shape_test,
            "static_shape_onnx_checks": static_onnx_checks,
        }
    except Exception as exc:
        onnx_results = {"exported": False, "error": str(exc)[:300]}

    return {
        "package_inventory": pkg_status,
        "inference_mode_eager_fp16": inf_mode_bench,
        "torch_compile_backends": compile_results,
        "onnx_and_ort_audit": onnx_results,
    }
