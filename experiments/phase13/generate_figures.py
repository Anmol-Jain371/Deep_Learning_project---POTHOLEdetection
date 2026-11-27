"""
Phase 13 Step 28: Publication-Ready Figure & Raw Plot-Data Generator (PNG + SVG).

Reads ONLY actual executed metrics from `aggregate_metrics.json`; never invents values.
Explicitly labels exploratory and synthetic sources in figure titles/notes.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _save_both(fig, fig_dir: str, stem: str, raw_data: Dict[str, Any]) -> Dict[str, str]:
    png_path = os.path.join(fig_dir, f"{stem}.png")
    svg_path = os.path.join(fig_dir, f"{stem}.svg")
    json_path = os.path.join(fig_dir, f"{stem}_data.json")
    fig.tight_layout()
    fig.savefig(png_path, dpi=180)
    fig.savefig(svg_path)
    plt.close(fig)
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(raw_data, fh, indent=2)
    return {"png": f"figures/{stem}.png", "svg": f"figures/{stem}.svg", "data": f"figures/{stem}_data.json"}


def generate_all_figures(metrics: Dict[str, Any], out_dir: str) -> List[Dict[str, str]]:
    fig_dir = os.path.join(out_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    generated: List[Dict[str, str]] = []

    plt.rcParams.update({
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "axes.edgecolor": "#475569",
        "axes.grid": True,
        "grid.alpha": 0.25,
    })

    # Figure 1: Detection Comparison (Exploratory dataset_unified_test)
    det = metrics.get("detection", {}).get("models", {})
    if det:
        labels = []
        p_vals, r_vals, map50_vals, fps_vals = [], [], [], []
        for mname, mdata in det.items():
            sp = (mdata.get("splits") or {}).get("dataset_unified_test") or next(iter((mdata.get("splits") or {}).values()), None)
            if not sp:
                continue
            labels.append(mname)
            p_vals.append(sp.get("precision") or 0.0)
            r_vals.append(sp.get("recall") or 0.0)
            map50_vals.append(sp.get("map50") or 0.0)
            fps_vals.append((mdata.get("runtime_benchmark") or {}).get("throughput_fps") or 0.0)

        if labels:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.5, 4.0))
            x = np.arange(len(labels))
            w = 0.24
            ax1.bar(x - w, p_vals, width=w, label="Precision", color="#0891b2")
            ax1.bar(x, r_vals, width=w, label="Recall", color="#22d3ee")
            ax1.bar(x + w, map50_vals, width=w, label="mAP@0.50", color="#f59e0b")
            ax1.set_xticks(x)
            ax1.set_xticklabels(labels)
            ax1.set_ylim(0, 1.0)
            ax1.set_ylabel("Score")
            ax1.set_title("Detection Metrics (Exploratory — Non-Independent)")
            ax1.legend(loc="upper right")

            ax2.bar(x, fps_vals, width=0.45, color="#0284c7")
            ax2.set_xticks(x)
            ax2.set_xticklabels(labels)
            ax2.set_ylabel("Throughput (FPS)")
            ax2.set_title("Measured Inference Throughput (RTX 2050)")
            generated.append(_save_both(fig, fig_dir, "fig1_detection_comparison", {"models": labels, "precision": p_vals, "recall": r_vals, "map50": map50_vals, "fps": fps_vals}))

    # Figure 2: Reference Plane Ablation (Synthetic)
    ref_ab = metrics.get("geometry", {}).get("reference_plane_ablation", {})
    if ref_ab:
        mnames = [k for k in ("B_ols_plane", "C_ransac_baseline", "D_optimized_msac_ransac") if k in ref_ab]
        if mnames:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.5, 4.0))
            out_rmse = [ref_ab[m].get("outlier_mean_plane_rmse") or 0.0 for m in mnames]
            lat_ms = [ref_ab[m].get("mean_latency_ms") or 0.0 for m in mnames]
            x = np.arange(len(mnames))
            ax1.bar(x, out_rmse, width=0.5, color="#f97316")
            ax1.set_xticks(x)
            ax1.set_xticklabels(["OLS", "Baseline RANSAC", "Optimized MSAC"])
            ax1.set_ylabel("Mean Plane RMSE (relative units)")
            ax1.set_title("Synthetic Outlier Plane Recovery Error")

            ax2.bar(x, lat_ms, width=0.5, color="#06b6d4")
            ax2.set_xticks(x)
            ax2.set_xticklabels(["OLS", "Baseline RANSAC", "Optimized MSAC"])
            ax2.set_ylabel("Latency per ROI (ms)")
            ax2.set_title("Reference Fitting CPU Latency")
            generated.append(_save_both(fig, fig_dir, "fig2_ref_plane_ablation", {"methods": mnames, "outlier_rmse": out_rmse, "mean_ms": lat_ms}))

    # Figure 3: Depth Model & Resolution Scaling
    depth_ab = metrics.get("depth_cavity", {}).get("depth_model_and_resolution_ablation", {})
    if depth_ab:
        arms = list(depth_ab.keys())
        inf_ms = [depth_ab[a].get("inference_ms_mean") or 0.0 for a in arms]
        cov_pct = [depth_ab[a].get("fresh_valid_coverage_pct") or 0.0 for a in arms]
        vram = [depth_ab[a].get("peak_allocated_mb") or 0.0 for a in arms]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.2))
        x = np.arange(len(arms))
        ax1.bar(x, inf_ms, width=0.55, color="#0891b2")
        ax1.set_xticks(x)
        ax1.set_xticklabels(arms, rotation=28, ha="right")
        ax1.set_ylabel("Mean GPU Inference Latency (ms)")
        ax1.set_title("Depth Model Latency (Exploratory Frames)")

        ax2.bar(x - 0.18, cov_pct, width=0.36, label="Fresh Valid Coverage (%)", color="#10b981")
        ax2.bar(x + 0.18, vram, width=0.36, label="Peak Alloc VRAM (MB)", color="#6366f1")
        ax2.set_xticks(x)
        ax2.set_xticklabels(arms, rotation=28, ha="right")
        ax2.set_title("Coverage (%) & Peak VRAM (MB)")
        ax2.legend(loc="upper right")
        generated.append(_save_both(fig, fig_dir, "fig3_depth_resolution_ablation", {"arms": arms, "inf_ms": inf_ms, "coverage_pct": cov_pct, "vram_mb": vram}))

    # Figure 4: Scheduler Trade-off
    sched_ab = metrics.get("scheduler_temporal", {}).get("scheduling_ablation", {})
    if sched_ab:
        strats = list(sched_ab.keys())
        red_pct = [sched_ab[s].get("depth_call_reduction_vs_always_on_pct") or 0.0 for s in strats]
        fresh_pct = [sched_ab[s].get("fresh_valid_coverage_pct") or 0.0 for s in strats]
        eff_pct = [sched_ab[s].get("effective_availability_pct") or 0.0 for s in strats]

        fig, ax = plt.subplots(figsize=(9.5, 4.2))
        x = np.arange(len(strats))
        w = 0.25
        ax.bar(x - w, red_pct, width=w, label="Depth Call Reduction (%)", color="#f59e0b")
        ax.bar(x, fresh_pct, width=w, label="Fresh Valid Coverage (%)", color="#06b6d4")
        ax.bar(x + w, eff_pct, width=w, label="Effective Availability (%)", color="#10b981")
        ax.set_xticks(x)
        ax.set_xticklabels(strats, rotation=15, ha="right")
        ax.set_ylim(0, 108)
        ax.set_ylabel("Percentage (%)")
        ax.set_title("Adaptive Depth Scheduler Trade-offs (Frozen Exploratory Stream)")
        ax.legend(loc="upper right")
        generated.append(_save_both(fig, fig_dir, "fig4_scheduler_tradeoff", {"strategies": strats, "reduction_pct": red_pct, "fresh_pct": fresh_pct, "effective_pct": eff_pct}))

    # Figure 5: Phase 9 Quality Estimator Ablation
    q_dist = metrics.get("quality_geo_maint", {}).get("phase9_quality_ablation", {}).get("real_exploratory_distributions", {})
    if q_dist:
        cats = ["HIGH_QUALITY", "MODERATE_QUALITY", "LOW_QUALITY", "INVALID", "INSUFFICIENT_EVIDENCE"]
        variants = list(q_dist.keys())
        fig, ax = plt.subplots(figsize=(9.5, 4.2))
        x = np.arange(len(cats))
        w = 0.25
        colors = ["#0891b2", "#f59e0b", "#10b981"]
        for idx, var in enumerate(variants):
            counts = [int(q_dist[var].get(c, 0)) for c in cats]
            ax.bar(x + (idx - 1) * w, counts, width=w, label=var, color=colors[idx % len(colors)])
        ax.set_xticks(x)
        ax.set_xticklabels(cats, rotation=15, ha="right")
        ax.set_ylabel("Observation Count")
        ax.set_title("Phase 9 Quality Estimator Variants (Exploratory Job 0f286074)")
        ax.legend(loc="upper right")
        generated.append(_save_both(fig, fig_dir, "fig5_quality_estimator_ablation", {"categories": cats, "distributions": q_dist}))

    return generated
