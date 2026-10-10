"""
========================================================================================
📏 RoadVision AI — Physical Depth Ground-Truth Validation Workflow
========================================================================================
Purpose:
  Statistically evaluates RoadVision's monocular 3D depth estimates against real
  physical ground-truth ruler/straightedge measurements.

Strict Project Integrity Rules:
  1. NEVER fabricate physical measurements, GPS coordinates, precision, or recall.
  2. Clearly mark unvalidated status when physical measurements are not yet entered.
  3. Locked Severity Thresholds:
       - MILD:     < 4.0 cm
       - MODERATE: 4.0 - 7.9 cm
       - SEVERE:   >= 8.0 cm
     Water presence is tracked as a distinct flag (Aquaplaning Risk).
  4. Distance Sensitivity Bands:
       - 2 to 5 m
       - 5 to 8 m
       - 8 to 11 m
       - 11 to 15 m
  5. Evaluates current calibration equation:
       Delta_Z = Delta_d_raw * (Z_road ^ 1.35) * 0.082
========================================================================================
"""

import os
import sys
import csv
import json
import math
import argparse
from typing import List, Dict, Any, Optional, Tuple

if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass


# ======================================================================================
# SEVERITY CLASSIFICATION (Locked Thresholds)
# ======================================================================================
def classify_severity(depth_cm: float, is_water: bool = False) -> str:
    """
    Classifies pothole cavity depression into municipal priority categories.
    Preserves locked thresholds:
      - Mild:     < 4.0 cm
      - Moderate: 4.0 - 7.9 cm
      - Severe:   >= 8.0 cm
    """
    if is_water:
        return "SEVERE [WATER]"
    if depth_cm >= 8.0:
        return "SEVERE"
    elif depth_cm >= 4.0:
        return "MODERATE"
    else:
        return "MILD"


def base_severity_category(severity_str: str) -> str:
    """Extracts the underlying category (MILD, MODERATE, SEVERE) ignoring tags."""
    sev = severity_str.upper()
    if "SEVERE" in sev:
        return "SEVERE"
    if "MODERATE" in sev:
        return "MODERATE"
    return "MILD"


# ======================================================================================
# MANIFEST PARSER & VALIDATOR
# ======================================================================================
def load_manifest(manifest_path: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Loads ground-truth manifest CSV.
    Returns:
      (valid_measured_records, empty_or_unmeasured_records)
    """
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"Manifest not found at: {manifest_path}")

    valid_records = []
    unmeasured_records = []

    with open(manifest_path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader, start=1):
            pothole_id = (row.get("Pothole_ID") or f"P{idx:02d}").strip()
            source = (row.get("Source_Video_or_Image") or "").strip()
            timestamp = (row.get("Timestamp_Sec_or_Frame") or "").strip()
            dist_str = (row.get("Approx_Camera_Dist_m") or "").strip()
            phys_str = (row.get("Physical_Depth_cm") or "").strip()
            est_str = (row.get("Estimated_Depth_cm") or "").strip()
            water_str = (row.get("Water_Present") or "").strip().upper()
            notes = (row.get("Cavity_Description_Notes") or "").strip()

            is_water = water_str in ["YES", "TRUE", "1", "Y"] or "WATER" in notes.upper() or "PUDDLE" in notes.upper()

            # Parse physical measurement
            phys_depth: Optional[float] = None
            if phys_str:
                try:
                    phys_depth = float(phys_str)
                except ValueError:
                    phys_depth = None

            # Parse estimated depth
            est_depth: Optional[float] = None
            if est_str:
                try:
                    est_depth = float(est_str)
                except ValueError:
                    est_depth = None

            # Parse distance
            cam_dist: Optional[float] = None
            if dist_str:
                try:
                    cam_dist = float(dist_str)
                except ValueError:
                    cam_dist = None

            record = {
                "pothole_id": pothole_id,
                "source": source,
                "timestamp": timestamp,
                "distance_m": cam_dist,
                "physical_depth_cm": phys_depth,
                "estimated_depth_cm": est_depth,
                "water_present": is_water,
                "notes": notes,
                "raw_row": row
            }

            if phys_depth is not None:
                valid_records.append(record)
            else:
                unmeasured_records.append(record)

    return valid_records, unmeasured_records


# ======================================================================================
# STATISTICAL EVALUATION ENGINE
# ======================================================================================
def calculate_metrics(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Computes rigorous error metrics:
      - Absolute Errors, Signed Errors, Relative Errors
      - MAE (Mean Absolute Error)
      - RMSE (Root Mean Squared Error)
      - MBE (Mean Bias Error)
      - Severity Concordance
      - Distance Sensitivity Bands (2-5m, 5-8m, 8-11m, 11-15m)
      - Water vs Dry Stratification
    """
    if not records:
        return {"count": 0, "status": "UNVALIDATED"}

    eval_rows = []
    abs_errors = []
    sq_errors = []
    signed_errors = []
    relative_errors = []
    category_matches = 0

    confusion_matrix = {
        "MILD": {"MILD": 0, "MODERATE": 0, "SEVERE": 0},
        "MODERATE": {"MILD": 0, "MODERATE": 0, "SEVERE": 0},
        "SEVERE": {"MILD": 0, "MODERATE": 0, "SEVERE": 0},
    }

    # Distance bands: [2-5m), [5-8m), [8-11m), [11-15m]
    bands = {
        "2-5m": {"count": 0, "abs_errors": [], "sq_errors": [], "matches": 0},
        "5-8m": {"count": 0, "abs_errors": [], "sq_errors": [], "matches": 0},
        "8-11m": {"count": 0, "abs_errors": [], "sq_errors": [], "matches": 0},
        "11-15m": {"count": 0, "abs_errors": [], "sq_errors": [], "matches": 0},
        "other": {"count": 0, "abs_errors": [], "sq_errors": [], "matches": 0},
    }

    # Water stratification
    water_group = {"count": 0, "abs_errors": [], "sq_errors": []}
    dry_group = {"count": 0, "abs_errors": [], "sq_errors": []}

    for r in records:
        y_true = r["physical_depth_cm"]
        y_pred = r["estimated_depth_cm"]
        if y_pred is None:
            continue

        err = y_pred - y_true
        abs_err = abs(err)
        sq_err = err ** 2
        rel_err = (abs_err / max(0.1, y_true)) * 100.0

        abs_errors.append(abs_err)
        sq_errors.append(sq_err)
        signed_errors.append(err)
        relative_errors.append(rel_err)

        true_cat = base_severity_category(classify_severity(y_true, False))
        pred_cat = base_severity_category(classify_severity(y_pred, r["water_present"]))
        is_match = (true_cat == pred_cat)
        if is_match:
            category_matches += 1

        if true_cat in confusion_matrix and pred_cat in confusion_matrix[true_cat]:
            confusion_matrix[true_cat][pred_cat] += 1

        # Distance band binning
        d = r.get("distance_m")
        band_key = "other"
        if d is not None:
            if 2.0 <= d < 5.0:
                band_key = "2-5m"
            elif 5.0 <= d < 8.0:
                band_key = "5-8m"
            elif 8.0 <= d < 11.0:
                band_key = "8-11m"
            elif 11.0 <= d <= 15.0:
                band_key = "11-15m"

        bands[band_key]["count"] += 1
        bands[band_key]["abs_errors"].append(abs_err)
        bands[band_key]["sq_errors"].append(sq_err)
        if is_match:
            bands[band_key]["matches"] += 1

        # Water stratification
        if r["water_present"]:
            water_group["count"] += 1
            water_group["abs_errors"].append(abs_err)
            water_group["sq_errors"].append(sq_err)
        else:
            dry_group["count"] += 1
            dry_group["abs_errors"].append(abs_err)
            dry_group["sq_errors"].append(sq_err)

        eval_rows.append({
            "id": r["pothole_id"],
            "distance_m": d,
            "physical_cm": y_true,
            "estimated_cm": y_pred,
            "water": r["water_present"],
            "abs_error_cm": round(abs_err, 2),
            "signed_error_cm": round(err, 2),
            "relative_error_pct": round(rel_err, 1),
            "true_category": true_cat,
            "pred_category": pred_cat,
            "category_match": is_match,
        })

    n = len(abs_errors)
    if n == 0:
        return {"count": 0, "status": "NO_PREDICTIONS_TO_COMPARE"}

    mae = sum(abs_errors) / n
    rmse = math.sqrt(sum(sq_errors) / n)
    mbe = sum(signed_errors) / n
    mape = sum(relative_errors) / n
    concordance = (category_matches / n) * 100.0

    # Compile band summaries
    band_summary = {}
    for b_name, b_data in bands.items():
        bc = b_data["count"]
        if bc > 0:
            b_mae = sum(b_data["abs_errors"]) / bc
            b_rmse = math.sqrt(sum(b_data["sq_errors"]) / bc)
            b_acc = (b_data["matches"] / bc) * 100.0
            band_summary[b_name] = {
                "samples": bc,
                "mae_cm": round(b_mae, 2),
                "rmse_cm": round(b_rmse, 2),
                "concordance_pct": round(b_acc, 1)
            }
        else:
            band_summary[b_name] = {"samples": 0, "mae_cm": None, "rmse_cm": None, "concordance_pct": None}

    # Calibration scale factor fit check
    # Model: y_true ~= k_scale * y_pred
    sum_true_pred = sum(r["physical_cm"] * r["estimated_cm"] for r in eval_rows)
    sum_pred_sq = sum(r["estimated_cm"] ** 2 for r in eval_rows)
    recommended_scale = (sum_true_pred / max(1e-6, sum_pred_sq)) if sum_pred_sq > 0 else 1.0

    return {
        "status": "VALIDATED",
        "sample_count": n,
        "mae_cm": round(mae, 2),
        "rmse_cm": round(rmse, 2),
        "mbe_cm": round(mbe, 2),
        "mape_pct": round(mape, 1),
        "severity_concordance_pct": round(concordance, 1),
        "confusion_matrix": confusion_matrix,
        "distance_bands": band_summary,
        "water_breakdown": {
            "dry": {
                "count": dry_group["count"],
                "mae_cm": round(sum(dry_group["abs_errors"]) / dry_group["count"], 2) if dry_group["count"] > 0 else None,
                "rmse_cm": round(math.sqrt(sum(dry_group["sq_errors"]) / dry_group["count"]), 2) if dry_group["count"] > 0 else None,
            },
            "water": {
                "count": water_group["count"],
                "mae_cm": round(sum(water_group["abs_errors"]) / water_group["count"], 2) if water_group["count"] > 0 else None,
                "rmse_cm": round(math.sqrt(sum(water_group["sq_errors"]) / water_group["count"]), 2) if water_group["count"] > 0 else None,
            }
        },
        "calibration_assessment": {
            "current_formula": "Delta_Z = Delta_d_raw * (Z_road ^ 1.35) * 0.082",
            "current_k_disp": 0.082,
            "optimal_linear_scaling_factor": round(recommended_scale, 4),
            "suggested_k_disp": round(0.082 * recommended_scale, 5),
            "mean_bias_direction": "OVERESTIMATING" if mbe > 0.5 else ("UNDERESTIMATING" if mbe < -0.5 else "BALANCED")
        },
        "eval_rows": eval_rows
    }


# ======================================================================================
# FORMATTED REPORT PRINTERS
# ======================================================================================
def print_unvalidated_report(manifest_path: str, unmeasured: List[Dict[str, Any]]):
    print("\n" + "=" * 80)
    print("[!] ROADVISION AI -- PHYSICAL DEPTH VALIDATION STATUS: UNVALIDATED")
    print("=" * 80)
    print(f"Manifest File: {manifest_path}")
    print(f"Registered Template Pothole Slots: {len(unmeasured)}")
    print(f"Valid Ruler Ground-Truth Measurements Found: 0")
    print("\n[NOTICE] INTEGRITY POLICY:")
    print("   Physical depth estimation in metric centimetres is an UNVALIDATED PARAMETRIC ESTIMATE.")
    print("   MiDaS produces scale- and shift-ambiguous relative inverse depth.")
    print("   RoadVision converts this via optical ground-plane pinhole geometry:")
    print("       Delta_Z_cm = Delta_d_raw * (Z_road ^ 1.35) * 0.082")
    print("   Until real ruler/straightedge measurements are recorded, MAE, RMSE, and metric")
    print("   accuracy CANNOT be claimed or certified.")
    print("\n[PROTOCOL] REQUIRED MEASUREMENT PROTOCOL (10-15 Potholes):")
    print("   1. Source_Video_or_Image: Filename of dashcam/phone video or frame")
    print("   2. Timestamp_Sec_or_Frame: Video timestamp (e.g. 4.2s) or frame number")
    print("   3. Approx_Camera_Dist_m: Camera-to-pothole distance in metres (e.g. 3.5m, 6.0m)")
    print("   4. Physical_Depth_cm: Direct physical straightedge ruler measurement (e.g. 6.2cm)")
    print("   5. Water_Present: YES / NO (water-filled puddle reflection flag)")
    print("   6. Cavity_Description_Notes: Asphalt crater texture, road condition")
    print("\n[INFO] Populate 'depth_ground_truth_manifest.csv' and re-run this tool to produce")
    print("   the certified municipal depth accuracy evaluation report.")
    print("=" * 80 + "\n")


def print_validated_report(metrics: Dict[str, Any]):
    print("\n" + "=" * 80)
    print("[REPORT] ROADVISION AI -- PHYSICAL DEPTH VALIDATION AUDIT REPORT")
    print("=" * 80)
    print(f"Validated Pothole Samples : {metrics['sample_count']}")
    print(f"Mean Absolute Error (MAE) : {metrics['mae_cm']:.2f} cm")
    print(f"Root Mean Squared Error   : {metrics['rmse_cm']:.2f} cm")
    print(f"Mean Bias Error (MBE)     : {metrics['mbe_cm']:+.2f} cm ({metrics['calibration_assessment']['mean_bias_direction']})")
    print(f"Mean Absolute % Error     : {metrics['mape_pct']:.1f}%")
    print(f"Severity Concordance Rate : {metrics['severity_concordance_pct']:.1f}%")
    print("-" * 80)

    # Per-pothole breakdown
    print(f"{'ID':<6} {'Dist(m)':<8} {'Phys(cm)':<10} {'Est(cm)':<9} {'AbsErr':<8} {'RelErr':<8} {'TrueCat':<10} {'PredCat':<10} {'Match'}")
    print("-" * 80)
    for r in metrics["eval_rows"]:
        d_str = f"{r['distance_m']:.1f}" if r["distance_m"] is not None else "N/A"
        match_str = "[MATCH]" if r["category_match"] else "[MISMATCH]"
        print(f"{r['id']:<6} {d_str:<8} {r['physical_cm']:<10.1f} {r['estimated_cm']:<9.1f} {r['abs_error_cm']:<8.2f} {r['relative_error_pct']:<7.1f}% {r['true_category']:<10} {r['pred_category']:<10} {match_str}")
    print("-" * 80)

    # Distance bands
    print("\n[DISTANCE SENSITIVITY BREAKDOWN]:")
    for b_name, b_info in metrics["distance_bands"].items():
        if b_info["samples"] > 0:
            print(f"  * Band {b_name:<7}: {b_info['samples']} samples | MAE: {b_info['mae_cm']:.2f} cm | RMSE: {b_info['rmse_cm']:.2f} cm | Concordance: {b_info['concordance_pct']:.1f}%")
        else:
            print(f"  * Band {b_name:<7}: 0 samples")

    # Water condition breakdown
    wb = metrics["water_breakdown"]
    print("\n[SURFACE CONDITION BREAKDOWN]:")
    if wb["dry"]["count"] > 0:
        print(f"  * Dry Cavities   : {wb['dry']['count']} samples | MAE: {wb['dry']['mae_cm']:.2f} cm | RMSE: {wb['dry']['rmse_cm']:.2f} cm")
    if wb["water"]["count"] > 0:
        print(f"  * Water Hazards  : {wb['water']['count']} samples | MAE: {wb['water']['mae_cm']:.2f} cm | RMSE: {wb['water']['rmse_cm']:.2f} cm")

    # Calibration recommendation
    cal = metrics["calibration_assessment"]
    print("\n[CALIBRATION EQUATION ASSESSMENT]:")
    print(f"  * Current Formula   : {cal['current_formula']}")
    print(f"  * Optimal Scale Fit : {cal['optimal_linear_scaling_factor']:.4f}")
    if abs(cal['optimal_linear_scaling_factor'] - 1.0) < 0.10:
        print(f"  * Verdict           : Current calibration (k_disp=0.082) is SOUND. Retain without modification.")
    else:
        print(f"  * Verdict           : Suggested k_disp recalibration to {cal['suggested_k_disp']:.5f} to minimize residual bias.")
    print("=" * 80 + "\n")


# ======================================================================================
# SYNTHETIC TEST HARNESS (FOR VERIFYING MATHEMATICAL FORMULAS)
# ======================================================================================
def generate_synthetic_test_data() -> List[Dict[str, Any]]:
    """
    Generates controlled synthetic test records purely for testing code calculation
    precision (MAE, RMSE, concordance, distance bands).
    EXPLICITLY LABELED AS SYNTHETIC UNIT TEST - NOT REAL DATA.
    """
    return [
        {"pothole_id": "T01", "source": "test.mp4", "timestamp": "1.0s", "distance_m": 3.2, "physical_depth_cm": 3.0, "estimated_depth_cm": 3.4, "water_present": False, "notes": "Mild asphalt cavity"},
        {"pothole_id": "T02", "source": "test.mp4", "timestamp": "2.5s", "distance_m": 4.1, "physical_depth_cm": 5.5, "estimated_depth_cm": 5.1, "water_present": False, "notes": "Moderate pothole"},
        {"pothole_id": "T03", "source": "test.mp4", "timestamp": "3.8s", "distance_m": 6.4, "physical_depth_cm": 6.2, "estimated_depth_cm": 6.9, "water_present": False, "notes": "Moderate pothole"},
        {"pothole_id": "T04", "source": "test.mp4", "timestamp": "5.0s", "distance_m": 7.0, "physical_depth_cm": 8.5, "estimated_depth_cm": 9.1, "water_present": False, "notes": "Severe deep crater"},
        {"pothole_id": "T05", "source": "test.mp4", "timestamp": "6.2s", "distance_m": 9.5, "physical_depth_cm": 4.8, "estimated_depth_cm": 5.6, "water_present": True,  "notes": "Moderate water-filled puddle"},
        {"pothole_id": "T06", "source": "test.mp4", "timestamp": "7.5s", "distance_m": 12.0, "physical_depth_cm": 10.0, "estimated_depth_cm": 11.2, "water_present": False, "notes": "Severe road breach"},
    ]


# ======================================================================================
# MAIN EXECUTION CLI
# ======================================================================================
def main():
    parser = argparse.ArgumentParser(description="RoadVision Physical Depth Ground-Truth Validation Workflow")
    parser.add_argument("--manifest", type=str, default="depth_ground_truth_manifest.csv", help="Path to manifest CSV")
    parser.add_argument("--output-json", type=str, default="runs/depth_validation_report.json", help="Path to export report JSON")
    parser.add_argument("--output-csv", type=str, default="runs/depth_validation_results.csv", help="Path to export results CSV")
    parser.add_argument("--test-harness", action="store_true", help="Run mathematical calculation test harness on synthetic data")
    args = parser.parse_args()

    if args.test_harness:
        print("\n[TEST] RUNNING UNIT TEST HARNESS ON SYNTHETIC TEST CASES...")
        print("[!] NOTICE: This is a unit test of the mathematical evaluation engine.")
        print("    It uses synthetic test samples and DOES NOT represent physical road validation.")
        synthetic_records = generate_synthetic_test_data()
        metrics = calculate_metrics(synthetic_records)
        print_validated_report(metrics)
        assert metrics["sample_count"] == 6, "Sample count mismatch"
        assert metrics["mae_cm"] > 0, "MAE must be positive"
        assert metrics["rmse_cm"] >= metrics["mae_cm"], "RMSE must be >= MAE by Cauchy-Schwarz"
        print("[PASS] Mathematical evaluation test harness passed successfully!\n")
        return

    manifest_path = os.path.abspath(args.manifest)
    valid_records, unmeasured_records = load_manifest(manifest_path)

    if len(valid_records) == 0:
        print_unvalidated_report(manifest_path, unmeasured_records)
        os.makedirs(os.path.dirname(os.path.abspath(args.output_json)), exist_ok=True)
        unvalidated_summary = {
            "status": "UNVALIDATED",
            "manifest": manifest_path,
            "registered_slots": len(unmeasured_records),
            "physical_measurements_recorded": 0,
            "message": "Physical depth validation pending collection of 10-15 ruler/straightedge measurements."
        }
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(unvalidated_summary, f, indent=2)
        return

    # Valid records exist — compute metrics
    metrics = calculate_metrics(valid_records)
    print_validated_report(metrics)

    # Save output reports
    os.makedirs(os.path.dirname(os.path.abspath(args.output_json)), exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"💾 Saved JSON report to: {args.output_json}")

    if metrics.get("eval_rows"):
        with open(args.output_csv, "w", newline="", encoding="utf-8") as f:
            fieldnames = [
                "id", "distance_m", "physical_cm", "estimated_cm", "water",
                "abs_error_cm", "signed_error_cm", "relative_error_pct",
                "true_category", "pred_category", "category_match"
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in metrics["eval_rows"]:
                writer.writerow(r)
        print(f"💾 Saved Results CSV to: {args.output_csv}")


if __name__ == "__main__":
    main()
