"""
Phase 11: Conservative Cross-Trip Geospatial Candidate Matching & Bounded Clustering.

Uses exact Haversine distance (never Euclidean lat/lon degrees) and prevents
unrestricted chain-merging via complete-linkage diameter constraints.

Scientific & Engineering Guarantees:
- Retains ALL original track records (creates cluster memberships instead of deleting tracks).
- Rejects or flags candidates with poor GPS horizontal accuracy or missing sync evidence.
- Never compares incompatible depth model/resolution outputs for depth similarity.
- Uses conservative terminology (POTENTIAL_MATCH, REVIEW_REQUIRED, UNMATCHED);
  never claims a cluster is a confirmed single physical pothole without ground truth.
"""

from __future__ import annotations

import math
import uuid
from typing import Any, Dict, List, Optional, Sequence, Tuple

from roadvision_v2.config import GeospatialConfig

EARTH_RADIUS_M = 6371000.0

MATCH_STATUSES = ("POTENTIAL_MATCH", "REVIEW_REQUIRED", "UNMATCHED")
REVIEW_STATUSES = ("PENDING_REVIEW", "CONFIRMED_REVIEW", "DISMISSED")


def haversine_distance_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in metres on the WGS84 sphere."""
    phi1 = math.radians(float(lat1))
    phi2 = math.radians(float(lat2))
    dphi = math.radians(float(lat2) - float(lat1))
    dlam = math.radians(float(lon2) - float(lon1))
    a = math.sin(dphi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * (math.sin(dlam / 2.0) ** 2)
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(max(0.0, 1.0 - a)))
    return float(EARTH_RADIUS_M * c)


def initial_bearing_deg(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Initial compass bearing [0, 360) from (lat1, lon1) to (lat2, lon2)."""
    phi1 = math.radians(float(lat1))
    phi2 = math.radians(float(lat2))
    dlam = math.radians(float(lon2) - float(lon1))
    y = math.sin(dlam) * math.cos(phi2)
    x = math.cos(phi1) * math.sin(phi2) - math.sin(phi1) * math.cos(phi2) * math.cos(dlam)
    return float((math.degrees(math.atan2(y, x)) + 360.0) % 360.0)


def heading_difference_deg(h1: float, h2: float) -> float:
    diff = abs(float(h1) - float(h2)) % 360.0
    return float(min(diff, 360.0 - diff))


def _normalize_gps_provenance(cand: Dict[str, Any]) -> Tuple[str, bool]:
    is_synth = bool(cand.get("is_synthetic", False))
    raw_prov = str(cand.get("gps_provenance") or "").strip().upper()
    if is_synth or raw_prov == "SYNTHETIC":
        return "SYNTHETIC", True
    if raw_prov in ("VERIFIED_FIELD_LOG", "USER_SUPPLIED_UNVERIFIED", "UNKNOWN"):
        return raw_prov, False
    # If provenance metadata is absent, mark conservatively as UNKNOWN rather than verified
    return "UNKNOWN", False


def _evaluate_candidate_eligibility(cand: Dict[str, Any], cfg: GeospatialConfig) -> Tuple[bool, List[str]]:
    reasons: List[str] = []
    if cand.get("lat") is None or cand.get("lon") is None:
        reasons.append("missing_coordinates")
    if cand.get("interpolation_status") == "UNAVAILABLE":
        reasons.append("gps_synchronization_unavailable")
    acc = cand.get("horizontal_accuracy_m")
    if acc is not None and float(acc) > cfg.max_horizontal_accuracy_m:
        reasons.append(f"poor_gps_accuracy_{float(acc):.1f}m_gt_{cfg.max_horizontal_accuracy_m:.1f}m")
    return len(reasons) == 0, reasons


def _pair_compatibility(
    a: Dict[str, Any],
    b: Dict[str, Any],
    cfg: GeospatialConfig,
) -> Dict[str, Any]:
    # Primary focus: cross-trip matching (distinct source inspections)
    if a["job_id"] == b["job_id"]:
        return {"compatible": False, "reason": "same_inspection_trip"}

    dist_m = haversine_distance_m(a["lat"], a["lon"], b["lat"], b["lon"])
    if dist_m > cfg.max_match_distance_m:
        return {"compatible": False, "dist_m": dist_m, "reason": "distance_exceeded"}

    flags: List[str] = []
    ha, hb = a.get("heading_deg"), b.get("heading_deg")
    if ha is not None and hb is not None:
        hdiff = heading_difference_deg(ha, hb)
        if hdiff > cfg.max_heading_diff_deg:
            return {"compatible": False, "dist_m": dist_m, "reason": "incompatible_heading"}
    else:
        flags.append("heading_unavailable")

    # Compare relative-depth profile ONLY when depth model & resolution match identically
    sa, sb = a.get("median_relative_score"), b.get("median_relative_score")
    if (
        sa is not None
        and sb is not None
        and a.get("depth_model_id")
        and a.get("depth_model_id") == b.get("depth_model_id")
        and a.get("depth_input_resolution") == b.get("depth_input_resolution")
    ):
        rel_diff = abs(float(sa) - float(sb)) / max(abs(float(sa)), abs(float(sb)), 1e-4)
        if rel_diff > cfg.relative_depth_diff_limit:
            flags.append(f"relative_depth_divergence_{rel_diff:.2f}")

    for side_name, c in (("a", a), ("b", b)):
        qi = c.get("quality_index")
        if qi is None:
            flags.append(f"missing_quality_{side_name}")
        elif float(qi) < cfg.min_quality_index_for_auto_match:
            flags.append(f"low_quality_{side_name}_{float(qi):.2f}")

    return {
        "compatible": True,
        "dist_m": round(dist_m, 3),
        "flags": flags,
        "needs_review": len(flags) > 0,
    }


def cluster_geospatial_candidates(
    candidates: Sequence[Dict[str, Any]],
    cfg: Optional[GeospatialConfig] = None,
) -> Dict[str, Any]:
    """
    Conservatively match cross-trip track observation candidates using Haversine distance
    with bounded-diameter (complete-linkage) clustering to prevent chain-merging.
    """
    cfg = cfg or GeospatialConfig()
    n = len(candidates)
    eligibility: List[Tuple[bool, List[str]]] = [_evaluate_candidate_eligibility(c, cfg) for c in candidates]

    # Collect eligible candidate pairs sorted by Haversine distance ascending
    pairs: List[Tuple[float, int, int, Dict[str, Any]]] = []
    for i in range(n):
        if not eligibility[i][0]:
            continue
        for j in range(i + 1, n):
            if not eligibility[j][0]:
                continue
            comp = _pair_compatibility(candidates[i], candidates[j], cfg)
            if comp["compatible"]:
                pairs.append((comp["dist_m"], i, j, comp))
    pairs.sort(key=lambda item: item[0])

    # Bounded-diameter clustering (prevents A-B-C chaining beyond max_cluster_diameter_m)
    clusters_idx: List[List[int]] = [[i] for i in range(n)]
    member_to_cluster: Dict[int, int] = {i: i for i in range(n)}
    pair_notes: Dict[Tuple[int, int], Dict[str, Any]] = {}

    for dist_m, i, j, comp in pairs:
        ci = member_to_cluster[i]
        cj = member_to_cluster[j]
        if ci == cj:
            continue
        grp_i = clusters_idx[ci]
        grp_j = clusters_idx[cj]

        # Enforce cross-trip & complete-linkage diameter constraint across ALL pairs
        can_merge = True
        max_cross_d = dist_m
        for u in grp_i:
            for v in grp_j:
                d_uv = haversine_distance_m(
                    candidates[u]["lat"], candidates[u]["lon"], candidates[v]["lat"], candidates[v]["lon"]
                )
                if d_uv > cfg.max_cluster_diameter_m:
                    can_merge = False
                    break
                if d_uv > max_cross_d:
                    max_cross_d = d_uv
            if not can_merge:
                break
        if not can_merge:
            continue

        merged = grp_i + grp_j
        clusters_idx[ci] = merged
        clusters_idx[cj] = []
        for idx_m in grp_j:
            member_to_cluster[idx_m] = ci
        pair_notes[(min(i, j), max(i, j))] = comp

    clusters_out: List[Dict[str, Any]] = []
    status_counts = {s: 0 for s in MATCH_STATUSES}
    rejected_count = 0

    for grp in clusters_idx:
        if not grp:
            continue
        members = [candidates[idx] for idx in grp]
        if len(members) == 1:
            idx0 = grp[0]
            elig_ok, elig_reasons = eligibility[idx0]
            if not elig_ok:
                rejected_count += 1
            m0 = members[0]
            prov0, synth0 = _normalize_gps_provenance(m0)
            status_counts["UNMATCHED"] += 1
            clusters_out.append(
                {
                    "cluster_id": str(uuid.uuid5(uuid.NAMESPACE_DNS, f"rv2:{m0['job_id']}:{m0['track_id']}:{m0.get('track_generation_id', 0)}")),
                    "match_status": "UNMATCHED",
                    "review_status": "PENDING_REVIEW",
                    "member_count": 1,
                    "distinct_trip_count": 1,
                    "centroid_lat": m0.get("lat"),
                    "centroid_lon": m0.get("lon"),
                    "diameter_m": 0.0 if m0.get("lat") is not None else None,
                    "match_reasons": elig_reasons if not elig_ok else ["no_cross_trip_match_within_threshold"],
                    "contains_synthetic": synth0,
                    "contains_unknown_provenance": prov0 == "UNKNOWN",
                    "cluster_provenance": prov0,
                    "members": [
                        {
                            **m0,
                            "is_synthetic": synth0,
                            "gps_provenance": prov0,
                            "distance_to_centroid_m": 0.0 if m0.get("lat") is not None else None,
                        }
                    ],
                }
            )
            continue

        # Multi-member candidate cluster
        c_lat = round(sum(float(m["lat"]) for m in members) / len(members), 7)
        c_lon = round(sum(float(m["lon"]) for m in members) / len(members), 7)
        max_pair_d = 0.0
        for a_i in range(len(members)):
            for b_i in range(a_i + 1, len(members)):
                d_ab = haversine_distance_m(members[a_i]["lat"], members[a_i]["lon"], members[b_i]["lat"], members[b_i]["lon"])
                if d_ab > max_pair_d:
                    max_pair_d = d_ab

        all_flags: List[str] = []
        needs_review = False
        for a_i in range(len(grp)):
            for b_i in range(a_i + 1, len(grp)):
                key = (min(grp[a_i], grp[b_i]), max(grp[a_i], grp[b_i]))
                note = pair_notes.get(key)
                if note:
                    if note.get("needs_review"):
                        needs_review = True
                    for fl in note.get("flags") or []:
                        if fl not in all_flags:
                            all_flags.append(fl)

        if max_pair_d > cfg.max_match_distance_m:
            needs_review = True
            all_flags.append(f"cluster_diameter_{max_pair_d:.1f}m_exceeds_pair_dist")

        member_provs = [_normalize_gps_provenance(m) for m in members]
        contains_synth = any(synth for _, synth in member_provs)
        contains_unknown = any(prov == "UNKNOWN" for prov, _ in member_provs)
        distinct_provs = sorted({prov for prov, _ in member_provs})
        if len(distinct_provs) == 1:
            cluster_prov = distinct_provs[0]
        elif contains_synth:
            cluster_prov = "MIXED_WITH_SYNTHETIC"
        elif contains_unknown:
            cluster_prov = "MIXED_WITH_UNKNOWN"
        else:
            cluster_prov = "MIXED_FIELD_PROVENANCE"

        if contains_unknown:
            needs_review = True
            if "unknown_gps_provenance" not in all_flags:
                all_flags.append("unknown_gps_provenance")
        if contains_synth and "contains_synthetic_gps_fixture" not in all_flags:
            all_flags.append("contains_synthetic_gps_fixture")

        match_status = "REVIEW_REQUIRED" if needs_review else "POTENTIAL_MATCH"
        status_counts[match_status] += 1
        trips = sorted({m["job_id"] for m in members})
        seed_str = "|".join(sorted(f"{m['job_id']}:{m['track_id']}" for m in members))

        enriched_members = []
        for m, (m_prov, m_synth) in zip(members, member_provs):
            d_cen = round(haversine_distance_m(m["lat"], m["lon"], c_lat, c_lon), 3)
            enriched_members.append(
                {
                    **m,
                    "is_synthetic": m_synth,
                    "gps_provenance": m_prov,
                    "distance_to_centroid_m": d_cen,
                }
            )

        clusters_out.append(
            {
                "cluster_id": str(uuid.uuid5(uuid.NAMESPACE_DNS, f"rv2cluster:{seed_str}")),
                "match_status": match_status,
                "review_status": "PENDING_REVIEW",
                "member_count": len(members),
                "distinct_trip_count": len(trips),
                "centroid_lat": c_lat,
                "centroid_lon": c_lon,
                "diameter_m": round(max_pair_d, 3),
                "match_reasons": all_flags or ["within_haversine_threshold_and_quality_bounds"],
                "contains_synthetic": contains_synth,
                "contains_unknown_provenance": contains_unknown,
                "cluster_provenance": cluster_prov,
                "members": enriched_members,
            }
        )

    # Sort multi-member clusters first, then unmatched
    clusters_out.sort(key=lambda c: (-c["member_count"], c["cluster_id"]))

    return {
        "clusters": clusters_out,
        "total_candidates": n,
        "total_clusters": len(clusters_out),
        "status_counts": status_counts,
        "ineligible_candidates_unmatched": rejected_count,
        "thresholds": {
            "max_match_distance_m": cfg.max_match_distance_m,
            "max_cluster_diameter_m": cfg.max_cluster_diameter_m,
            "max_horizontal_accuracy_m": cfg.max_horizontal_accuracy_m,
            "max_heading_diff_deg": cfg.max_heading_diff_deg,
            "min_quality_index_for_auto_match": cfg.min_quality_index_for_auto_match,
        },
        "scientific_disclaimer": (
            "Cross-trip clusters indicate spatial & diagnostic proximity of approximate "
            "vehicle GPS observations only; they are NOT ground-truth-verified unique physical potholes."
        ),
    }


def export_locations_geojson(
    locations: Sequence[Dict[str, Any]],
    route_points: Sequence[Dict[str, Any]] = (),
) -> Dict[str, Any]:
    """Build a standard RFC 7946 WGS84 GeoJSON FeatureCollection for actual synced records."""
    features: List[Dict[str, Any]] = []
    if len(route_points) >= 2:
        coords = [[p["lon"], p["lat"]] for p in route_points if p.get("lat") is not None and p.get("lon") is not None]
        if len(coords) >= 2:
            features.append(
                {
                    "type": "Feature",
                    "geometry": {"type": "LineString", "coordinates": coords},
                    "properties": {
                        "feature_kind": "approximate_inspection_route",
                        "provenance": "EXTERNAL_GPS_TELEMETRY",
                        "point_count": len(coords),
                    },
                }
            )
    for loc in locations:
        if loc.get("lat") is None or loc.get("lon") is None:
            continue
        features.append(
            {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [loc["lon"], loc["lat"]]},
                "properties": {
                    "feature_kind": "approximate_track_observation_location",
                    "job_id": loc.get("job_id"),
                    "track_id": loc.get("track_id"),
                    "observation_timestamp_sec": loc.get("observation_timestamp_sec"),
                    "horizontal_accuracy_m": loc.get("horizontal_accuracy_m"),
                    "interpolation_status": loc.get("interpolation_status"),
                    "sync_quality": loc.get("sync_quality"),
                    "provenance_label": loc.get("provenance_label", "APPROXIMATE_VEHICLE_GPS"),
                    "gps_provenance": loc.get("gps_provenance", "UNKNOWN"),
                    "is_synthetic": bool(loc.get("is_synthetic", False)),
                    "quality_category": loc.get("quality_category"),
                    "quality_index": loc.get("quality_index"),
                },
            }
        )
    return {"type": "FeatureCollection", "features": features}
