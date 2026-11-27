import React, { useEffect, useState, useRef, useMemo } from 'react';
import { getGeometrySnapshots, getGeometrySnapshot } from '../api/client';
import { RotateCcw, ZoomIn, ZoomOut, Box, Layers, Image as ImageIcon, AlertTriangle } from 'lucide-react';
import InspectionSelector from '../components/InspectionSelector';
import {
  StatePanel,
  ScientificDisclaimer,
  QualityBreakdownCard,
} from '../components/StateBanner';
import { isValidUuid, formatNullable } from '../utils/formatters';

function residualToColor(val, vMin, vMax) {
  if (val === null || val === undefined || Number.isNaN(val)) {
    return '#111827';
  }
  const span = Math.max(vMax - vMin, 1e-5);
  const t = Math.max(0, Math.min(1, (val - vMin) / span));
  // Dark-slate blue (protrusion / low) -> teal -> bright cyan -> warm amber (deeper cavity)
  if (t < 0.35) {
    const u = t / 0.35;
    const r = Math.round(30 + u * (14 - 30));
    const g = Math.round(58 + u * (116 - 58));
    const b = Math.round(138 + u * (144 - 138));
    return `rgb(${r},${g},${b})`;
  }
  if (t < 0.7) {
    const u = (t - 0.35) / 0.35;
    const r = Math.round(14 + u * (34 - 14));
    const g = Math.round(116 + u * (211 - 116));
    const b = Math.round(144 + u * (238 - 144));
    return `rgb(${r},${g},${b})`;
  }
  const u = (t - 0.7) / 0.3;
  const r = Math.round(34 + u * (245 - 34));
  const g = Math.round(211 + u * (158 - 211));
  const b = Math.round(238 + u * (11 - 238));
  return `rgb(${r},${g},${b})`;
}

function buildCrossSection(grid, validGrid, rowIdx, colIdx) {
  const gh = grid?.length || 0;
  const gw = gh > 0 ? (grid[0]?.length || 0) : 0;
  if (gh === 0 || gw === 0) return { horizontal: [], vertical: [] };
  const r = Math.max(0, Math.min(gh - 1, rowIdx));
  const c = Math.max(0, Math.min(gw - 1, colIdx));
  const horizontal = [];
  for (let x = 0; x < gw; x += 1) {
    const isOk = Boolean(validGrid?.[r]?.[x]) && grid[r][x] !== null && grid[r][x] !== undefined;
    horizontal.push({ pos: (x + 0.5) / gw, val: isOk ? Number(grid[r][x]) : null, valid: isOk });
  }
  const vertical = [];
  for (let y = 0; y < gh; y += 1) {
    const isOk = Boolean(validGrid?.[y]?.[c]) && grid[y][c] !== null && grid[y][c] !== undefined;
    vertical.push({ pos: (y + 0.5) / gh, val: isOk ? Number(grid[y][c]) : null, valid: isOk });
  }
  return { horizontal, vertical };
}

function percentileSorted(sortedArr, p) {
  if (!sortedArr.length) return null;
  if (sortedArr.length === 1) return sortedArr[0];
  const idx = (p / 100) * (sortedArr.length - 1);
  const lo = Math.floor(idx);
  const hi = Math.ceil(idx);
  if (lo === hi) return sortedArr[lo];
  return sortedArr[lo] + (sortedArr[hi] - sortedArr[lo]) * (idx - lo);
}

function medianOf(arr) {
  if (!arr.length) return null;
  const s = [...arr].sort((a, b) => a - b);
  return percentileSorted(s, 50);
}

function computeSnapshotStats(snapshot) {
  if (!snapshot) return null;
  const grid = snapshot.signed_residual_grid || [];
  const valid = snapshot.validity_mask_grid || [];
  const cav = snapshot.cavity_mask_grid || [];
  const ref = snapshot.reference_mask_grid || [];
  const gh = grid.length;
  const gw = gh > 0 ? grid[0]?.length || 0 : 0;

  const allVals = [];
  const cavVals = [];
  const refVals = [];
  const nonCavVals = [];

  for (let r = 0; r < gh; r += 1) {
    for (let c = 0; c < gw; c += 1) {
      const raw = grid[r]?.[c];
      if (!valid[r]?.[c] || raw === null || raw === undefined) continue;
      const val = Number(raw);
      if (Number.isNaN(val)) continue;
      allVals.push(val);
      if (cav[r]?.[c]) {
        cavVals.push(val);
      } else {
        nonCavVals.push(val);
      }
      if (ref[r]?.[c]) {
        refVals.push(val);
      }
    }
  }

  const totalCount = gh * gw;
  if (allVals.length === 0) {
    return {
      validCount: 0,
      totalCount,
      min: null,
      p5: null,
      median: null,
      p95: null,
      max: null,
      std: null,
      span: 0,
      cavityCount: 0,
      cavityMedian: null,
      refCount: 0,
      refMedian: null,
      nonCavityMedian: null,
    };
  }

  const sorted = [...allVals].sort((a, b) => a - b);
  const mean = allVals.reduce((acc, v) => acc + v, 0) / allVals.length;
  const variance = allVals.reduce((acc, v) => acc + (v - mean) ** 2, 0) / allVals.length;
  const min = sorted[0];
  const max = sorted[sorted.length - 1];

  return {
    validCount: allVals.length,
    totalCount,
    min,
    p5: percentileSorted(sorted, 5),
    median: percentileSorted(sorted, 50),
    p95: percentileSorted(sorted, 95),
    max,
    std: Math.sqrt(variance),
    span: max - min,
    cavityCount: cavVals.length,
    cavityMedian: medianOf(cavVals),
    refCount: refVals.length,
    refMedian: medianOf(refVals),
    nonCavityMedian: medianOf(nonCavVals),
  };
}

function ColorLegendBar({ vMin, vMax, scaleMode }) {
  const span = Math.max(vMax - vMin, 1e-6);
  const mid = (vMin + vMax) * 0.5;
  const zeroPct = vMin <= 0 && vMax >= 0 ? Math.max(0, Math.min(100, ((0 - vMin) / span) * 100)) : null;

  return (
    <div
      style={{
        marginTop: 12,
        padding: '10px 12px',
        background: '#080e1a',
        border: '1px solid var(--rv-border)',
        borderRadius: 6,
      }}
    >
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 6, gap: 8, flexWrap: 'wrap' }}>
        <span style={{ fontSize: 11, fontWeight: 600, color: 'var(--rv-text-primary)' }}>
          Residual Color Scale ({scaleMode === 'global' ? 'Global Job Bounds' : 'Local Snapshot Min/Max'})
        </span>
        <span className="rv-mono" style={{ fontSize: 11, color: 'var(--rv-text-secondary)' }}>
          Uncalibrated relative residual (+ = depression below road plane, − = above plane)
        </span>
      </div>
      <div style={{ position: 'relative', height: 14, borderRadius: 4, border: '1px solid rgba(148,163,184,0.28)', background: 'linear-gradient(90deg, rgb(30,58,138) 0%, rgb(14,116,144) 35%, rgb(34,211,238) 70%, rgb(245,158,11) 100%)' }}>
        {zeroPct !== null && (
          <div
            title="Zero residual (fitted road reference plane)"
            style={{
              position: 'absolute',
              left: `${zeroPct.toFixed(1)}%`,
              top: -2,
              bottom: -2,
              width: 2,
              background: '#f8fafc',
              boxShadow: '0 0 3px rgba(0,0,0,0.9)',
            }}
          />
        )}
      </div>
      <div className="rv-mono" style={{ display: 'flex', justifyContent: 'space-between', marginTop: 5, fontSize: 11, color: 'var(--rv-text-secondary)' }}>
        <span>{vMin.toFixed(4)}</span>
        {zeroPct !== null ? (
          <span style={{ color: '#e2e8f0' }}>0.0000 (ref plane) · mid {mid.toFixed(4)}</span>
        ) : (
          <span>mid {mid.toFixed(4)}</span>
        )}
        <span>{vMax.toFixed(4)}</span>
      </div>
    </div>
  );
}

function CrossSectionChart({ points, title, axisLabel, vMin, vMax }) {
  const W = 440;
  const H = 165;
  const PAD = { l: 48, r: 14, t: 18, b: 26 };
  const plotW = W - PAD.l - PAD.r;
  const plotH = H - PAD.t - PAD.b;

  const span = Math.max(vMax - vMin, 1e-4);
  const toY = (val) => PAD.t + plotH * (1 - (val - vMin) / span);
  const zeroY = Math.max(PAD.t, Math.min(PAD.t + plotH, toY(0)));

  // Split polyline cleanly at invalid gaps (never interpolate across missing values)
  const segments = [];
  const invalidMarkers = [];
  let current = [];
  const stepW = points.length > 0 ? plotW / points.length : 4;
  for (let idx = 0; idx < points.length; idx += 1) {
    const pt = points[idx];
    if (pt.valid && pt.val !== null) {
      const sx = PAD.l + pt.pos * plotW;
      const sy = Math.max(PAD.t, Math.min(PAD.t + plotH, toY(pt.val)));
      current.push(`${sx.toFixed(1)},${sy.toFixed(1)}`);
    } else {
      invalidMarkers.push({ x: PAD.l + (idx / Math.max(1, points.length)) * plotW, w: Math.max(2, stepW) });
      if (current.length > 0) {
        segments.push(current);
        current = [];
      }
    }
  }
  if (current.length > 0) segments.push(current);

  return (
    <div className="rv-card" style={{ padding: 14 }}>
      <div className="rv-card-header" style={{ marginBottom: 8 }}>
        <h4 className="rv-card-title" style={{ fontSize: 13 }}>{title}</h4>
        <span className="rv-card-meta">
          {axisLabel}
          {invalidMarkers.length > 0 ? ` (${invalidMarkers.length} invalid omitted)` : ''}
        </span>
      </div>
      <svg viewBox={`0 0 ${W} ${H}`} style={{ width: '100%', height: 165, background: '#070b13', borderRadius: 6 }}>
        {invalidMarkers.map((m, idx) => (
          <rect
            key={`gap-${idx}`}
            x={m.x.toFixed(1)}
            y={PAD.t}
            width={m.w.toFixed(1)}
            height={plotH}
            fill="rgba(244, 63, 94, 0.14)"
          />
        ))}
        <line x1={PAD.l} y1={zeroY} x2={W - PAD.r} y2={zeroY} stroke="#475569" strokeDasharray="4 4" strokeWidth="1" />
        <text x={4} y={zeroY + 3} fill="#94a3b8" fontSize="9" fontFamily="monospace">0.0</text>
        <text x={4} y={PAD.t + 8} fill="#94a3b8" fontSize="9" fontFamily="monospace">{vMax.toFixed(4)}</text>
        <text x={4} y={PAD.t + plotH} fill="#94a3b8" fontSize="9" fontFamily="monospace">{vMin.toFixed(4)}</text>
        {segments.map((seg, i) =>
          seg.length > 1 ? (
            <polyline key={i} fill="none" stroke="#22d3ee" strokeWidth="2" points={seg.join(' ')} />
          ) : (
            <circle key={i} cx={seg[0].split(',')[0]} cy={seg[0].split(',')[1]} r="2.5" fill="#22d3ee" />
          )
        )}
      </svg>
    </div>
  );
}

export default function Relative3DAnalysis({ selectedJobId, onSelectJob, onNavigate }) {
  const [manifest, setManifest] = useState(null);
  const [selectedTrack, setSelectedTrack] = useState('');
  const [selectedSnapshotId, setSelectedSnapshotId] = useState('');
  const [snapshot, setSnapshot] = useState(null);

  const [loadingList, setLoadingList] = useState(false);
  const [loadingSnap, setLoadingSnap] = useState(false);
  const [error, setError] = useState(null);

  // Interactive 3D & profile controls
  const [yaw, setYaw] = useState(-0.62);
  const [pitch, setPitch] = useState(0.58);
  const [zoom, setZoom] = useState(1.0);
  const [renderMode, setRenderMode] = useState('hybrid'); // surface | wireframe | hybrid
  const [exaggeration, setExaggeration] = useState(2.2);
  const [scaleMode, setScaleMode] = useState('global'); // global | snapshot
  const [sliceRow, setSliceRow] = useState(0);
  const [sliceCol, setSliceCol] = useState(0);

  const canvas3dRef = useRef(null);
  const canvasHeatRef = useRef(null);
  const dragRef = useRef({ active: false, x: 0, y: 0 });

  useEffect(() => {
    let mounted = true;
    if (!selectedJobId || !isValidUuid(selectedJobId)) {
      setManifest(null);
      setSnapshot(null);
      return undefined;
    }
    setLoadingList(true);
    setError(null);
    getGeometrySnapshots(selectedJobId)
      .then((resp) => {
        if (!mounted) return;
        setManifest(resp);
        const list = resp?.snapshots || [];
        if (list.length > 0) {
          setSelectedTrack('');
          setSelectedSnapshotId(list[0].snapshot_id);
        } else {
          setSelectedSnapshotId('');
          setSnapshot(null);
        }
      })
      .catch((err) => {
        if (!mounted) return;
        setManifest(null);
        setSnapshot(null);
        setError(err);
      })
      .finally(() => {
        if (mounted) setLoadingList(false);
      });
    return () => {
      mounted = false;
    };
  }, [selectedJobId]);

  const filteredSnapshots = useMemo(() => {
    const list = manifest?.snapshots || [];
    if (!selectedTrack) return list;
    return list.filter((s) => String(s.track_id) === String(selectedTrack));
  }, [manifest, selectedTrack]);

  useEffect(() => {
    if (filteredSnapshots.length > 0 && !filteredSnapshots.some((s) => s.snapshot_id === selectedSnapshotId)) {
      setSelectedSnapshotId(filteredSnapshots[0].snapshot_id);
    }
  }, [filteredSnapshots, selectedSnapshotId]);

  useEffect(() => {
    let mounted = true;
    if (!selectedJobId || !selectedSnapshotId) {
      setSnapshot(null);
      return undefined;
    }
    setLoadingSnap(true);
    getGeometrySnapshot(selectedJobId, selectedSnapshotId)
      .then((snap) => {
        if (!mounted) return;
        setSnapshot(snap);
        const gh = snap?.grid_shape?.[0] || 1;
        const gw = snap?.grid_shape?.[1] || 1;
        setSliceRow(Math.floor(gh / 2));
        setSliceCol(Math.floor(gw / 2));
      })
      .catch((err) => {
        if (mounted) setError(err);
      })
      .finally(() => {
        if (mounted) setLoadingSnap(false);
      });
    return () => {
      mounted = false;
    };
  }, [selectedJobId, selectedSnapshotId]);

  const colorBounds = useMemo(() => {
    if (!snapshot) return { vMin: -0.05, vMax: 0.1 };
    const useGlobal =
      scaleMode === 'global' &&
      snapshot.global_min_residual !== null &&
      snapshot.global_min_residual !== undefined &&
      snapshot.global_max_residual !== null &&
      snapshot.global_max_residual !== undefined;
    const rawMin = useGlobal ? snapshot.global_min_residual : snapshot.min_residual;
    const rawMax = useGlobal ? snapshot.global_max_residual : snapshot.max_residual;
    const vMin = rawMin !== null && rawMin !== undefined ? Number(rawMin) : -0.05;
    const vMax = rawMax !== null && rawMax !== undefined ? Number(rawMax) : 0.1;
    return vMax > vMin ? { vMin, vMax } : { vMin: vMin - 0.01, vMax: vMin + 0.05 };
  }, [snapshot, scaleMode]);

  // Render 2.5D Surface Canvas
  useEffect(() => {
    const canvas = canvas3dRef.current;
    if (!canvas || !snapshot) return;
    const ctx = canvas.getContext('2d');
    const W = (canvas.width = canvas.clientWidth || 640);
    const H = (canvas.height = canvas.clientHeight || 390);
    ctx.clearRect(0, 0, W, H);

    const grid = snapshot.signed_residual_grid || [];
    const valid = snapshot.validity_mask_grid || [];
    const gh = grid.length;
    const gw = gh > 0 ? grid[0].length : 0;
    if (gh < 2 || gw < 2) return;

    const { vMin, vMax } = colorBounds;
    const resSpan = Math.max(vMax - vMin, 1e-4);

    const cy = Math.cos(yaw);
    const sy = Math.sin(yaw);
    const cp = Math.cos(pitch);
    const sp = Math.sin(pitch);
    const scalePx = Math.min(W, H) * 0.62 * zoom;

    const project = (nx, ny, nz) => {
      // Orbit rotation around center (0,0,0)
      const x1 = nx * cy - ny * sy;
      const y1 = nx * sy + ny * cy;
      const z2 = nz * cp - y1 * sp;
      const y2 = nz * sp + y1 * cp;
      const persp = 3.2 / Math.max(1.2, 3.2 - y2 * 0.65);
      return {
        sx: W * 0.5 + x1 * scalePx * persp,
        sy: H * 0.54 - z2 * scalePx * persp,
        depth: y2,
      };
    };

    // Draw reference plane baseline frame (z = 0)
    const corners = [
      project(-0.5, -0.5, 0),
      project(0.5, -0.5, 0),
      project(0.5, 0.5, 0),
      project(-0.5, 0.5, 0),
    ];
    ctx.strokeStyle = 'rgba(148, 163, 184, 0.35)';
    ctx.lineWidth = 1;
    ctx.setLineDash([4, 4]);
    ctx.beginPath();
    ctx.moveTo(corners[0].sx, corners[0].sy);
    for (let i = 1; i < 4; i += 1) ctx.lineTo(corners[i].sx, corners[i].sy);
    ctx.closePath();
    ctx.stroke();
    ctx.setLineDash([]);

    // Precompute projected grid vertices (null where invalid!)
    const pts = [];
    for (let r = 0; r < gh; r += 1) {
      const rowPts = [];
      for (let c = 0; c < gw; c += 1) {
        if (!valid[r]?.[c] || grid[r][c] === null || grid[r][c] === undefined) {
          rowPts.push(null);
          continue;
        }
        const nx = (c / (gw - 1)) - 0.5;
        const ny = (r / (gh - 1)) - 0.5;
        // Positive residual = depressed cavity downward (-Z) relative to road plane
        const val = Number(grid[r][c]);
        const nz = -(val / resSpan) * 0.2 * exaggeration;
        rowPts.push({ ...project(nx, ny, nz), val });
      }
      pts.push(rowPts);
    }

    // Collect valid quads and painter-sort back-to-front (smallest y2 = farthest drawn first)
    const quads = [];
    for (let r = 0; r < gh - 1; r += 1) {
      for (let c = 0; c < gw - 1; c += 1) {
        const p00 = pts[r][c];
        const p10 = pts[r][c + 1];
        const p11 = pts[r + 1][c + 1];
        const p01 = pts[r + 1][c];
        if (!p00 || !p10 || !p11 || !p01) continue;
        const avgDepth = (p00.depth + p10.depth + p11.depth + p01.depth) * 0.25;
        const avgVal = (p00.val + p10.val + p11.val + p01.val) * 0.25;
        quads.push({ p00, p10, p11, p01, avgDepth, avgVal });
      }
    }
    quads.sort((a, b) => a.avgDepth - b.avgDepth);

    for (const q of quads) {
      ctx.beginPath();
      ctx.moveTo(q.p00.sx, q.p00.sy);
      ctx.lineTo(q.p10.sx, q.p10.sy);
      ctx.lineTo(q.p11.sx, q.p11.sy);
      ctx.lineTo(q.p01.sx, q.p01.sy);
      ctx.closePath();
      if (renderMode === 'surface' || renderMode === 'hybrid') {
        ctx.fillStyle = residualToColor(q.avgVal, vMin, vMax);
        ctx.fill();
      }
      if (renderMode === 'wireframe' || renderMode === 'hybrid') {
        ctx.strokeStyle = renderMode === 'wireframe' ? '#22d3ee' : 'rgba(15, 23, 42, 0.55)';
        ctx.lineWidth = renderMode === 'wireframe' ? 0.9 : 0.6;
        ctx.stroke();
      }
    }
  }, [snapshot, yaw, pitch, zoom, renderMode, exaggeration, colorBounds]);

  // Render 2D Heatmap Canvas
  useEffect(() => {
    const canvas = canvasHeatRef.current;
    if (!canvas || !snapshot) return;
    const ctx = canvas.getContext('2d');
    const W = (canvas.width = canvas.clientWidth || 480);
    const H = (canvas.height = canvas.clientHeight || 320);
    ctx.clearRect(0, 0, W, H);

    const grid = snapshot.signed_residual_grid || [];
    const valid = snapshot.validity_mask_grid || [];
    const cavity = snapshot.cavity_mask_grid || [];
    const refMask = snapshot.reference_mask_grid || [];
    const gh = grid.length;
    const gw = gh > 0 ? grid[0].length : 0;
    if (gh === 0 || gw === 0) return;

    const cw = W / gw;
    const ch = H / gh;
    const { vMin, vMax } = colorBounds;

    for (let r = 0; r < gh; r += 1) {
      for (let c = 0; c < gw; c += 1) {
        const x = c * cw;
        const y = r * ch;
        if (!valid[r]?.[c] || grid[r][c] === null || grid[r][c] === undefined) {
          ctx.fillStyle = '#0f172a';
          ctx.fillRect(x, y, cw + 0.5, ch + 0.5);
          ctx.strokeStyle = 'rgba(244, 63, 94, 0.45)';
          ctx.beginPath();
          ctx.moveTo(x, y);
          ctx.lineTo(x + cw, y + ch);
          ctx.stroke();
        } else {
          ctx.fillStyle = residualToColor(Number(grid[r][c]), vMin, vMax);
          ctx.fillRect(x, y, cw + 0.5, ch + 0.5);
        }
        if (refMask[r]?.[c]) {
          ctx.fillStyle = 'rgba(16, 185, 129, 0.18)';
          ctx.fillRect(x, y, cw, ch);
        }
        if (cavity[r]?.[c]) {
          ctx.strokeStyle = 'rgba(255, 255, 255, 0.5)';
          ctx.lineWidth = 0.8;
          ctx.strokeRect(x + 0.5, y + 0.5, cw - 1, ch - 1);
        }
      }
    }

    // Detection bounding box inside ROI
    const bnorm = snapshot.bbox_in_roi_norm;
    if (Array.isArray(bnorm) && bnorm.length === 4) {
      ctx.strokeStyle = '#f8fafc';
      ctx.lineWidth = 2;
      ctx.setLineDash([5, 3]);
      ctx.strokeRect(
        bnorm[0] * W,
        bnorm[1] * H,
        (bnorm[2] - bnorm[0]) * W,
        (bnorm[3] - bnorm[1]) * H
      );
      ctx.setLineDash([]);
    }

    // Crosshair lines for selected cross-sections
    ctx.strokeStyle = 'rgba(34, 211, 238, 0.85)';
    ctx.lineWidth = 1.2;
    const hy = ((sliceRow + 0.5) / gh) * H;
    const vx = ((sliceCol + 0.5) / gw) * W;
    ctx.beginPath();
    ctx.moveTo(0, hy);
    ctx.lineTo(W, hy);
    ctx.moveTo(vx, 0);
    ctx.lineTo(vx, H);
    ctx.stroke();
  }, [snapshot, colorBounds, sliceRow, sliceCol]);

  const handleMouseDown = (e) => {
    dragRef.current = { active: true, x: e.clientX, y: e.clientY };
  };
  const handleMouseMove = (e) => {
    if (!dragRef.current.active) return;
    const dx = e.clientX - dragRef.current.x;
    const dy = e.clientY - dragRef.current.y;
    dragRef.current = { active: true, x: e.clientX, y: e.clientY };
    setYaw((v) => v + dx * 0.01);
    setPitch((v) => Math.max(0.1, Math.min(1.45, v + dy * 0.01)));
  };
  const handleMouseUp = () => {
    dragRef.current.active = false;
  };

  // Support non-passive mouse wheel zoom on the 3D canvas without altering residual values
  useEffect(() => {
    const canvas = canvas3dRef.current;
    if (!canvas) return undefined;
    const onWheel = (e) => {
      e.preventDefault();
      const delta = e.deltaY < 0 ? 0.15 : -0.15;
      setZoom((z) => Math.max(0.4, Math.min(4.0, Number((z + delta).toFixed(2)))));
    };
    canvas.addEventListener('wheel', onWheel, { passive: false });
    return () => canvas.removeEventListener('wheel', onWheel);
  }, [snapshot]);

  const snapshotStats = useMemo(() => computeSnapshotStats(snapshot), [snapshot]);

  const reliefDiagnostics = useMemo(() => {
    if (!snapshot || !snapshotStats || snapshotStats.validCount === 0) return null;
    const p95MinP5 = (snapshotStats.p95 ?? 0) - (snapshotStats.p5 ?? 0);
    const isNearPlanar = snapshotStats.span < 0.05 || p95MinP5 < 0.035;
    const hasGlobal =
      snapshot.global_min_residual !== null &&
      snapshot.global_min_residual !== undefined &&
      snapshot.global_max_residual !== null &&
      snapshot.global_max_residual !== undefined;
    const globalSpan = hasGlobal
      ? Math.max(Number(snapshot.global_max_residual) - Number(snapshot.global_min_residual), 1e-6)
      : null;
    const globalCoveragePct = globalSpan ? (snapshotStats.span / globalSpan) * 100 : null;
    const isGloballyCompressed =
      scaleMode === 'global' && globalCoveragePct !== null && globalCoveragePct < 20;
    return {
      isNearPlanar,
      p95MinP5,
      globalSpan,
      globalCoveragePct,
      isGloballyCompressed,
    };
  }, [snapshot, snapshotStats, scaleMode]);

  const trackIds = useMemo(() => {
    const ids = new Set((manifest?.snapshots || []).map((s) => s.track_id));
    return Array.from(ids).sort((a, b) => a - b);
  }, [manifest]);

  const profiles = useMemo(() => {
    if (!snapshot) return { horizontal: [], vertical: [] };
    return buildCrossSection(
      snapshot.signed_residual_grid,
      snapshot.validity_mask_grid,
      sliceRow,
      sliceCol
    );
  }, [snapshot, sliceRow, sliceCol]);

  const gh = snapshot?.grid_shape?.[0] || 0;
  const gw = snapshot?.grid_shape?.[1] || 0;

  return (
    <div className="rv-page">
      <div className="rv-page-header">
        <div>
          <h2 className="rv-page-title">Interactive Relative 3D Analysis</h2>
          <p className="rv-page-subtitle">
            Inspect 2.5D local reference-plane residual surfaces, masks, and cross-sectional profiles from FRESH_VALID snapshots.
          </p>
        </div>
      </div>

      <InspectionSelector
        selectedJobId={isValidUuid(selectedJobId) ? selectedJobId : ''}
        onSelectJob={onSelectJob}
        onNavigate={onNavigate}
        label="Completed Inspection"
      />

      {!selectedJobId ? (
        <StatePanel
          variant="empty"
          title="Select an inspection"
          message="Choose an inspection with geometry recording enabled to inspect 2.5D relative cavity surfaces."
        />
      ) : loadingList ? (
        <StatePanel variant="loading" title="Loading geometry manifest" message="Checking available relative 2.5D snapshots…" />
      ) : error ? (
        <StatePanel
          variant={error.status === 409 ? 'unavailable' : error.isNetworkError ? 'disconnected' : 'error'}
          title={error.status === 409 ? 'Geometry unavailable before completion' : 'Failed to load geometry snapshots'}
          message={error.message}
        />
      ) : manifest && !manifest.geometry_available ? (
        <StatePanel
          variant="unavailable"
          title="No relative geometry recorded for this inspection"
          message={
            manifest.reason ||
            'Older inspections or default runs without enable_geometry=true do not contain 2.5D surface snapshots.'
          }
          action={
            <button
              type="button"
              className="rv-btn rv-btn-primary rv-btn-sm"
              onClick={() => onNavigate('new', undefined, { enableDepth: true, enableGeometry: true })}
            >
              New Inspection with Geometry Recording
            </button>
          }
        />
      ) : (manifest?.snapshots?.length || 0) === 0 ? (
        <StatePanel
          variant="empty"
          title="Zero valid geometry snapshots captured"
          message="Geometry recording was enabled, but no FRESH_VALID observations passed quality gates to produce a snapshot."
        />
      ) : (
        <>
          <section className="rv-card">
            <div className="rv-filter-bar" style={{ marginBottom: 0 }}>
              <div className="rv-field">
                <label className="rv-field-label" htmlFor="rv-3d-track">Track</label>
                <select
                  id="rv-3d-track"
                  className="rv-select"
                  value={selectedTrack}
                  onChange={(e) => setSelectedTrack(e.target.value)}
                >
                  <option value="">All Tracks ({manifest.snapshots.length} snapshots)</option>
                  {trackIds.map((tid) => (
                    <option key={tid} value={tid}>Track #{tid}</option>
                  ))}
                </select>
              </div>

              <div className="rv-field" style={{ flex: 1, minWidth: 240 }}>
                <label className="rv-field-label" htmlFor="rv-3d-snap">Geometry Snapshot (FRESH_VALID)</label>
                <select
                  id="rv-3d-snap"
                  className="rv-select"
                  value={selectedSnapshotId}
                  onChange={(e) => setSelectedSnapshotId(e.target.value)}
                >
                  {filteredSnapshots.map((s) => (
                    <option key={s.snapshot_id} value={s.snapshot_id}>
                      Track #{s.track_id} · Frame {s.frame_idx} ({s.timestamp_sec}s) · {s.quality_category || 'N/A'}
                    </option>
                  ))}
                </select>
              </div>

              <div className="rv-field">
                <label className="rv-field-label" htmlFor="rv-3d-mode">Render Mode</label>
                <select
                  id="rv-3d-mode"
                  className="rv-select"
                  value={renderMode}
                  onChange={(e) => setRenderMode(e.target.value)}
                >
                  <option value="hybrid">Surface + Wireframe</option>
                  <option value="surface">Surface Only</option>
                  <option value="wireframe">Wireframe Only</option>
                </select>
              </div>

              <div className="rv-field">
                <label className="rv-field-label" htmlFor="rv-3d-scale">Color Scale</label>
                <select
                  id="rv-3d-scale"
                  className="rv-select"
                  value={scaleMode}
                  onChange={(e) => setScaleMode(e.target.value)}
                >
                  <option value="global">Global Job Scale (Comparable)</option>
                  <option value="snapshot">Local Snapshot Min/Max</option>
                </select>
              </div>
            </div>
          </section>

          {loadingSnap && !snapshot ? (
            <StatePanel variant="loading" title="Loading surface grid" message="Fetching verified residual snapshot…" />
          ) : snapshot && (
            <>
              {(reliefDiagnostics?.isNearPlanar || reliefDiagnostics?.isGloballyCompressed) && (
                <div
                  role="alert"
                  style={{
                    display: 'flex',
                    alignItems: 'flex-start',
                    justifyContent: 'space-between',
                    gap: 12,
                    flexWrap: 'wrap',
                    padding: '10px 14px',
                    borderRadius: 8,
                    border: '1px solid rgba(251, 191, 36, 0.45)',
                    background: 'rgba(120, 53, 15, 0.22)',
                    color: '#fde68a',
                    fontSize: 12,
                  }}
                >
                  <div style={{ display: 'flex', gap: 8, alignItems: 'flex-start', flex: 1, minWidth: 260 }}>
                    <AlertTriangle size={16} style={{ marginTop: 2, flexShrink: 0, color: '#fbbf24' }} />
                    <div>
                      {reliefDiagnostics.isNearPlanar && (
                        <div>
                          <strong>Low-relief / near-planar residual surface:</strong> Span is{' '}
                          <span className="rv-mono">{snapshotStats.span.toFixed(5)}</span> (P95−P5{' '}
                          <span className="rv-mono">{reliefDiagnostics.p95MinP5.toFixed(5)}</span>, std{' '}
                          <span className="rv-mono">{snapshotStats.std?.toFixed(5)}</span>, cavity median{' '}
                          <span className="rv-mono">{formatNullable(snapshotStats.cavityMedian, 5)}</span>). The
                          underlying inverse-depth model residuals are nearly flat; no depression is manufactured.
                        </div>
                      )}
                      {reliefDiagnostics.isGloballyCompressed && (
                        <div style={{ marginTop: reliefDiagnostics.isNearPlanar ? 4 : 0 }}>
                          <strong>Global scale compression:</strong> This snapshot covers only{' '}
                          <span className="rv-mono">{reliefDiagnostics.globalCoveragePct.toFixed(1)}%</span> of the job
                          global range [
                          <span className="rv-mono">{Number(snapshot.global_min_residual).toFixed(4)}</span> ..{' '}
                          <span className="rv-mono">{Number(snapshot.global_max_residual).toFixed(4)}</span>], flattening
                          visual color &amp; Z contrast.
                        </div>
                      )}
                    </div>
                  </div>
                  {reliefDiagnostics.isGloballyCompressed && (
                    <button
                      type="button"
                      className="rv-btn rv-btn-secondary rv-btn-sm"
                      onClick={() => setScaleMode('snapshot')}
                    >
                      Switch to Local Snapshot Scale
                    </button>
                  )}
                </div>
              )}

              <div className="rv-grid-2">
                <section className="rv-card">
                  <div className="rv-card-header">
                    <h3 className="rv-card-title">
                      <Box size={16} className="rv-icon-cyan" /> 2.5D Relative Surface (Drag to Orbit · Scroll to Zoom)
                    </h3>
                    <div className="rv-header-actions">
                      <span className="rv-mono" style={{ fontSize: 11, color: 'var(--rv-text-secondary)', marginRight: 4 }}>
                        {zoom.toFixed(1)}x
                      </span>
                      <button type="button" className="rv-btn rv-btn-ghost rv-btn-sm" onClick={() => setZoom((z) => Math.min(4.0, Number((z + 0.2).toFixed(2))))} title="Zoom In (up to 4.0x)">
                        <ZoomIn size={14} />
                      </button>
                      <button type="button" className="rv-btn rv-btn-ghost rv-btn-sm" onClick={() => setZoom((z) => Math.max(0.4, Number((z - 0.2).toFixed(2))))} title="Zoom Out (down to 0.4x)">
                        <ZoomOut size={14} />
                      </button>
                      <button
                        type="button"
                        className="rv-btn rv-btn-ghost rv-btn-sm"
                        onClick={() => {
                          setYaw(-0.62);
                          setPitch(0.58);
                          setZoom(1.0);
                          setExaggeration(2.2);
                        }}
                      >
                        <RotateCcw size={14} /> Reset
                      </button>
                    </div>
                  </div>

                  <div className="rv-canvas-box">
                    <canvas
                      ref={canvas3dRef}
                      className="rv-canvas-3d"
                      onMouseDown={handleMouseDown}
                      onMouseMove={handleMouseMove}
                      onMouseUp={handleMouseUp}
                      onMouseLeave={handleMouseUp}
                    />
                  </div>

                  <div className="rv-filter-bar" style={{ marginTop: 12, marginBottom: 0 }}>
                    <div className="rv-field" style={{ flex: 1 }}>
                      <label className="rv-field-label" htmlFor="rv-exag">
                        Vertical Exaggeration Factor: {exaggeration.toFixed(1)}x (Visual Z-Scale Only)
                      </label>
                      <input
                        id="rv-exag"
                        type="range"
                        min="0.5"
                        max="8.0"
                        step="0.1"
                        value={exaggeration}
                        onChange={(e) => setExaggeration(Number(e.target.value))}
                      />
                    </div>
                    <div className="rv-field" style={{ minWidth: 140 }}>
                      <label className="rv-field-label" htmlFor="rv-zoom-slider">
                        Zoom: {zoom.toFixed(1)}x
                      </label>
                      <input
                        id="rv-zoom-slider"
                        type="range"
                        min="0.4"
                        max="4.0"
                        step="0.1"
                        value={zoom}
                        onChange={(e) => setZoom(Number(e.target.value))}
                      />
                    </div>
                  </div>

                  <ColorLegendBar
                    vMin={colorBounds.vMin}
                    vMax={colorBounds.vMax}
                    scaleMode={scaleMode}
                  />
                </section>

                <section className="rv-card">
                  <div className="rv-card-header">
                    <h3 className="rv-card-title">
                      <Layers size={16} className="rv-icon-cyan" /> 2D Relative-Depth Heatmap &amp; Masks
                    </h3>
                    <span className="rv-card-meta">Grid: {gw}×{gh} · Dashed = BBox</span>
                  </div>

                  <div className="rv-canvas-box">
                    <canvas ref={canvasHeatRef} className="rv-canvas-heatmap" />
                  </div>

                  <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: 10, fontSize: 11, color: 'var(--rv-text-secondary)', flexWrap: 'wrap', gap: 8 }}>
                    <span>Blue: Elevated / Protrusion · Cyan/Amber: Depressed Cavity · Green tint: Annular Reference · Red hatch: Invalid/Missing</span>
                    {scaleMode === 'snapshot' && (
                      <span style={{ color: '#fbbf24' }}>Warning: Local snapshot scaling is active (not cross-frame comparable).</span>
                    )}
                  </div>

                  <div className="rv-filter-bar" style={{ marginTop: 12, marginBottom: 0 }}>
                    <div className="rv-field" style={{ flex: 1 }}>
                      <label className="rv-field-label" htmlFor="rv-slice-row">
                        Horizontal Slice Row: {sliceRow} / {Math.max(0, gh - 1)}
                      </label>
                      <input
                        id="rv-slice-row"
                        type="range"
                        min={0}
                        max={Math.max(0, gh - 1)}
                        value={sliceRow}
                        onChange={(e) => setSliceRow(Number(e.target.value))}
                      />
                    </div>
                    <div className="rv-field" style={{ flex: 1 }}>
                      <label className="rv-field-label" htmlFor="rv-slice-col">
                        Vertical Slice Col: {sliceCol} / {Math.max(0, gw - 1)}
                      </label>
                      <input
                        id="rv-slice-col"
                        type="range"
                        min={0}
                        max={Math.max(0, gw - 1)}
                        value={sliceCol}
                        onChange={(e) => setSliceCol(Number(e.target.value))}
                      />
                    </div>
                  </div>
                </section>
              </div>

              <div className="rv-grid-2">
                <CrossSectionChart
                  points={profiles.horizontal}
                  title="Horizontal Cross-Section (Signed Relative Residual)"
                  axisLabel={`Row ${sliceRow} · gaps = invalid/missing`}
                  vMin={colorBounds.vMin}
                  vMax={colorBounds.vMax}
                />
                <CrossSectionChart
                  points={profiles.vertical}
                  title="Vertical Cross-Section (Signed Relative Residual)"
                  axisLabel={`Col ${sliceCol} · gaps = invalid/missing`}
                  vMin={colorBounds.vMin}
                  vMax={colorBounds.vMax}
                />
              </div>

              {snapshotStats && (
                <section className="rv-card">
                  <div className="rv-card-header">
                    <h3 className="rv-card-title">Snapshot Residual Distribution &amp; Mask Summary</h3>
                    <span className="rv-card-meta">
                      Valid cells: {snapshotStats.validCount} / {snapshotStats.totalCount}
                    </span>
                  </div>
                  <div className="rv-kv-list" style={{ gridTemplateColumns: 'repeat(auto-fit, minmax(175px, 1fr))' }}>
                    <div className="rv-kv-item">
                      <span className="rv-kv-label">Min / Max Residual</span>
                      <span className="rv-kv-value rv-mono">
                        {formatNullable(snapshotStats.min, 5)} / {formatNullable(snapshotStats.max, 5)}
                      </span>
                    </div>
                    <div className="rv-kv-item">
                      <span className="rv-kv-label">P5 / Median / P95</span>
                      <span className="rv-kv-value rv-mono">
                        {formatNullable(snapshotStats.p5, 4)} / {formatNullable(snapshotStats.median, 4)} / {formatNullable(snapshotStats.p95, 4)}
                      </span>
                    </div>
                    <div className="rv-kv-item">
                      <span className="rv-kv-label">Std Dev / Total Span</span>
                      <span className="rv-kv-value rv-mono">
                        {formatNullable(snapshotStats.std, 5)} / {formatNullable(snapshotStats.span, 5)}
                      </span>
                    </div>
                    <div className="rv-kv-item">
                      <span className="rv-kv-label">Cavity Core ({snapshotStats.cavityCount} cells)</span>
                      <span className="rv-kv-value rv-mono">
                        med {formatNullable(snapshotStats.cavityMedian, 5)}
                      </span>
                    </div>
                    <div className="rv-kv-item">
                      <span className="rv-kv-label">Reference Ring ({snapshotStats.refCount} cells)</span>
                      <span className="rv-kv-value rv-mono">
                        {snapshotStats.refCount > 0
                          ? `med ${formatNullable(snapshotStats.refMedian, 5)}`
                          : `none · surround med ${formatNullable(snapshotStats.nonCavityMedian, 5)}`}
                      </span>
                    </div>
                  </div>
                </section>
              )}

              <div className="rv-grid-2">
                <section className="rv-card">
                  <div className="rv-card-header">
                    <h3 className="rv-card-title"><ImageIcon size={16} className="rv-icon-cyan" /> Source ROI Crop &amp; Snapshot Metadata</h3>
                    <span className="rv-card-meta">Frame {snapshot.frame_idx} · {snapshot.timestamp_sec}s</span>
                  </div>
                  {snapshot.source_crop_data_url ? (
                    <img
                      src={snapshot.source_crop_data_url}
                      alt={`Track ${snapshot.track_id} ROI crop`}
                      className="rv-crop-thumb"
                    />
                  ) : (
                    <StatePanel variant="unavailable" title="Source crop not stored" message="Thumbnail disabled in job snapshot config." />
                  )}
                  <div className="rv-kv-list" style={{ marginTop: 12 }}>
                    <div className="rv-kv-item">
                      <span className="rv-kv-label">Model &amp; Resolution</span>
                      <span className="rv-kv-value">
                        {snapshot.model_metadata?.depth_model_id} ({formatNullable(snapshot.model_metadata?.depth_input_resolution, null, 'px')})
                      </span>
                    </div>
                    <div className="rv-kv-item">
                      <span className="rv-kv-label">Cavity Strategy</span>
                      <span className="rv-kv-value">{snapshot.model_metadata?.cavity_strategy || 'Unavailable'}</span>
                    </div>
                    <div className="rv-kv-item">
                      <span className="rv-kv-label">RANSAC Inlier Ratio</span>
                      <span className="rv-kv-value">{formatNullable(snapshot.plane_representation?.inlier_ratio, 3)}</span>
                    </div>
                    <div className="rv-kv-item">
                      <span className="rv-kv-label">Residual Polarity</span>
                      <span className="rv-kv-value" style={{ fontSize: 11 }}>{snapshot.residual_polarity}</span>
                    </div>
                  </div>
                </section>

                <section className="rv-card">
                  <div className="rv-card-header">
                    <h3 className="rv-card-title">Snapshot Measurement Quality</h3>
                  </div>
                  <QualityBreakdownCard assessment={snapshot.quality_assessment} />
                </section>
              </div>
            </>
          )}

          <ScientificDisclaimer text={snapshot?.scientific_disclaimer || manifest?.scientific_disclaimer} />
        </>
      )}
    </div>
  );
}
