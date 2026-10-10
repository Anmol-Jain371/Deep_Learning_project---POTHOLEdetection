import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  Target, Activity, Layers, Disc, Compass, Droplet, Sliders,
  ArrowRight, ShieldCheck, Cpu, GitBranch, Terminal, CheckCircle2,
  ChevronRight, Info, AlertTriangle
} from 'lucide-react';

// ======================================================================================
// 1. PIPELINE STAGE SPECIFICATIONS & DEFINITIONS
// ======================================================================================
const pipelineStages = [
  {
    step: '01',
    id: 'yolo',
    title: 'YOLOv8 Spatial Localization',
    summary: 'Deep convolutional detection backbone localized to road surface damage, outputting bounding box tensors in real time.',
    icon: Target,
    inputLabel: 'Raw Dashcam Frame (RGB)',
    processLabel: 'Darknet C2f Feature Pyramid',
    outputLabel: 'Bounding Box Tensors [x1, y1, x2, y2, conf]',
    specs: [
      { label: 'Architecture', val: 'YOLOv8s Feature Pyramid' },
      { label: 'Backbone Network', val: 'Darknet C2f Multi-Scale' },
      { label: 'Detection Heads', val: 'Anchor-Free P3, P4, P5' },
      { label: 'Output Tensor', val: '[x1, y1, x2, y2, conf]' }
    ],
    technicalNote: 'Processes video frames to detect asphalt cavities and surface fracture perimeters without requiring stereo camera hardware.'
  },
  {
    step: '02',
    id: 'bytetrack',
    title: 'ByteTrack Multi-Object Tracker',
    summary: 'Two-stage Kalman filter and Hungarian association engine maintaining persistent hazard identities across video frames.',
    icon: Activity,
    inputLabel: 'Frame Detections (High + Low Conf)',
    processLabel: 'Kalman Prediction + Hungarian Matching',
    outputLabel: 'Persistent Track IDs (#P-001, #P-002)',
    specs: [
      { label: 'Motion Model', val: 'Constant Velocity Kalman' },
      { label: 'Association Method', val: 'Two-Stage Hungarian IoU' },
      { label: 'State Vector', val: '[x, y, a, h, vx, vy]' },
      { label: 'Track Association', val: 'Zero Duplicate Recounts' }
    ],
    technicalNote: 'Assigns a stable municipal identifier to each defect across moving camera frames to prevent identical craters from being counted multiple times.'
  },
  {
    step: '03',
    id: 'midas',
    title: 'MiDaS 3D Monocular Depth',
    summary: 'EfficientNet inverse-depth encoder calculating continuous relative disparity maps from single camera viewpoints.',
    icon: Layers,
    inputLabel: 'Cropped Defect Patch',
    processLabel: 'MiDaS Small Depth Encoder',
    outputLabel: 'Relative Disparity Matrix D(x, y)',
    specs: [
      { label: 'Depth Backbone', val: 'Intel MiDaS Small' },
      { label: 'Execution Mode', val: 'Conditional Inference Gating' },
      { label: 'Disparity Format', val: 'Dense Inverse Depth D(x, y)' },
      { label: 'Interpolation', val: 'Bilinear Spatial Resampling' }
    ],
    technicalNote: 'Produces an uncalibrated relative inverse disparity gradient across the surface. Absolute physical measurement is resolved in Stage 05.'
  },
  {
    step: '04',
    id: 'donut',
    title: 'Annular Donut Surface Mask',
    summary: 'Concentric road-surface sampler isolating undamaged asphalt perimeter to establish an uncontaminated reference ground level.',
    icon: Disc,
    inputLabel: 'Detected Pothole Bounding Box',
    processLabel: 'M_donut = M_outer (25px) - M_inner',
    outputLabel: 'Median Road Baseline Reference',
    specs: [
      { label: 'Expansion Margin', val: '25px Perimeter Ring' },
      { label: 'Mask Formulation', val: 'M_donut = M_outer - M_inner' },
      { label: 'Reference Metric', val: 'Median Donut Disparity' },
      { label: 'Depression Metric', val: '15th Percentile Cavity Floor' }
    ],
    technicalNote: 'Subtracts the interior cavity from an expanded perimeter ring to measure the height difference between clean road and crater floor.'
  },
  {
    step: '05',
    id: 'pinhole',
    title: 'Pinhole Camera Calibration',
    summary: 'Optical focal ground-plane ray projection translating unitless relative disparity into absolute physical centimeters.',
    icon: Compass,
    inputLabel: 'Disparity Delta Δd + Optical Parameters',
    processLabel: 'Horizon Ray Projection Z_road = (H·fy)/dy',
    outputLabel: 'Estimated Depth in Centimeters (cm)',
    specs: [
      { label: 'Mounting Height (H)', val: '1.25 m Vehicle Standard' },
      { label: 'Vertical FOV', val: '60.0° Optical Apex' },
      { label: 'Ray Formulation', val: 'Z_road = (H_cam · fy) / dy' },
      { label: 'Scale Gain', val: 'ΔZ = Δd · (Z^1.35) · k_disp' }
    ],
    technicalNote: 'Uses known camera mounting height and ground-plane ray projection to calibrate relative disparity into estimated physical depth.'
  },
  {
    step: '06',
    id: 'aquaplane',
    title: 'Specular Aquaplane Detector',
    summary: 'HSV glare clustering and Laplacian texture variance testing to identify standing water and hydroplaning hazards.',
    icon: Droplet,
    inputLabel: 'Crater Pavement Patch (RGB)',
    processLabel: 'HSV Specular + Laplacian Variance',
    outputLabel: 'Aquaplane Risk Flag [WATER]',
    specs: [
      { label: 'HSV Glare Threshold', val: 'Value > 185 & Saturation < 45' },
      { label: 'Texture Variance', val: 'Laplacian σ² < 210 (Smooth)' },
      { label: 'Glare Threshold', val: '> 6.0% Surface Specularity' },
      { label: 'Municipal Status', val: 'Drain Clearing Work Order' }
    ],
    technicalNote: 'Combines color specularity with edge variance to distinguish glassy puddle water from rough dry asphalt surface.'
  },
  {
    step: '07',
    id: 'ema',
    title: 'Temporal EMA Depth Smoother',
    summary: 'Proximity-weighted Exponential Moving Average filter suppressing vehicle suspension pitch and camera vibration jitter.',
    icon: Sliders,
    inputLabel: 'Per-Frame Depth Estimates Z(t)',
    processLabel: 'Z_smooth = α·Z(t) + (1-α)·Z_prev',
    outputLabel: 'Locked Flicker-Free Depth (cm)',
    specs: [
      { label: 'Proximity Weight', val: 'α = 0.40 (Near) / 0.15 (Far)' },
      { label: 'Filter Mechanism', val: 'Proximity Exponential Moving Avg' },
      { label: 'Severity Scale', val: 'Mild (<4cm) / Mod (4-8) / Sev (≥8)' },
      { label: 'Temporal Window', val: 'Persistent Track Lifecycle' }
    ],
    technicalNote: 'Weights closer camera observations more heavily than distant ones to converge on a stable physical depth before classification.'
  }
];

// ======================================================================================
// 2. STAGE-SPECIFIC TECHNICAL ILLUSTRATIONS (Visual Engineering Diagrams)
// ======================================================================================

function Stage01Visual() {
  return (
    <div className="diagram-canvas-wrap">
      <div className="diagram-top-badge">
        <span className="diagram-tech-tag">STAGE 01 · SPATIAL CONVOLUTION & BOUNDING BOX EXTRACTION</span>
      </div>
      <svg viewBox="0 0 740 320" className="tech-diagram-svg" aria-label="YOLOv8 spatial localization diagram">
        {/* Background Grid */}
        <defs>
          <pattern id="archGrid" width="20" height="20" patternUnits="userSpaceOnUse">
            <path d="M 20 0 L 0 0 0 20" fill="none" stroke="rgba(0,0,0,0.04)" strokeWidth="1" />
          </pattern>
          <linearGradient id="roadGrad" x1="0%" y1="0%" x2="0%" y2="100%">
            <stop offset="0%" stopColor="#e2e8f0" />
            <stop offset="100%" stopColor="#cbd5e1" />
          </linearGradient>
          <linearGradient id="heatGrad" x1="0%" y1="0%" x2="100%" y2="0%">
            <stop offset="0%" stopColor="#f97316" stopOpacity="0.2" />
            <stop offset="100%" stopColor="#ea580c" stopOpacity="0.8" />
          </linearGradient>
        </defs>
        <rect width="740" height="320" fill="url(#archGrid)" />

        {/* 1. Input Road Frame */}
        <g transform="translate(30, 45)">
          <rect width="180" height="190" rx="8" fill="url(#roadGrad)" stroke="#94a3b8" strokeWidth="1.5" />
          {/* Road markings */}
          <path d="M 90 20 L 70 190" stroke="#ffffff" strokeWidth="2" strokeDasharray="8 6" opacity="0.8" />
          <path d="M 90 20 L 120 190" stroke="#ffffff" strokeWidth="2" strokeDasharray="8 6" opacity="0.8" />
          {/* Unprocessed pothole cavity */}
          <ellipse cx="98" cy="120" rx="28" ry="16" fill="#475569" opacity="0.7" />
          <ellipse cx="98" cy="120" rx="18" ry="9" fill="#1e293b" opacity="0.9" />
          {/* Label */}
          <rect x="10" y="10" width="105" height="22" rx="4" fill="#0f172a" />
          <text x="18" y="25" fill="#f8fafc" fontSize="11" fontFamily="'Space Mono', monospace" fontWeight="600">INPUT: 1080p RGB</text>
          <text x="90" y="210" textAnchor="middle" fill="#64748b" fontSize="12" fontWeight="600">Video Frame Tensor</text>
        </g>

        {/* Arrow 1 */}
        <path d="M 225 140 L 265 140" stroke="#e8651a" strokeWidth="2" strokeDasharray="4 4" />
        <polygon points="268,140 260,135 260,145" fill="#e8651a" />

        {/* 2. YOLOv8 C2f Backbone */}
        <g transform="translate(280, 40)">
          <rect width="180" height="200" rx="8" fill="#ffffff" stroke="#e2e8f0" strokeWidth="2" filter="drop-shadow(0 4px 12px rgba(0,0,0,0.04))" />
          <rect x="0" y="0" width="180" height="28" rx="8 8 0 0" fill="#f8fafc" stroke="#e2e8f0" strokeWidth="1" />
          <text x="90" y="19" textAnchor="middle" fill="#0f172a" fontSize="11" fontWeight="700" letterSpacing="0.05em">DARKNET C2f BACKBONE</text>

          {/* Feature pyramid blocks */}
          <rect x="25" y="45" width="130" height="26" rx="4" fill="#fff7ed" stroke="#fdba74" strokeWidth="1.5" />
          <text x="90" y="62" textAnchor="middle" fill="#c2410c" fontSize="11" fontWeight="600">P3 Layer (80×80)</text>

          <rect x="35" y="82" width="110" height="26" rx="4" fill="#ffedd5" stroke="#fb923c" strokeWidth="1.5" />
          <text x="90" y="99" textAnchor="middle" fill="#c2410c" fontSize="11" fontWeight="600">P4 Layer (40×40)</text>

          <rect x="45" y="119" width="90" height="26" rx="4" fill="#fed7aa" stroke="#f97316" strokeWidth="1.5" />
          <text x="90" y="136" textAnchor="middle" fill="#9a3412" fontSize="11" fontWeight="600">P5 Layer (20×20)</text>

          {/* Detection Head */}
          <rect x="15" y="156" width="150" height="30" rx="6" fill="#0f172a" />
          <text x="90" y="175" textAnchor="middle" fill="#ffffff" fontSize="11" fontWeight="600">Anchor-Free Head</text>
        </g>

        {/* Arrow 2 */}
        <path d="M 475 140 L 515 140" stroke="#e8651a" strokeWidth="2" strokeDasharray="4 4" />
        <polygon points="518,140 510,135 510,145" fill="#e8651a" />

        {/* 3. Output Detections */}
        <g transform="translate(530, 45)">
          <rect width="180" height="190" rx="8" fill="url(#roadGrad)" stroke="#94a3b8" strokeWidth="1.5" />
          {/* Road markings */}
          <path d="M 90 20 L 70 190" stroke="#ffffff" strokeWidth="2" strokeDasharray="8 6" opacity="0.8" />
          <path d="M 90 20 L 120 190" stroke="#ffffff" strokeWidth="2" strokeDasharray="8 6" opacity="0.8" />
          {/* Cavity */}
          <ellipse cx="98" cy="120" rx="28" ry="16" fill="#475569" opacity="0.7" />
          {/* Bounding Box Highlight */}
          <rect x="62" y="96" width="72" height="48" rx="2" fill="none" stroke="#22c55e" strokeWidth="2.5" strokeDasharray="none" />
          <rect x="62" y="78" width="78" height="18" fill="#22c55e" rx="2" />
          <text x="66" y="91" fill="#052e16" fontSize="10" fontFamily="'Space Mono', monospace" fontWeight="700">POTHOLE 0.88</text>
          <text x="90" y="210" textAnchor="middle" fill="#15803d" fontSize="12" fontWeight="700">Localized 2D Box</text>
        </g>

        {/* Footer equation */}
        <text x="370" y="295" textAnchor="middle" fill="#64748b" fontSize="12" fontFamily="'Space Mono', monospace">
          Loss Function: L_total = λ_box · L_CIOU + λ_cls · L_BCE + λ_dfl · L_DFL
        </text>
      </svg>
    </div>
  );
}

function Stage02Visual() {
  return (
    <div className="diagram-canvas-wrap">
      <div className="diagram-top-badge">
        <span className="diagram-tech-tag">STAGE 02 · KALMAN MOTION PREDICTION & TEMPORAL ASSOCIATION</span>
      </div>
      <svg viewBox="0 0 740 320" className="tech-diagram-svg" aria-label="ByteTrack multi-object tracker diagram">
        <defs>
          <pattern id="archGrid2" width="20" height="20" patternUnits="userSpaceOnUse">
            <path d="M 20 0 L 0 0 0 20" fill="none" stroke="rgba(0,0,0,0.04)" strokeWidth="1" />
          </pattern>
        </defs>
        <rect width="740" height="320" fill="url(#archGrid2)" />

        {/* Consecutive Frames Timeline */}
        {/* Frame t-1 */}
        <g transform="translate(40, 50)">
          <rect width="160" height="170" rx="8" fill="#f8fafc" stroke="#cbd5e1" strokeWidth="1.5" />
          <text x="80" y="24" textAnchor="middle" fill="#475569" fontSize="11" fontWeight="700">FRAME (t - 1)</text>
          <ellipse cx="80" cy="75" rx="24" ry="13" fill="#cbd5e1" />
          <rect x="52" y="58" width="56" height="34" rx="3" fill="none" stroke="#94a3b8" strokeWidth="2" strokeDasharray="4 2" />
          <rect x="52" y="44" width="48" height="14" fill="#94a3b8" rx="2" />
          <text x="56" y="54" fill="#ffffff" fontSize="9" fontFamily="'Space Mono', monospace" fontWeight="600">ID: ?</text>
        </g>

        {/* Association Arrow 1 */}
        <path d="M 215 135 L 265 135" stroke="#e8651a" strokeWidth="2" />
        <polygon points="268,135 260,130 260,140" fill="#e8651a" />

        {/* Frame t (Current with Kalman prediction) */}
        <g transform="translate(280, 50)">
          <rect width="170" height="170" rx="8" fill="#ffffff" stroke="#e8651a" strokeWidth="2" filter="drop-shadow(0 6px 16px rgba(232,101,26,0.08))" />
          <rect x="0" y="0" width="170" height="26" rx="8 8 0 0" fill="#fff7ed" />
          <text x="85" y="18" textAnchor="middle" fill="#c2410c" fontSize="11" fontWeight="700">FRAME (t) · CURRENT</text>

          {/* Previous position ghost */}
          <rect x="42" y="62" width="56" height="34" rx="3" fill="none" stroke="#cbd5e1" strokeWidth="1.5" strokeDasharray="3 3" />
          {/* Kalman Motion Vector */}
          <path d="M 70 79 L 95 105" stroke="#ea580c" strokeWidth="2.5" strokeDasharray="none" />
          <polygon points="98,108 90,103 94,97" fill="#ea580c" />
          {/* Current Track Box */}
          <ellipse cx="102" cy="115" rx="24" ry="13" fill="#fdba74" opacity="0.6" />
          <rect x="74" y="98" width="56" height="34" rx="3" fill="none" stroke="#e8651a" strokeWidth="2.5" />
          <rect x="74" y="84" width="58" height="14" fill="#e8651a" rx="2" />
          <text x="78" y="94" fill="#ffffff" fontSize="9" fontFamily="'Space Mono', monospace" fontWeight="700">#P-001</text>
        </g>

        {/* Association Arrow 2 */}
        <path d="M 465 135 L 515 135" stroke="#e8651a" strokeWidth="2" />
        <polygon points="518,135 510,130 510,140" fill="#e8651a" />

        {/* Frame t+1 (Persistent verified track) */}
        <g transform="translate(530, 50)">
          <rect width="160" height="170" rx="8" fill="#f8fafc" stroke="#cbd5e1" strokeWidth="1.5" />
          <text x="80" y="24" textAnchor="middle" fill="#475569" fontSize="11" fontWeight="700">FRAME (t + 1)</text>
          <ellipse cx="118" cy="135" rx="24" ry="13" fill="#bbf7d0" opacity="0.7" />
          <rect x="90" y="118" width="56" height="34" rx="3" fill="none" stroke="#16a34a" strokeWidth="2.5" />
          <rect x="90" y="104" width="58" height="14" fill="#16a34a" rx="2" />
          <text x="94" y="114" fill="#ffffff" fontSize="9" fontFamily="'Space Mono', monospace" fontWeight="700">#P-001</text>
        </g>

        {/* Footnote */}
        <text x="370" y="260" textAnchor="middle" fill="#0f172a" fontSize="12" fontWeight="700">
          Kalman State: x_k = F·x_{'{k-1}'} + w_k  |  Two-Stage Hungarian IoU Matching
        </text>
        <text x="370" y="285" textAnchor="middle" fill="#64748b" fontSize="11">
          Low-confidence detections matched in second pass to recover partially occluded defects.
        </text>
      </svg>
    </div>
  );
}

function Stage03Visual() {
  return (
    <div className="diagram-canvas-wrap">
      <div className="diagram-top-badge">
        <span className="diagram-tech-tag">STAGE 03 · MONOCULAR RELATIVE DISPARITY ENCODING</span>
      </div>
      <svg viewBox="0 0 740 320" className="tech-diagram-svg" aria-label="MiDaS 3D Monocular Depth diagram">
        <defs>
          <pattern id="archGrid3" width="20" height="20" patternUnits="userSpaceOnUse">
            <path d="M 20 0 L 0 0 0 20" fill="none" stroke="rgba(0,0,0,0.04)" strokeWidth="1" />
          </pattern>
          <linearGradient id="disparityMap" x1="0%" y1="100%" x2="0%" y2="0%">
            <stop offset="0%" stopColor="#f59e0b" />
            <stop offset="35%" stopColor="#8b5cf6" />
            <stop offset="70%" stopColor="#3b82f6" />
            <stop offset="100%" stopColor="#0f172a" />
          </linearGradient>
        </defs>
        <rect width="740" height="320" fill="url(#archGrid3)" />

        {/* Left: Input RGB Road Image */}
        <g transform="translate(60, 45)">
          <rect width="180" height="190" rx="8" fill="#e2e8f0" stroke="#94a3b8" strokeWidth="1.5" />
          <rect x="12" y="12" width="100" height="20" rx="4" fill="#0f172a" />
          <text x="20" y="26" fill="#ffffff" fontSize="10" fontFamily="'Space Mono', monospace" fontWeight="600">INPUT: RGB</text>
          <ellipse cx="90" cy="115" rx="36" ry="22" fill="#475569" />
          <text x="90" y="210" textAnchor="middle" fill="#475569" fontSize="12" fontWeight="600">Camera Color Space</text>
        </g>

        {/* Middle: MiDaS Deep CNN Encoder */}
        <g transform="translate(280, 50)">
          <rect width="170" height="180" rx="8" fill="#ffffff" stroke="#e2e8f0" strokeWidth="2" filter="drop-shadow(0 4px 12px rgba(0,0,0,0.04))" />
          <rect x="0" y="0" width="170" height="26" rx="8 8 0 0" fill="#f8fafc" />
          <text x="85" y="18" textAnchor="middle" fill="#0f172a" fontSize="11" fontWeight="700">INTEL MiDaS SMALL</text>

          <rect x="20" y="42" width="130" height="32" rx="4" fill="#ede9fe" stroke="#c4b5fd" strokeWidth="1" />
          <text x="85" y="62" textAnchor="middle" fill="#5b21b6" fontSize="11" fontWeight="600">EfficientNet Encoder</text>

          <path d="M 85 80 L 85 96" stroke="#8b5cf6" strokeWidth="2" />
          <polygon points="85,98 81,90 89,90" fill="#8b5cf6" />

          <rect x="20" y="104" width="130" height="32" rx="4" fill="#fae8ff" stroke="#f0abfc" strokeWidth="1" />
          <text x="85" y="124" textAnchor="middle" fill="#86198f" fontSize="11" fontWeight="600">Disparity Decoder</text>

          <text x="85" y="160" textAnchor="middle" fill="#64748b" fontSize="10" fontFamily="'Space Mono', monospace">Gating: Skip Clean Frames</text>
        </g>

        {/* Right: Output Relative Disparity Map */}
        <g transform="translate(490, 45)">
          <rect width="180" height="190" rx="8" fill="url(#disparityMap)" stroke="#6366f1" strokeWidth="2" />
          <rect x="12" y="12" width="140" height="20" rx="4" fill="rgba(15,23,42,0.85)" />
          <text x="20" y="26" fill="#f8fafc" fontSize="10" fontFamily="'Space Mono', monospace" fontWeight="600">DISPARITY D(x, y)</text>
          {/* Depth gradient legend */}
          <ellipse cx="90" cy="115" rx="36" ry="22" fill="#dc2626" opacity="0.8" />
          <text x="90" y="210" textAnchor="middle" fill="#4338ca" fontSize="12" fontWeight="700">Relative Disparity Map</text>
        </g>

        <text x="370" y="280" textAnchor="middle" fill="#0f172a" fontSize="12" fontWeight="600">
          *Important: Raw output represents unitless relative inverse disparity, not validated physical depth in centimeters.
        </text>
      </svg>
    </div>
  );
}

function Stage04Visual() {
  return (
    <div className="diagram-canvas-wrap">
      <div className="diagram-top-badge">
        <span className="diagram-tech-tag">STAGE 04 · CONCENTRIC ANNULAR DONUT ROAD-RIM SAMPLING</span>
      </div>
      <svg viewBox="0 0 740 320" className="tech-diagram-svg" aria-label="Annular Donut Surface Mask diagram">
        <defs>
          <pattern id="archGrid4" width="20" height="20" patternUnits="userSpaceOnUse">
            <path d="M 20 0 L 0 0 0 20" fill="none" stroke="rgba(0,0,0,0.04)" strokeWidth="1" />
          </pattern>
        </defs>
        <rect width="740" height="320" fill="url(#archGrid4)" />

        {/* Concentric Mask Diagram */}
        <g transform="translate(130, 40)">
          {/* Outer Ring M_outer (expanded 25px) */}
          <rect width="240" height="200" rx="8" fill="#f8fafc" stroke="#cbd5e1" strokeWidth="1.5" />
          <rect x="25" y="25" width="190" height="150" rx="8" fill="#ecfdf5" stroke="#10b981" strokeWidth="2.5" strokeDasharray="6 4" />

          {/* Inner Mask M_inner (crater cavity subtracted) */}
          <rect x="65" y="55" width="110" height="90" rx="6" fill="#fef2f2" stroke="#ef4444" strokeWidth="2.5" />

          {/* Labels on diagram */}
          <text x="120" y="44" textAnchor="middle" fill="#047857" fontSize="11" fontWeight="700">M_outer (Expanded 25px Boundary)</text>
          <text x="120" y="102" textAnchor="middle" fill="#b91c1c" fontSize="11" fontWeight="700">M_inner: Cavity</text>
          <text x="120" y="118" textAnchor="middle" fill="#b91c1c" fontSize="10">(Subtracted Region)</text>
          <text x="120" y="194" textAnchor="middle" fill="#047857" fontSize="11" fontWeight="600">Annular Donut Rim = Baseline</text>
        </g>

        {/* Mathematical formulation block */}
        <g transform="translate(420, 50)">
          <rect width="250" height="180" rx="8" fill="#ffffff" stroke="#e2e8f0" strokeWidth="1.5" filter="drop-shadow(0 4px 12px rgba(0,0,0,0.04))" />
          <rect x="0" y="0" width="250" height="28" rx="8 8 0 0" fill="#f8fafc" />
          <text x="125" y="19" textAnchor="middle" fill="#0f172a" fontSize="11" fontWeight="700">BASELINE CALCULATION</text>

          <g transform="translate(16, 44)">
            <text x="0" y="18" fill="#0f172a" fontSize="12" fontWeight="700">1. Road Rim Baseline:</text>
            <text x="12" y="36" fill="#047857" fontSize="11" fontFamily="'Space Mono', monospace">d_rim = Median(D[M_donut])</text>

            <text x="0" y="68" fill="#0f172a" fontSize="12" fontWeight="700">2. Crater Floor Estimator:</text>
            <text x="12" y="86" fill="#b91c1c" fontSize="11" fontFamily="'Space Mono', monospace">d_floor = Quantile(D[cavity], 0.15)</text>

            <text x="0" y="118" fill="#0f172a" fontSize="12" fontWeight="700">3. Relative Delta:</text>
            <text x="12" y="136" fill="#e8651a" fontSize="11" fontFamily="'Space Mono', monospace">Δd = |d_rim - d_floor|</text>
          </g>
        </g>

        <text x="370" y="275" textAnchor="middle" fill="#475569" fontSize="12">
          Annular sampling eliminates surrounding grass, curbs, and shadows by computing disparity strictly on adjacent pavement.
        </text>
      </svg>
    </div>
  );
}

function Stage05Visual() {
  return (
    <div className="diagram-canvas-wrap">
      <div className="diagram-top-badge">
        <span className="diagram-tech-tag">STAGE 05 · OPTICAL GROUND-PLANE RAY PROJECTION GEOMETRY</span>
      </div>
      <svg viewBox="0 0 740 320" className="tech-diagram-svg" aria-label="Pinhole camera calibration diagram">
        <defs>
          <pattern id="archGrid5" width="20" height="20" patternUnits="userSpaceOnUse">
            <path d="M 20 0 L 0 0 0 20" fill="none" stroke="rgba(0,0,0,0.04)" strokeWidth="1" />
          </pattern>
        </defs>
        <rect width="740" height="320" fill="url(#archGrid5)" />

        {/* Optical Ray Diagram Side View */}
        <g transform="translate(40, 30)">
          {/* Horizon Line */}
          <line x1="80" y1="40" x2="620" y2="40" stroke="#94a3b8" strokeWidth="1.5" strokeDasharray="6 4" />
          <text x="625" y="44" fill="#64748b" fontSize="10" fontFamily="'Space Mono', monospace">Vanishing Line (Horizon)</text>

          {/* Vehicle / Camera Origin */}
          <g transform="translate(80, 80)">
            {/* Camera Body */}
            <rect x="-15" y="-12" width="30" height="24" rx="4" fill="#0f172a" />
            <circle cx="15" cy="0" r="5" fill="#38bdf8" />
            <text x="0" y="-18" textAnchor="middle" fill="#0f172a" fontSize="11" fontWeight="700">Cam (H = 1.25m)</text>
            {/* Mounting pole to ground */}
            <line x1="0" y1="12" x2="0" y2="140" stroke="#475569" strokeWidth="2" />
            <circle cx="0" cy="140" r="4" fill="#475569" />
          </g>

          {/* Ground Plane */}
          <line x1="40" y1="220" x2="620" y2="220" stroke="#334155" strokeWidth="3" />
          <text x="625" y="224" fill="#334155" fontSize="11" fontWeight="700">Ground Plane (Z_road)</text>

          {/* Focal Projection Rays */}
          {/* Center Optical Axis */}
          <line x1="95" y1="80" x2="380" y2="220" stroke="#e8651a" strokeWidth="2.5" />
          {/* Depression Angle Ray */}
          <line x1="95" y1="80" x2="520" y2="220" stroke="#ea580c" strokeWidth="1.5" strokeDasharray="4 2" />

          {/* Pothole Cavity with Depth Delta ΔZ */}
          <g transform="translate(380, 220)">
            <path d="M 0 0 Q 30 35 60 0" fill="#fee2e2" stroke="#dc2626" strokeWidth="2.5" />
            <line x1="30" y1="0" x2="30" y2="25" stroke="#dc2626" strokeWidth="2" strokeDasharray="2 2" />
            <text x="30" y="44" textAnchor="middle" fill="#dc2626" fontSize="11" fontWeight="700" fontFamily="'Space Mono', monospace">ΔZ (cm)</text>
          </g>

          {/* Distance Annotation */}
          <path d="M 95 240 L 380 240" stroke="#64748b" strokeWidth="1" />
          <text x="237" y="255" textAnchor="middle" fill="#475569" fontSize="11" fontFamily="'Space Mono', monospace">Z_road = (H_cam · fy) / dy</text>
        </g>

        {/* Formula Footnote */}
        <text x="370" y="295" textAnchor="middle" fill="#0f172a" fontSize="12" fontWeight="700">
          Scale Conversion Equation:  ΔZ_est = Δd · (Z_road^1.35) · k_disp  [cm]
        </text>
      </svg>
    </div>
  );
}

function Stage06Visual() {
  return (
    <div className="diagram-canvas-wrap">
      <div className="diagram-top-badge">
        <span className="diagram-tech-tag">STAGE 06 · HSV SPECULAR GLARE & LAPLACIAN TEXTURE TEXTURING</span>
      </div>
      <svg viewBox="0 0 740 320" className="tech-diagram-svg" aria-label="Specular Aquaplane Detector diagram">
        <defs>
          <pattern id="archGrid6" width="20" height="20" patternUnits="userSpaceOnUse">
            <path d="M 20 0 L 0 0 0 20" fill="none" stroke="rgba(0,0,0,0.04)" strokeWidth="1" />
          </pattern>
        </defs>
        <rect width="740" height="320" fill="url(#archGrid6)" />

        {/* Input Patch */}
        <g transform="translate(40, 50)">
          <rect width="140" height="150" rx="8" fill="#334155" stroke="#64748b" strokeWidth="1.5" />
          {/* Water reflection highlight */}
          <ellipse cx="70" cy="80" rx="45" ry="25" fill="#38bdf8" opacity="0.6" />
          <ellipse cx="65" cy="75" rx="20" ry="8" fill="#ffffff" opacity="0.9" />
          <text x="70" y="25" textAnchor="middle" fill="#ffffff" fontSize="11" fontWeight="700">SURFACE PATCH</text>
          <text x="70" y="180" textAnchor="middle" fill="#64748b" fontSize="11" fontWeight="600">Puddle Candidate</text>
        </g>

        {/* Branch Arrow */}
        <path d="M 190 120 L 235 85" stroke="#0284c7" strokeWidth="2" />
        <path d="M 190 120 L 235 160" stroke="#0284c7" strokeWidth="2" />

        {/* Dual Test Branches */}
        {/* Branch A: HSV Specular Filter */}
        <g transform="translate(245, 40)">
          <rect width="210" height="75" rx="6" fill="#f0f9ff" stroke="#38bdf8" strokeWidth="1.5" />
          <text x="14" y="22" fill="#0369a1" fontSize="11" fontWeight="700">TEST 1: HSV Specular Filter</text>
          <text x="14" y="42" fill="#0c4a6e" fontSize="11" fontFamily="'Space Mono', monospace">Value &gt; 185 (High Luminance)</text>
          <text x="14" y="60" fill="#0c4a6e" fontSize="11" fontFamily="'Space Mono', monospace">Saturation &lt; 45 (White Glare)</text>
        </g>

        {/* Branch B: Laplacian Variance */}
        <g transform="translate(245, 130)">
          <rect width="210" height="75" rx="6" fill="#ecfeff" stroke="#06b6d4" strokeWidth="1.5" />
          <text x="14" y="22" fill="#0e7490" fontSize="11" fontWeight="700">TEST 2: Laplacian Texture Var</text>
          <text x="14" y="42" fill="#155e75" fontSize="11" fontFamily="'Space Mono', monospace">σ²_Lap &lt; 210.0 (Smooth Glass)</text>
          <text x="14" y="60" fill="#155e75" fontSize="11" fontFamily="'Space Mono', monospace">Dry Asphalt: σ² &gt; 450 (Rough)</text>
        </g>

        {/* Merge Arrows */}
        <path d="M 465 77 L 515 110" stroke="#0284c7" strokeWidth="2" />
        <path d="M 465 167 L 515 130" stroke="#0284c7" strokeWidth="2" />

        {/* Decision Badge */}
        <g transform="translate(525, 60)">
          <rect width="170" height="130" rx="8" fill="#ffffff" stroke="#0284c7" strokeWidth="2" filter="drop-shadow(0 4px 12px rgba(2,132,199,0.15))" />
          <rect x="0" y="0" width="170" height="26" rx="8 8 0 0" fill="#e0f2fe" />
          <text x="85" y="18" textAnchor="middle" fill="#0369a1" fontSize="11" fontWeight="700">CLASSIFICATION</text>
          <circle cx="85" cy="55" r="16" fill="#0284c7" />
          <path d="M 85 45 C 80 52 75 58 75 62 A 10 10 0 0 0 95 62 C 95 58 90 52 85 45 Z" fill="#ffffff" />
          <text x="85" y="90" textAnchor="middle" fill="#0c4a6e" fontSize="12" fontWeight="800">SEVERE [WATER]</text>
          <text x="85" y="108" textAnchor="middle" fill="#0284c7" fontSize="10" fontWeight="600">Aquaplaning Risk</text>
        </g>

        <text x="370" y="275" textAnchor="middle" fill="#475569" fontSize="12">
          High reflectance coupled with low surface roughness confirms water presence, triggering emergency drain-clearing protocols.
        </text>
      </svg>
    </div>
  );
}

function Stage07Visual() {
  return (
    <div className="diagram-canvas-wrap">
      <div className="diagram-top-badge">
        <span className="diagram-tech-tag">STAGE 07 · PROXIMITY-WEIGHTED EXPONENTIAL MOVING AVERAGE</span>
      </div>
      <svg viewBox="0 0 740 320" className="tech-diagram-svg" aria-label="Temporal EMA Depth Smoother diagram">
        <defs>
          <pattern id="archGrid7" width="20" height="20" patternUnits="userSpaceOnUse">
            <path d="M 20 0 L 0 0 0 20" fill="none" stroke="rgba(0,0,0,0.04)" strokeWidth="1" />
          </pattern>
        </defs>
        <rect width="740" height="320" fill="url(#archGrid7)" />

        {/* Waveform Comparison Graph */}
        <g transform="translate(60, 40)">
          {/* Axes */}
          <line x1="40" y1="20" x2="40" y2="180" stroke="#94a3b8" strokeWidth="1.5" />
          <line x1="40" y1="180" x2="600" y2="180" stroke="#94a3b8" strokeWidth="1.5" />
          <text x="25" y="100" textAnchor="middle" transform="rotate(-90 25 100)" fill="#64748b" fontSize="11" fontWeight="600">Depth (cm)</text>
          <text x="320" y="200" textAnchor="middle" fill="#64748b" fontSize="11" fontWeight="600">Observation Frame Sequence [t1 → tN]</text>

          {/* Grid lines */}
          <line x1="40" y1="60" x2="600" y2="60" stroke="#e2e8f0" strokeDasharray="3 3" />
          <text x="32" y="64" textAnchor="end" fill="#94a3b8" fontSize="10">10cm</text>
          <line x1="40" y1="120" x2="600" y2="120" stroke="#e2e8f0" strokeDasharray="3 3" />
          <text x="32" y="124" textAnchor="end" fill="#94a3b8" fontSize="10">5cm</text>

          {/* Raw Noisy Signal (Red jagged) */}
          <path
            d="M 40 140 L 80 85 L 120 160 L 160 70 L 200 135 L 240 60 L 280 120 L 320 75 L 360 115 L 400 80 L 440 110 L 480 88 L 520 102 L 560 92 L 600 95"
            fill="none"
            stroke="#f87171"
            strokeWidth="2"
            strokeDasharray="4 2"
          />

          {/* Smoothed EMA Signal (Solid Amber) */}
          <path
            d="M 40 140 Q 120 115 200 105 T 360 98 T 480 95 T 600 94"
            fill="none"
            stroke="#e8651a"
            strokeWidth="3.5"
          />

          {/* Legend */}
          <g transform="translate(380, 25)">
            <rect width="210" height="50" rx="6" fill="#ffffff" stroke="#e2e8f0" strokeWidth="1" />
            <line x1="15" y1="18" x2="45" y2="18" stroke="#f87171" strokeWidth="2" strokeDasharray="4 2" />
            <text x="55" y="22" fill="#ef4444" fontSize="11" fontWeight="600">Raw Frame Jitter (±3.5cm)</text>

            <line x1="15" y1="36" x2="45" y2="36" stroke="#e8651a" strokeWidth="3" />
            <text x="55" y="40" fill="#e8651a" fontSize="11" fontWeight="700">Smoothed EMA (Stable 6.8cm)</text>
          </g>
        </g>

        {/* Mathematical formulation block */}
        <text x="370" y="275" textAnchor="middle" fill="#0f172a" fontSize="12" fontWeight="700">
          Proximity Formulation:  Z_smooth = α·Z_cur + (1 - α)·Z_prev  |  α_near = 0.40, α_far = 0.15
        </text>
      </svg>
    </div>
  );
}

// Stage map registry
const stageVisuals = {
  yolo: Stage01Visual,
  bytetrack: Stage02Visual,
  midas: Stage03Visual,
  donut: Stage04Visual,
  pinhole: Stage05Visual,
  aquaplane: Stage06Visual,
  ema: Stage07Visual
};

// ======================================================================================
// 3. MAIN PIPELINE SECTION COMPONENT (EDITORIAL ARCHITECTURE CATALOG)
// ======================================================================================
export default function PipelineSection({ expanded = false }) {
  const [selectedStage, setSelectedStage] = useState(pipelineStages[0]);
  const ActiveVisual = stageVisuals[selectedStage.id] || Stage01Visual;
  const ActiveIcon = selectedStage.icon;

  return (
    <section className="architecture-editorial-page" aria-label="RoadVision Pipeline Architecture">
      {/* ── TOP SECTION HEADER ── */}
      <div className="arch-editorial-header">
        <span className="arch-eyebrow">SYSTEM ARCHITECTURE</span>
        <h1 className="arch-main-title">Inside the RoadVision Pipeline</h1>
        <p className="arch-main-lead">
          The autonomous perception engine processes road video footage through seven sequentially coupled engineering stages.
        </p>
      </div>

      {/* ── TWO-COLUMN WORKSPACE: LEFT NAVIGATOR + RIGHT ILLUSTRATED VISUALIZATION ── */}
      <div className="arch-workspace-grid">

        {/* LEFT COLUMN: 7-STAGE VERTICAL SELECTOR */}
        <aside className="arch-left-navigator" aria-label="Pipeline Stages List">
          <div className="nav-col-header">
            <span className="nav-col-title">PIPELINE STAGES</span>
            <span className="nav-col-count">7 SEQUENTIAL STAGES</span>
          </div>

          <div className="nav-stages-list" role="tablist">
            {pipelineStages.map((stage) => {
              const Icon = stage.icon;
              const isActive = selectedStage.id === stage.id;

              return (
                <button
                  key={stage.id}
                  role="tab"
                  aria-selected={isActive}
                  aria-controls={`stage-panel-${stage.id}`}
                  className={`arch-stage-item ${isActive ? 'is-active' : ''}`}
                  onClick={() => setSelectedStage(stage)}
                >
                  <span className="stage-item-num">{stage.step}</span>
                  <div className="stage-item-icon">
                    <Icon size={16} />
                  </div>
                  <div className="stage-item-details">
                    <span className="stage-item-title">{stage.title}</span>
                  </div>
                  <ChevronRight size={14} className="stage-item-arrow" />
                </button>
              );
            })}
          </div>
        </aside>

        {/* RIGHT COLUMN: ILLUSTRATED TECHNICAL VISUALIZATION AREA */}
        <main
          id={`stage-panel-${selectedStage.id}`}
          className="arch-right-stage"
          role="tabpanel"
          aria-label={selectedStage.title}
        >
          {/* STAGE HEADER BAR */}
          <div className="stage-banner-bar">
            <div className="banner-left">
              <span className="banner-step-pill">STAGE {selectedStage.step}</span>
              <h2 className="banner-title">{selectedStage.title}</h2>
            </div>
            <p className="banner-summary">{selectedStage.summary}</p>
          </div>

          {/* 3-STEP DIRECTIONAL DATA FLOW RIBBON */}
          <div className="directional-flow-ribbon" aria-label="Data flow ribbon">
            <div className="flow-step-pill input">
              <span className="flow-tag">INPUT</span>
              <span className="flow-text">{selectedStage.inputLabel}</span>
            </div>

            <ArrowRight size={16} className="flow-arrow-icon" />

            <div className="flow-step-pill process">
              <span className="flow-tag">PROCESSING</span>
              <span className="flow-text">{selectedStage.processLabel}</span>
            </div>

            <ArrowRight size={16} className="flow-arrow-icon" />

            <div className="flow-step-pill output">
              <span className="flow-tag">OUTPUT</span>
              <span className="flow-text">{selectedStage.outputLabel}</span>
            </div>
          </div>

          {/* TECHNICAL VISUAL ENGINEERING ILLUSTRATION */}
          <div className="stage-primary-visual-box">
            <ActiveVisual />
          </div>

          {/* TECHNICAL PARAMETERS SPECIFICATIONS BAR */}
          <div className="stage-specs-strip">
            {selectedStage.specs.map((item, idx) => (
              <div key={idx} className="spec-tile">
                <span className="tile-label">{item.label}</span>
                <span className="tile-val">{item.val}</span>
              </div>
            ))}
          </div>

          {/* TECHNICAL SUMMARY NOTE */}
          <div className="stage-grounded-note">
            <Info size={15} className="note-info-icon" />
            <p>{selectedStage.technicalNote}</p>
          </div>
        </main>

      </div>
    </section>
  );
}
