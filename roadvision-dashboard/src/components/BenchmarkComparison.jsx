import React, { useState } from 'react';
import { motion } from 'framer-motion';
import {
  Scale, Zap, HardDrive, Cpu, ShieldAlert, CheckCircle2,
  AlertTriangle, ArrowRight, Gauge, Layers, Info
} from 'lucide-react';

const benchmarkData = {
  yolo: {
    name: 'YOLOv8s',
    paradigm: 'One-Stage (Single-Shot)',
    backbone: 'Modified CSPDarknet (Anchor-Free)',
    latency: '37.8 ms',
    fps: '26.5 FPS',
    fpsVal: 26.45,
    params: '11.13 M',
    size: '5.96 MB',
    coverage: '100.0%',
    status: 'PRODUCTION (84.7% mAP50)',
    statusType: 'success',
    color: '#10b981',
    pros: [
      'Real-time automotive throughput (26.5 FPS, 37.8 ms latency)',
      'High empirical accuracy (84.7% mAP@50 with TTA, 92.4% precision)',
      'Sub-40ms latency leaves compute headroom for 3D depth computation',
      'Ultra-compact footprint (5.96 MB) easily deployable on NVIDIA Jetson / Edge devices',
      'Anchor-free regression prevents redundant overlapping proposals'
    ],
    cons: [
      'Single-stage trade-off requires multi-scale training for fine cracks'
    ]
  },
  rcnn: {
    name: 'Faster R-CNN',
    paradigm: 'Two-Stage (RPN + RoIAlign)',
    backbone: 'ResNet-50-FPN-v2 (Feature Pyramid)',
    latency: '256.5 ms',
    fps: '3.90 FPS',
    fpsVal: 3.90,
    params: '43.26 M',
    size: '329.58 MB',
    coverage: '100.0%',
    status: 'BASELINE REFERENCE',
    statusType: 'warning',
    color: '#f59e0b',
    pros: [
      'Dense region proposals on subtle surface irregularities',
      'Standard academic reference architecture for two-stage detection',
      'High theoretical spatial recall via RoIAlign feature extraction'
    ],
    cons: [
      'Prohibitive 256.5ms latency creates ~2.9m vehicle blind-spots at 40 km/h',
      'Heavy 330 MB checkpoint strains edge memory and memory bandwidth',
      'Dense overlapping proposal duplicates (235 candidates at 0.39 confidence)',
      'Incapable of running real-time 3D depth pipeline concurrently'
    ]
  }
};

export default function BenchmarkComparison() {
  const [activeTab, setActiveTab] = useState('overview');

  return (
    <section className="benchmark-section" id="benchmark-comparison">
      <div className="benchmark-container">
        {/* Section Header */}
        <div className="benchmark-header">
          <div className="benchmark-eyebrow">
            <Scale size={14} className="eyebrow-icon" />
            <span>EMPIRICAL ARCHITECTURAL ABLATION STUDY</span>
          </div>
          <h2 className="benchmark-title">One-Stage vs. Two-Stage Detection Benchmark</h2>
          <p className="benchmark-subtitle">
            Controlled evaluation of <strong>YOLOv8s</strong> against <strong>Faster R-CNN (ResNet-50-FPN-v2)</strong> on
            held-out road damage video streams (<code className="code-pill">Rangoon Road</code> monsoon sequence)
            evaluated on hardware acceleration (NVIDIA RTX 2050 CUDA).
          </p>
        </div>

        {/* Top KPI Comparison Grid */}
        <div className="benchmark-card-grid">
          {/* Card 1: Throughput */}
          <motion.div
            className="benchmark-kpi-card"
            initial={{ opacity: 0, y: 15 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.3 }}
          >
            <div className="card-top">
              <span className="card-label">INFERENCE THROUGHPUT</span>
              <Zap size={18} className="card-icon text-amber" />
            </div>
            <div className="card-comparison">
              <div className="comp-item highlight">
                <span className="comp-model">YOLOv8s</span>
                <span className="comp-val text-emerald">{benchmarkData.yolo.fps}</span>
                <span className="comp-badge success">Real-Time</span>
              </div>
              <div className="comp-divider">vs</div>
              <div className="comp-item">
                <span className="comp-model">Faster R-CNN</span>
                <span className="comp-val text-rose">{benchmarkData.rcnn.fps}</span>
                <span className="comp-badge warning">5.7× Slower</span>
              </div>
            </div>
            <div className="card-footer-note">
              YOLOv8s delivers <strong>572% higher frame rates</strong>, ensuring seamless integration with downstream MiDaS depth.
            </div>
          </motion.div>

          {/* Card 2: Latency */}
          <motion.div
            className="benchmark-kpi-card"
            initial={{ opacity: 0, y: 15 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.3, delay: 0.1 }}
          >
            <div className="card-top">
              <span className="card-label">PER-FRAME LATENCY</span>
              <Gauge size={18} className="card-icon text-blue" />
            </div>
            <div className="card-comparison">
              <div className="comp-item highlight">
                <span className="comp-model">YOLOv8s</span>
                <span className="comp-val text-emerald">{benchmarkData.yolo.latency}</span>
                <span className="comp-badge success">Edge-Viable</span>
              </div>
              <div className="comp-divider">vs</div>
              <div className="comp-item">
                <span className="comp-model">Faster R-CNN</span>
                <span className="comp-val text-rose">{benchmarkData.rcnn.latency}</span>
                <span className="comp-badge warning">+215.7 ms</span>
              </div>
            </div>
            <div className="card-footer-note">
              Faster R-CNN incurs over <strong>1/4 second per frame</strong>, producing spatial lag in high-speed vehicular perception.
            </div>
          </motion.div>

          {/* Card 3: Model Footprint */}
          <motion.div
            className="benchmark-kpi-card"
            initial={{ opacity: 0, y: 15 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.3, delay: 0.2 }}
          >
            <div className="card-top">
              <span className="card-label">MODEL WEIGHT FOOTPRINT</span>
              <HardDrive size={18} className="card-icon text-purple" />
            </div>
            <div className="card-comparison">
              <div className="comp-item highlight">
                <span className="comp-model">YOLOv8s</span>
                <span className="comp-val text-emerald">{benchmarkData.yolo.size}</span>
                <span className="comp-badge success">11.13M params</span>
              </div>
              <div className="comp-divider">vs</div>
              <div className="comp-item">
                <span className="comp-model">Faster R-CNN</span>
                <span className="comp-val text-rose">{benchmarkData.rcnn.size}</span>
                <span className="comp-badge warning">54× Larger</span>
              </div>
            </div>
            <div className="card-footer-note">
              YOLOv8s is <strong>54× more compact</strong>, making over-the-air municipal vehicle updates friction-free.
            </div>
          </motion.div>
        </div>

        {/* Detailed Side-by-Side Comparison Table */}
        <div className="benchmark-table-wrapper">
          <div className="table-header-strip">
            <h3 className="strip-title">
              <Layers size={16} /> Architectural Specifications & Telemetry Comparison
            </h3>
            <span className="dataset-tag">Dataset: RDD2022 India D40 Pothole</span>
          </div>

          <div className="table-responsive">
            <table className="benchmark-table">
              <thead>
                <tr>
                  <th>Evaluation Metric</th>
                  <th className="th-highlight">
                    YOLOv8 Small <span className="th-badge-active">Selected Production</span>
                  </th>
                  <th>Faster R-CNN (ResNet-50-FPN) <span className="th-badge-baseline">Comparative Baseline</span></th>
                  <th>Engineering Significance</th>
                </tr>
              </thead>
              <tbody>
                <tr>
                  <td className="metric-name">Paradigm</td>
                  <td className="metric-val highlight">One-Stage (Anchor-free grid regression)</td>
                  <td className="metric-val">Two-Stage (RPN + RoIAlign head)</td>
                  <td className="metric-desc">Single-pass prediction avoids candidate proposal overhead</td>
                </tr>
                <tr>
                  <td className="metric-name">Validation Accuracy (mAP@50)</td>
                  <td className="metric-val highlight text-emerald font-mono">84.7% (with TTA) / 82.1%</td>
                  <td className="metric-val text-amber font-mono">71.0%</td>
                  <td className="metric-desc">High precision (92.4%) eliminates false alarm fatigue</td>
                </tr>
                <tr>
                  <td className="metric-name">Inference Latency</td>
                  <td className="metric-val highlight text-emerald font-mono">37.80 ms</td>
                  <td className="metric-val text-rose font-mono">256.55 ms</td>
                  <td className="metric-desc">YOLOv8s is <strong>6.8× faster</strong> per frame</td>
                </tr>
                <tr>
                  <td className="metric-name">Inference Speed (FPS)</td>
                  <td className="metric-val highlight text-emerald font-mono">26.45 FPS</td>
                  <td className="metric-val text-rose font-mono">3.90 FPS</td>
                  <td className="metric-desc">Critical threshold for 25–30 FPS automotive video feeds</td>
                </tr>
                <tr>
                  <td className="metric-name">Parameter Count</td>
                  <td className="metric-val highlight font-mono">11.13 Million</td>
                  <td className="metric-val font-mono">43.26 Million</td>
                  <td className="metric-desc">3.9× parameter reduction; lower thermal & power draw</td>
                </tr>
                <tr>
                  <td className="metric-name">Checkpoint File Size</td>
                  <td className="metric-val highlight font-mono">5.96 MB</td>
                  <td className="metric-val font-mono">329.58 MB</td>
                  <td className="metric-desc">55× memory footprint savings on onboard storage</td>
                </tr>
                <tr>
                  <td className="metric-name">Detection Coverage %</td>
                  <td className="metric-val highlight font-mono">100.0% (50/50 frames)</td>
                  <td className="metric-val font-mono">100.0% (50/50 frames)</td>
                  <td className="metric-desc">Both architectures successfully localized hazard zones</td>
                </tr>
                <tr>
                  <td className="metric-name">Raw Defect Proposals</td>
                  <td className="metric-val highlight font-mono">158 confident detections</td>
                  <td className="metric-val font-mono">235 dense proposals</td>
                  <td className="metric-desc">Higher defect sensitivity with clean spatial non-redundancy</td>
                </tr>
                <tr>
                  <td className="metric-name">Downstream Compatibility</td>
                  <td className="metric-val highlight">
                    <span className="badge-inline-success">✓ 3D Depth + Tracking Ready</span>
                  </td>
                  <td className="metric-val">
                    <span className="badge-inline-danger">✕ Pipeline Choke Point</span>
                  </td>
                  <td className="metric-desc">MiDaS depth & ByteTrack require sub-50ms detection inputs</td>
                </tr>
              </tbody>
            </table>
          </div>
        </div>

        {/* Architectural Defense Box */}
        <div className="benchmark-viva-box">
          <div className="viva-header">
            <CheckCircle2 size={18} className="text-emerald" />
            <h4>Academic Defense & Production Rationale</h4>
          </div>
          <p className="viva-text">
            While <strong>Faster R-CNN</strong> serves as a venerable baseline in academic computer vision literature,
            its two-stage Region Proposal Network (RPN) architecture introduces a severe latency penalty (<strong>261.5 ms</strong> on GPU).
            In a practical municipal survey vehicle traveling at 40 km/h, this latency translates to nearly <strong>3 meters of unobserved roadway</strong> between frames.
            <strong> YOLOv8s</strong> operates along the optimal Pareto frontier: matching detection coverage at <strong>21.9 FPS</strong>,
            providing the requisite compute budget to execute 3D monocular depth estimation, annular road-plane detrending, and Kalman-filtered tracking in real time.
          </p>
        </div>
      </div>
    </section>
  );
}
