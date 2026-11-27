import React, { useState, useEffect } from 'react';
import { Scale, AlertTriangle, Info, ShieldAlert, Cpu } from 'lucide-react';

const API_BASE = import.meta.env.VITE_API_URL || '';

export default function BenchmarkComparison() {
  const [archiveMeta, setArchiveMeta] = useState(null);

  useEffect(() => {
    let isMounted = true;
    async function fetchArchiveMeta() {
      try {
        const res = await fetch(`${API_BASE}/api/benchmark`);
        if (!res.ok) return;
        const data = await res.json();
        if (isMounted) {
          setArchiveMeta(data);
        }
      } catch {
        // Ignore connectivity errors for optional legacy archive metadata
      }
    }
    fetchArchiveMeta();
    return () => { isMounted = false; };
  }, []);

  const rows = [
    {
      metric: 'Legacy Model Architecture (pothole_rdd1666_50e)',
      value: 'YOLOv8n-scale (3,011,043 params, 5.96 MiB)',
      fasterRcnn: 'Faster R-CNN ResNet-50-FPN-v2 (unverified historical reference)',
      note: 'Verified directly from checkpoint parameter count (historical JSON mislabeled it as YOLOv8s 11.13M).',
    },
    {
      metric: 'Legacy Validation Claims (84.7% mAP50 / 92.4% Prec)',
      value: 'Historical benchmark — verification unavailable.',
      fasterRcnn: 'Historical benchmark — verification unavailable.',
      note: 'Removed from active outputs: archived training results.csv recorded 0.45765 mAP50 at epoch 50 rather than 84.7%.',
    },
    {
      metric: 'Legacy Clip Throughput / Latency (37.8 ms / 26.5 vs 21.9 FPS)',
      value: 'Historical benchmark — verification unavailable.',
      fasterRcnn: 'Historical benchmark — verification unavailable.',
      note: 'Legacy timing figures contradicted earlier prose summaries and are excluded from active evaluation claims.',
    },
    {
      metric: 'Reconciled Exploratory Evaluation (Phase 13 / 13B)',
      value: 'See RoadVision AI 2.0 Performance & Evaluation Artifacts',
      fasterRcnn: 'Not part of RoadVision AI 2.0 runtime',
      note: 'Use the V2 Inspection Workspace & Phase 13B reproducible evaluation runs instead of V1 archive figures.',
    },
  ];

  return (
    <section className="benchmark-section" id="benchmark-comparison">
      <div className="benchmark-container">
        <div className="benchmark-header">
          <div className="benchmark-eyebrow">
            <Scale size={14} className="eyebrow-icon" />
            <span>LEGACY V1 RESEARCH ARCHIVE</span>
          </div>
          <h2 className="benchmark-title">Legacy V1 Model Comparison Archive</h2>
          <p className="benchmark-subtitle">
            Historical V1 exploratory artifacts retained only for engineering traceability.
          </p>
        </div>

        <div
          className="rv-state-panel rv-state-api-disabled"
          role="alert"
          style={{ marginBottom: 20, border: '1px solid rgba(245, 158, 11, 0.45)', padding: '14px 16px' }}
        >
          <ShieldAlert size={18} style={{ color: '#f59e0b', flexShrink: 0 }} />
          <div className="rv-state-body">
            <h4 className="rv-state-title" style={{ marginBottom: 4 }}>
              Legacy V1 Research Archive Warning — Historical benchmark — verification unavailable.
            </h4>
            <p className="rv-state-message" style={{ margin: 0 }}>
              Previously displayed V1 figures (<code>84.7% mAP50</code>, <code>92.4% precision</code>, and <code>YOLOv8s 11.13M</code> misidentification) did not reconcile with saved training logs or PyTorch checkpoint metadata and have been removed from active application outputs.
              {archiveMeta?.model_metadata_note ? ` ${archiveMeta.model_metadata_note}` : ''}
            </p>
          </div>
        </div>

        <div className="benchmark-table-wrapper">
          <div className="table-header-strip">
            <h3 className="strip-title">
              <Cpu size={16} /> Reconciled Checkpoint &amp; Archive Status
            </h3>
            <span className="dataset-tag">Status: Historical Archive Only</span>
          </div>

          <div className="table-responsive">
            <table className="benchmark-table">
              <thead>
                <tr>
                  <th>Benchmark Dimension</th>
                  <th>V1 Checkpoint Record</th>
                  <th>Two-Stage Reference Record</th>
                  <th>Audit Reconciliation</th>
                </tr>
              </thead>
              <tbody>
                {rows.map((r) => (
                  <tr key={r.metric}>
                    <td className="metric-name">{r.metric}</td>
                    <td className="metric-val font-mono">{r.value}</td>
                    <td className="metric-val font-mono">{r.fasterRcnn}</td>
                    <td className="metric-desc">{r.note}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

        <div className="benchmark-viva-box">
          <div className="viva-header">
            <Info size={18} className="text-amber" />
            <h4>Traceability Notice</h4>
          </div>
          <p className="viva-text">
            Original historical JSON/CSV files remain preserved in the repository archive for audit provenance.
            Active comparative evaluation in RoadVision AI 2.0 is conducted through the frozen Phase 2 YOLOv8n/YOLOv8s checkpoints and the Phase 13B reproducible evaluation suite.
          </p>
        </div>
      </div>
    </section>
  );
}
