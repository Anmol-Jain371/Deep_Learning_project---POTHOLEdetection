import React, { useEffect, useState, useCallback } from 'react';
import { getPerformance } from '../api/client';
import { Activity, Cpu, Clock, Gauge, ChevronLeft, ChevronRight } from 'lucide-react';
import InspectionSelector from '../components/InspectionSelector';
import { StatePanel } from '../components/StateBanner';
import { isValidUuid, formatNullable, formatSeconds } from '../utils/formatters';

const STAGE_KEYS = [
  { key: 'decode_ms', label: 'Decode' },
  { key: 'detect_track_ms', label: 'Detect & Track' },
  { key: 'scheduler_ms', label: 'Scheduler' },
  { key: 'depth_pre_ms', label: 'Depth Preprocess' },
  { key: 'depth_inf_ms', label: 'Depth Inference' },
  { key: 'depth_resize_ms', label: 'Depth Resize' },
  { key: 'extract_ms', label: 'Cavity Extraction' },
  { key: 'horizon_diag_ms', label: 'Horizon Diagnostics' },
  { key: 'visualize_ms', label: 'Annotation' },
];

export default function Performance({ selectedJobId, onSelectJob, onNavigate }) {
  const [perf, setPerf] = useState(null);
  const [page, setPage] = useState(1);
  const [pageSize] = useState(25);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const loadPerf = useCallback(async () => {
    if (!selectedJobId || !isValidUuid(selectedJobId)) return;
    setLoading(true);
    setError(null);
    try {
      const resp = await getPerformance(selectedJobId, { page, page_size: pageSize });
      setPerf(resp);
    } catch (err) {
      setPerf(null);
      setError(err);
    } finally {
      setLoading(false);
    }
  }, [selectedJobId, page, pageSize]);

  useEffect(() => {
    setPage(1);
  }, [selectedJobId]);

  useEffect(() => {
    loadPerf();
  }, [loadPerf]);

  const telemetry = perf?.telemetry || {};
  const frames = perf?.frames || [];
  const totalRows = perf?.total ?? 0;
  const totalPages = Math.max(1, Math.ceil(totalRows / pageSize));

  const stageAverages = STAGE_KEYS.map(({ key, label }) => {
    const vals = frames.map((f) => f[key]).filter((v) => typeof v === 'number' && !Number.isNaN(v));
    const avg = vals.length > 0 ? vals.reduce((a, b) => a + b, 0) / vals.length : null;
    return { key, label, avg };
  });

  const maxStageMs = stageAverages.reduce((mx, s) => Math.max(mx, s.avg || 0), 0.1);

  return (
    <div className="rv-page">
      <div className="rv-page-header">
        <div>
          <h2 className="rv-page-title">Pipeline Performance</h2>
          <p className="rv-page-subtitle">
            Stage-by-stage frame latency breakdown, GPU memory telemetry, and scheduler invocation counts.
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
        <StatePanel variant="empty" title="Select an inspection" message="Choose a completed inspection above to view telemetry." />
      ) : loading ? (
        <StatePanel variant="loading" title="Loading performance telemetry" message="Fetching worker and per-frame timing metrics…" />
      ) : error ? (
        <StatePanel
          variant={error.status === 409 ? 'unavailable' : error.isNetworkError ? 'disconnected' : 'error'}
          title={error.status === 409 ? 'Telemetry unavailable before completion' : 'Failed to load performance data'}
          message={error.message}
        />
      ) : perf && (
        <>
          <div className="rv-kpi-grid">
            <div className="rv-kpi-card">
              <div className="rv-kpi-top"><span>Sustained FPS</span><Gauge size={16} className="rv-icon-cyan" /></div>
              <div className="rv-kpi-val">{formatNullable(telemetry.sustained_processing_fps, 2)}</div>
              <div className="rv-kpi-sub">Frames processed: {formatNullable(telemetry.processed_frames)}</div>
            </div>
            <div className="rv-kpi-card">
              <div className="rv-kpi-top"><span>Worker Execution</span><Clock size={16} className="rv-icon-cyan" /></div>
              <div className="rv-kpi-val">{formatSeconds(telemetry.total_worker_execution_sec)}</div>
              <div className="rv-kpi-sub">Queue wait: {formatSeconds(telemetry.queue_wait_sec)}</div>
            </div>
            <div className="rv-kpi-card">
              <div className="rv-kpi-top"><span>Model Init / Video Loop</span><Activity size={16} className="rv-icon-cyan" /></div>
              <div className="rv-kpi-val">{formatSeconds(telemetry.video_processing_sec)}</div>
              <div className="rv-kpi-sub">Init: {formatSeconds(telemetry.model_init_sec)}</div>
            </div>
            <div className="rv-kpi-card">
              <div className="rv-kpi-top"><span>Peak GPU Allocated</span><Cpu size={16} className="rv-icon-cyan" /></div>
              <div className="rv-kpi-val">{formatNullable(telemetry.peak_allocated_gpu_mb, 1, ' MiB')}</div>
              <div className="rv-kpi-sub">Reserved: {formatNullable(telemetry.peak_reserved_gpu_mb, 1, ' MiB')}</div>
            </div>
          </div>

          <section className="rv-card">
            <div className="rv-card-header">
              <h3 className="rv-card-title">Mean Stage Latency (Current Page: {frames.length} frames)</h3>
              <span className="rv-card-meta">
                Depth executions: {formatNullable(telemetry.actual_depth_executions)} / Requests: {formatNullable(telemetry.scheduler_depth_requests)}
              </span>
            </div>

            {frames.length === 0 ? (
              <StatePanel
                variant="unavailable"
                title="Per-frame CSV not available"
                message="Per-frame stage breakdown requires save_csv=true during job submission."
              />
            ) : (
              <div className="rv-bar-list">
                {stageAverages.map((st) => (
                  <div key={st.key} className="rv-bar-row">
                    <span className="rv-bar-label">{st.label}</span>
                    <div className="rv-bar-track">
                      <div
                        className="rv-bar-fill"
                        style={{ width: st.avg !== null ? `${Math.min(100, (st.avg / maxStageMs) * 100)}%` : '0%' }}
                      />
                    </div>
                    <span className="rv-bar-val">{st.avg !== null ? `${st.avg.toFixed(2)} ms` : 'Unavailable'}</span>
                  </div>
                ))}
              </div>
            )}
          </section>

          {frames.length > 0 && (
            <section className="rv-card">
              <div className="rv-card-header">
                <h3 className="rv-card-title">Per-Frame Stage Timings (ms)</h3>
                <span className="rv-card-meta">{totalRows} total frame records</span>
              </div>
              <div className="rv-table-wrap">
                <table className="rv-table">
                  <thead>
                    <tr>
                      <th>Frame</th>
                      <th>Decode</th>
                      <th>Detect+Track</th>
                      <th>Scheduler</th>
                      <th>Depth Inf</th>
                      <th>Extract</th>
                      <th>Visualize</th>
                      <th>Total (ms)</th>
                    </tr>
                  </thead>
                  <tbody>
                    {frames.map((row, idx) => (
                      <tr key={`${row.frame_idx}-${idx}`}>
                        <td className="rv-mono">{formatNullable(row.frame_idx)}</td>
                        <td className="rv-mono">{formatNullable(row.decode_ms, 2)}</td>
                        <td className="rv-mono">{formatNullable(row.detect_track_ms, 2)}</td>
                        <td className="rv-mono">{formatNullable(row.scheduler_ms, 2)}</td>
                        <td className="rv-mono">{formatNullable(row.depth_inf_ms, 2)}</td>
                        <td className="rv-mono">{formatNullable(row.extract_ms, 2)}</td>
                        <td className="rv-mono">{formatNullable(row.visualize_ms, 2)}</td>
                        <td className="rv-mono" style={{ fontWeight: 700 }}>{formatNullable(row.total_frame_ms, 2)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>

              <div className="rv-pagination">
                <span>Page {page} of {totalPages}</span>
                <div className="rv-pagination-btns">
                  <button
                    type="button"
                    className="rv-btn rv-btn-ghost rv-btn-sm"
                    disabled={page <= 1}
                    onClick={() => setPage((p) => Math.max(1, p - 1))}
                  >
                    <ChevronLeft size={14} /> Prev
                  </button>
                  <button
                    type="button"
                    className="rv-btn rv-btn-ghost rv-btn-sm"
                    disabled={page >= totalPages}
                    onClick={() => setPage((p) => p + 1)}
                  >
                    Next <ChevronRight size={14} />
                  </button>
                </div>
              </div>
            </section>
          )}
        </>
      )}
    </div>
  );
}
