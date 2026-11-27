import React, { useEffect, useState, useRef, useCallback } from 'react';
import {
  getJob,
  cancelJob,
  getSummary,
  getObservations,
  getTracks,
  getPerformance,
  getArtifacts,
  getArtifactUrl,
  downloadArtifact,
  getGeometrySnapshots,
  getJobRoute,
} from '../api/client';
import {
  XCircle,
  Download,
  ArrowRight,
  Video,
  FileSpreadsheet,
  AlertTriangle,
  Box,
  MapPin,
} from 'lucide-react';
import InspectionSelector from '../components/InspectionSelector';
import {
  StatePanel,
  StatusBadge,
  MeasurementStateBadge,
  QualityBadge,
  ScientificDisclaimer,
  MeasurementStateDistribution,
} from '../components/StateBanner';
import {
  ACTIVE_JOB_STATUSES,
  TERMINAL_JOB_STATUSES,
  isValidUuid,
  formatNullable,
  formatRatePercent,
  formatBytes,
  formatSeconds,
  formatDateTime,
  downloadBlobFile,
} from '../utils/formatters';

export default function Workspace({ selectedJobId, onSelectJob, onNavigate }) {
  const [job, setJob] = useState(null);
  const [summaryData, setSummaryData] = useState(null);
  const [observations, setObservations] = useState([]);
  const [obsTotal, setObsTotal] = useState(0);
  const [tracks, setTracks] = useState([]);
  const [perfData, setPerfData] = useState(null);
  const [artifacts, setArtifacts] = useState([]);
  const [partialArtifacts, setPartialArtifacts] = useState(false);
  const [geomInfo, setGeomInfo] = useState(null);
  const [routeInfo, setRouteInfo] = useState(null);

  const [activeTab, setActiveTab] = useState('overview');
  const [loading, setLoading] = useState(false);
  const [cancelling, setCancelling] = useState(false);
  const [error, setError] = useState(null);
  const [downloadError, setDownloadError] = useState(null);

  const pollTimerRef = useRef(null);
  const resultsLoadedForRef = useRef(null);

  const stopPolling = useCallback(() => {
    if (pollTimerRef.current) {
      clearInterval(pollTimerRef.current);
      pollTimerRef.current = null;
    }
  }, []);

  const loadCompletedResults = useCallback(async (jid) => {
    if (resultsLoadedForRef.current === jid) return;
    const [sumResp, artResp, obsResp, trkResp, perfResp, geomResp, routeResp] = await Promise.all([
      getSummary(jid).catch(() => null),
      getArtifacts(jid).catch(() => null),
      getObservations(jid, { page: 1, page_size: 20 }).catch(() => null),
      getTracks(jid, { page: 1, page_size: 20 }).catch(() => null),
      getPerformance(jid, { page: 1, page_size: 20 }).catch(() => null),
      getGeometrySnapshots(jid).catch(() => null),
      getJobRoute(jid).catch(() => null),
    ]);
    resultsLoadedForRef.current = jid;
    setSummaryData(sumResp);
    setArtifacts(artResp?.artifacts || []);
    setPartialArtifacts(Boolean(artResp?.partial));
    setObservations(obsResp?.observations || []);
    setObsTotal(obsResp?.total ?? 0);
    setTracks(trkResp?.tracks || []);
    setPerfData(perfResp);
    setGeomInfo(geomResp);
    setRouteInfo(routeResp);
  }, []);

  const loadTerminalPartialArtifacts = useCallback(async (jid) => {
    const artResp = await getArtifacts(jid).catch(() => null);
    setArtifacts(artResp?.artifacts || []);
    setPartialArtifacts(Boolean(artResp?.partial));
  }, []);

  useEffect(() => {
    stopPolling();
    setError(null);
    setDownloadError(null);
    setJob(null);
    setSummaryData(null);
    setObservations([]);
    setObsTotal(0);
    setTracks([]);
    setPerfData(null);
    setArtifacts([]);
    setGeomInfo(null);
    setRouteInfo(null);
    resultsLoadedForRef.current = null;

    if (!selectedJobId) return undefined;
    if (!isValidUuid(selectedJobId)) {
      setError(new Error(`Invalid inspection job ID format: "${selectedJobId}". Select a valid inspection from History.`));
      return undefined;
    }

    let cancelledEffect = false;
    setLoading(true);

    async function fetchState() {
      try {
        const j = await getJob(selectedJobId);
        if (cancelledEffect) return;
        setJob(j);
        setError(null);

        if (j.status === 'COMPLETED') {
          stopPolling();
          await loadCompletedResults(selectedJobId);
        } else if (TERMINAL_JOB_STATUSES.has(j.status)) {
          stopPolling();
          await loadTerminalPartialArtifacts(selectedJobId);
        }
      } catch (err) {
        if (cancelledEffect) return;
        setError(err);
        stopPolling();
      } finally {
        if (!cancelledEffect) setLoading(false);
      }
    }

    fetchState();
    pollTimerRef.current = setInterval(fetchState, 1500);

    return () => {
      cancelledEffect = true;
      stopPolling();
    };
  }, [selectedJobId, stopPolling, loadCompletedResults, loadTerminalPartialArtifacts]);

  const handleCancel = async () => {
    if (!selectedJobId || cancelling) return;
    setCancelling(true);
    try {
      const resp = await cancelJob(selectedJobId);
      setJob((prev) => (prev ? { ...prev, status: resp.status, cancel_requested: resp.cancel_requested } : prev));
    } catch (err) {
      setError(err);
    } finally {
      setCancelling(false);
    }
  };

  const handleDownload = async (artifactId, filename) => {
    setDownloadError(null);
    try {
      const blob = await downloadArtifact(selectedJobId, artifactId);
      downloadBlobFile(blob, filename || artifactId);
    } catch (err) {
      setDownloadError(err.message);
    }
  };

  const annotatedEntry = artifacts.find((a) => a.artifact_id === 'annotated_video');
  const sessionSummary = summaryData?.session_summary || {};
  const coverage = summaryData?.measurement_coverage || job?.measurement_coverage || {};
  const telemetry = summaryData?.telemetry || perfData?.telemetry || null;

  return (
    <div className="rv-page">
      <div className="rv-page-header">
        <div>
          <h2 className="rv-page-title">Inspection Workspace</h2>
          <p className="rv-page-subtitle">
            Monitor live progress, inspect annotated video, review uncalibrated depth states, and download artifacts.
          </p>
        </div>
      </div>

      <InspectionSelector
        selectedJobId={isValidUuid(selectedJobId) ? selectedJobId : ''}
        onSelectJob={onSelectJob}
        onNavigate={onNavigate}
      />

      {!selectedJobId && (
        <StatePanel
          variant="empty"
          title="No inspection selected"
          message="Select an inspection above or submit a new road video to inspect results."
          action={
            <button type="button" className="rv-btn rv-btn-primary rv-btn-sm" onClick={() => onNavigate('new')}>
              New Inspection
            </button>
          }
        />
      )}

      {selectedJobId && loading && !job && (
        <StatePanel variant="loading" title="Loading inspection workspace" message="Fetching job status and metadata…" />
      )}

      {error && (
        <StatePanel
          variant={error.isApiDisabled ? 'api-disabled' : error.isNetworkError ? 'disconnected' : 'error'}
          title={error.isNetworkError ? 'Backend disconnected' : 'Inspection error'}
          message={error.message}
          action={
            <button type="button" className="rv-btn rv-btn-secondary rv-btn-sm" onClick={() => onNavigate('history')}>
              View History
            </button>
          }
        />
      )}

      {job && (
        <>
          <section className="rv-card">
            <div className="rv-card-header">
              <div>
                <h3 className="rv-card-title">
                  <span>{job.original_filename || 'Inspection Video'}</span>
                  <StatusBadge status={job.status} />
                  <span className="rv-badge rv-badge-neutral">{(job.mode || '').toUpperCase()}</span>
                </h3>
                <div className="rv-mono" style={{ fontSize: 11, color: 'var(--rv-text-muted)', marginTop: 4 }}>
                  Job ID: {job.job_id} · Created {formatDateTime(job.created_at)}
                </div>
              </div>

              <div className="rv-header-actions">
                {ACTIVE_JOB_STATUSES.has(job.status) && (
                  <button
                    type="button"
                    className="rv-btn rv-btn-danger rv-btn-sm"
                    onClick={handleCancel}
                    disabled={cancelling || job.cancel_requested}
                  >
                    <XCircle size={14} />
                    <span>{job.cancel_requested ? 'Cancelling…' : 'Cancel Job'}</span>
                  </button>
                )}
                {job.status === 'COMPLETED' && (
                  <>
                    <button
                      type="button"
                      className="rv-btn rv-btn-secondary rv-btn-sm"
                      onClick={() => onNavigate('relative3d', job.job_id)}
                    >
                      <Box size={14} /> 3D Surface ({geomInfo?.snapshot_count ?? (job.geometry_available ? 'Ready' : 0)})
                    </button>
                    <button
                      type="button"
                      className="rv-btn rv-btn-secondary rv-btn-sm"
                      onClick={() => onNavigate('geospatial', job.job_id)}
                    >
                      <MapPin size={14} /> GPS Map ({routeInfo?.gps_available || job.gps_available ? 'Synced' : 'No GPS'})
                    </button>
                    <button
                      type="button"
                      className="rv-btn rv-btn-secondary rv-btn-sm"
                      onClick={() => onNavigate('reports', job.job_id)}
                    >
                      <FileSpreadsheet size={14} /> Report Centre
                    </button>
                  </>
                )}
              </div>
            </div>

            {ACTIVE_JOB_STATUSES.has(job.status) && (
              <div className="rv-progress-wrap">
                <div className="rv-progress-meta">
                  <span>
                    Processed {job.frames_processed ?? 0}
                    {job.total_frames ? ` / ${job.total_frames} frames` : ' frames'}
                    {job.throughput_fps !== null && job.throughput_fps !== undefined
                      ? ` · ${job.throughput_fps.toFixed(1)} FPS`
                      : ''}
                  </span>
                  <span className="rv-mono">
                    {job.progress_percentage !== null && job.progress_percentage !== undefined
                      ? `${job.progress_percentage.toFixed(1)}%`
                      : job.status}
                  </span>
                </div>
                <div className="rv-progress-track">
                  <div
                    className="rv-progress-fill"
                    style={{ width: `${Math.min(100, Math.max(4, Number(job.progress_percentage || 0)))}%` }}
                  />
                </div>
              </div>
            )}

            {job.latest_warning && (
              <div className="rv-state-panel rv-state-api-disabled" style={{ marginTop: 12, padding: 12 }}>
                <AlertTriangle size={16} className="rv-icon-warning" />
                <div className="rv-state-body">
                  <div className="rv-state-message">{job.latest_warning}</div>
                </div>
              </div>
            )}

            {job.status === 'FAILED' && (
              <div className="rv-state-panel rv-state-error" style={{ marginTop: 12 }}>
                <AlertTriangle size={18} className="rv-icon-danger" />
                <div className="rv-state-body">
                  <h4 className="rv-state-title">Job Failed ({job.error?.code || 'ERROR'})</h4>
                  <p className="rv-state-message">{job.error?.message || 'Processing failed on worker.'}</p>
                </div>
              </div>
            )}

            {job.status === 'CANCELLED' && (
              <div className="rv-state-panel" style={{ marginTop: 12 }}>
                <div className="rv-state-body">
                  <h4 className="rv-state-title">Job Cancelled</h4>
                  <p className="rv-state-message">
                    Inspection stopped cooperatively after {job.frames_processed ?? 0} frames. Full summary metrics are unavailable for cancelled runs; any partial artifacts are listed below.
                  </p>
                </div>
              </div>
            )}
          </section>

          <div className="rv-kpi-grid">
            <div className="rv-kpi-card">
              <div className="rv-kpi-top"><span>Frames Processed</span></div>
              <div className="rv-kpi-val">{job.frames_processed ?? 0}</div>
              <div className="rv-kpi-sub">Total: {formatNullable(job.total_frames)}</div>
            </div>
            <div className="rv-kpi-card">
              <div className="rv-kpi-top"><span>Candidate Observations</span></div>
              <div className="rv-kpi-val">{job.candidate_observation_count ?? 0}</div>
              <div className="rv-kpi-sub">Per-frame detections</div>
            </div>
            <div className="rv-kpi-card">
              <div className="rv-kpi-top"><span>Session Candidate Tracks</span></div>
              <div className="rv-kpi-val">{job.unique_candidate_track_count ?? 0}</div>
              <div className="rv-kpi-sub">Unique ByteTrack IDs</div>
            </div>
            <div className="rv-kpi-card">
              <div className="rv-kpi-top"><span>Fresh Valid Rate</span></div>
              <div className="rv-kpi-val">{formatRatePercent(coverage?.fresh_valid_rate_over_observations)}</div>
              <div className="rv-kpi-sub">Cached: {formatRatePercent(coverage?.cached_valid_rate_over_observations)}</div>
            </div>
            <div className="rv-kpi-card">
              <div className="rv-kpi-top"><span>Sustained FPS</span></div>
              <div className="rv-kpi-val">{formatNullable(job.throughput_fps, 1)}</div>
              <div className="rv-kpi-sub">Depth calls: {job.depth_invocation_count ?? 0}</div>
            </div>
          </div>

          {job.status === 'COMPLETED' && (
            <>
              <div className="rv-tabs" role="tablist">
                {[
                  { id: 'overview', label: 'Video & Summary' },
                  { id: 'observations', label: `Observations (${obsTotal})` },
                  { id: 'tracks', label: `Tracks (${job.unique_candidate_track_count ?? tracks.length})` },
                  { id: 'performance', label: 'Performance' },
                  { id: 'artifacts', label: `Artifacts (${artifacts.length})` },
                ].map((tab) => (
                  <button
                    key={tab.id}
                    type="button"
                    role="tab"
                    aria-selected={activeTab === tab.id}
                    className={`rv-tab ${activeTab === tab.id ? 'active' : ''}`}
                    onClick={() => setActiveTab(tab.id)}
                  >
                    {tab.label}
                  </button>
                ))}
              </div>

              {activeTab === 'overview' && (
                <div className="rv-grid-2">
                  <section className="rv-card">
                    <div className="rv-card-header">
                      <h3 className="rv-card-title"><Video size={16} className="rv-icon-cyan" /> Annotated Inspection Video</h3>
                      {annotatedEntry && (
                        <button
                          type="button"
                          className="rv-btn rv-btn-secondary rv-btn-sm"
                          onClick={() => handleDownload(annotatedEntry.artifact_id, annotatedEntry.filename)}
                        >
                          <Download size={14} /> Download MP4
                        </button>
                      )}
                    </div>
                    {annotatedEntry ? (
                      <>
                        <video
                          className="rv-video-player"
                          controls
                          preload="metadata"
                          src={getArtifactUrl(job.job_id, annotatedEntry.artifact_id)}
                        />
                        <div style={{ marginTop: 8, fontSize: 12, color: 'var(--rv-text-secondary)' }}>
                          Codec: {annotatedEntry.codec || 'Unavailable'} · Size: {formatBytes(annotatedEntry.size_bytes)}
                          {annotatedEntry.browser_compatible === false && ' · Browser playback may require download if H.264 transcode was unavailable.'}
                        </div>
                      </>
                    ) : (
                      <StatePanel
                        variant="unavailable"
                        title="Annotated video unavailable"
                        message="Annotated video output was disabled or not generated for this inspection."
                      />
                    )}
                  </section>

                  <section className="rv-card">
                    <div className="rv-card-header">
                      <h3 className="rv-card-title">Session Summary & Coverage</h3>
                      <span className="rv-card-meta">v{job.pipeline_version || '2.0'}</span>
                    </div>
                    <MeasurementStateDistribution
                      countsByState={coverage?.counts_by_state || job.measurement_state_counts}
                      totalObservations={coverage?.denominator_total_observations ?? job.candidate_observation_count ?? 0}
                    />
                    <div className="rv-kv-list" style={{ marginTop: 14 }}>
                      <div className="rv-kv-item">
                        <span className="rv-kv-label">Depth Model</span>
                        <span className="rv-kv-value">
                          {sessionSummary.depth_enabled ? `${sessionSummary.depth_model_id} (${sessionSummary.depth_input_size}px)` : 'Disabled'}
                        </span>
                      </div>
                      <div className="rv-kv-item">
                        <span className="rv-kv-label">Scheduler Strategy</span>
                        <span className="rv-kv-value">{formatNullable(sessionSummary.scheduler_strategy)}</span>
                      </div>
                      <div className="rv-kv-item">
                        <span className="rv-kv-label">Depth Executions / Requests</span>
                        <span className="rv-kv-value">
                          {sessionSummary.depth_calls ?? 0} / {sessionSummary.scheduler_depth_requests ?? 0}
                        </span>
                      </div>
                      <div className="rv-kv-item">
                        <span className="rv-kv-label">Elapsed Processing</span>
                        <span className="rv-kv-value">{formatSeconds(sessionSummary.elapsed_sec)}</span>
                      </div>
                      <div className="rv-kv-item">
                        <span className="rv-kv-label">Median Fresh Quality (Heuristic)</span>
                        <span className="rv-kv-value">
                          {formatNullable(summaryData?.quality_summary?.median_fresh_quality_index, 3)}
                        </span>
                      </div>
                      <div className="rv-kv-item">
                        <span className="rv-kv-label">2.5D Snapshots / GPS Route</span>
                        <span className="rv-kv-value">
                          {geomInfo?.geometry_available ? `${geomInfo.snapshot_count} snapshots` : 'Geometry off'} ·{' '}
                          {routeInfo?.gps_available ? `${routeInfo.metadata?.sample_count ?? 0} GPS pts` : 'No GPS'}
                        </span>
                      </div>
                    </div>
                  </section>
                </div>
              )}

              {activeTab === 'observations' && (
                <section className="rv-card">
                  <div className="rv-card-header">
                    <h3 className="rv-card-title">Recent Observations Preview</h3>
                    <button
                      type="button"
                      className="rv-btn rv-btn-secondary rv-btn-sm"
                      onClick={() => onNavigate('hazard', job.job_id)}
                    >
                      <span>Open Hazard Explorer</span> <ArrowRight size={14} />
                    </button>
                  </div>
                  {observations.length === 0 ? (
                    <StatePanel variant="empty" title="No observations recorded" message="No candidate potholes were detected in this inspection." />
                  ) : (
                    <div className="rv-table-wrap">
                      <table className="rv-table">
                        <thead>
                          <tr>
                            <th>Frame</th>
                            <th>Time (s)</th>
                            <th>Track ID</th>
                            <th>Confidence</th>
                            <th>State</th>
                            <th>Quality (Heuristic)</th>
                            <th>Raw Rel. Score</th>
                            <th>Smoothed Rel. Score</th>
                          </tr>
                        </thead>
                        <tbody>
                          {observations.map((obs, idx) => {
                            const fc = obs.frame_context || {};
                            const det = obs.detection || {};
                            const meas = obs.measurement || {};
                            const qa = meas.quality_assessment;
                            return (
                              <tr key={`${fc.frame_idx}-${det.track_id}-${idx}`}>
                                <td className="rv-mono">{formatNullable(fc.frame_idx)}</td>
                                <td className="rv-mono">{formatNullable(fc.timestamp_sec, 2)}</td>
                                <td className="rv-mono">#{formatNullable(det.track_id)}</td>
                                <td className="rv-mono">
                                  {det.detector_confidence !== null && det.detector_confidence !== undefined
                                    ? `${(det.detector_confidence * 100).toFixed(1)}%`
                                    : 'Unavailable'}
                                </td>
                                <td><MeasurementStateBadge state={meas.measurement_state} /></td>
                                <td>
                                  {qa ? <QualityBadge category={qa.quality_category} /> : 'Unavailable'}
                                </td>
                                <td className="rv-mono">{formatNullable(meas.raw_relative_cavity_score, 4)}</td>
                                <td className="rv-mono">{formatNullable(meas.smoothed_relative_score, 4)}</td>
                              </tr>
                            );
                          })}
                        </tbody>
                      </table>
                    </div>
                  )}
                </section>
              )}

              {activeTab === 'tracks' && (
                <section className="rv-card">
                  <div className="rv-card-header">
                    <h3 className="rv-card-title">Session ByteTrack Preview</h3>
                    <button
                      type="button"
                      className="rv-btn rv-btn-secondary rv-btn-sm"
                      onClick={() => onNavigate('track', job.job_id)}
                    >
                      <span>Open Track Analysis</span> <ArrowRight size={14} />
                    </button>
                  </div>
                  {tracks.length === 0 ? (
                    <StatePanel variant="empty" title="No tracks recorded" message="No candidate tracks were formed in this inspection." />
                  ) : (
                    <div className="rv-table-wrap">
                      <table className="rv-table">
                        <thead>
                          <tr>
                            <th>Track ID</th>
                            <th>Frame Span</th>
                            <th>Observations</th>
                            <th>Latest State</th>
                          </tr>
                        </thead>
                        <tbody>
                          {tracks.map((trk) => (
                            <tr key={`${trk.track_id}-${trk.track_generation_id}`}>
                              <td className="rv-mono">#{trk.track_id}</td>
                              <td className="rv-mono">
                                {trk.first_seen_frame_idx} → {trk.last_seen_frame_idx}
                              </td>
                              <td className="rv-mono">{trk.observation_count}</td>
                              <td><MeasurementStateBadge state={trk.latest_state} /></td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  )}
                </section>
              )}

              {activeTab === 'performance' && (
                <section className="rv-card">
                  <div className="rv-card-header">
                    <h3 className="rv-card-title">Execution Telemetry Summary</h3>
                    <button
                      type="button"
                      className="rv-btn rv-btn-secondary rv-btn-sm"
                      onClick={() => onNavigate('performance', job.job_id)}
                    >
                      <span>Open Full Performance</span> <ArrowRight size={14} />
                    </button>
                  </div>
                  {telemetry ? (
                    <div className="rv-kv-list">
                      <div className="rv-kv-item">
                        <span className="rv-kv-label">Model Init</span>
                        <span className="rv-kv-value">{formatSeconds(telemetry.model_init_sec)}</span>
                      </div>
                      <div className="rv-kv-item">
                        <span className="rv-kv-label">Video Processing</span>
                        <span className="rv-kv-value">{formatSeconds(telemetry.video_processing_sec)}</span>
                      </div>
                      <div className="rv-kv-item">
                        <span className="rv-kv-label">Total Worker Execution</span>
                        <span className="rv-kv-value">{formatSeconds(telemetry.total_worker_execution_sec)}</span>
                      </div>
                      <div className="rv-kv-item">
                        <span className="rv-kv-label">Peak GPU Allocated</span>
                        <span className="rv-kv-value">{formatNullable(telemetry.peak_allocated_gpu_mb, 1, ' MiB')}</span>
                      </div>
                    </div>
                  ) : (
                    <StatePanel variant="unavailable" title="Telemetry unavailable" message="Performance telemetry was not recorded." />
                  )}
                </section>
              )}

              {activeTab === 'artifacts' && (
                <section className="rv-card">
                  <div className="rv-card-header">
                    <h3 className="rv-card-title">Published Artifacts</h3>
                    {partialArtifacts && <span className="rv-badge rv-badge-warning">Partial Artifacts</span>}
                  </div>
                  {downloadError && (
                    <div className="rv-state-panel rv-state-error" style={{ marginBottom: 12 }}>
                      <div className="rv-state-body"><p className="rv-state-message">{downloadError}</p></div>
                    </div>
                  )}
                  {artifacts.length === 0 ? (
                    <StatePanel variant="empty" title="No artifacts available" message="No files were generated for this job." />
                  ) : (
                    <div className="rv-artifact-list">
                      {artifacts.map((art) => (
                        <div key={art.artifact_id} className="rv-artifact-row">
                          <div className="rv-artifact-info">
                            <span className="rv-artifact-name">{art.filename}</span>
                            <span className="rv-artifact-meta">
                              {art.artifact_id} · {art.mime_type} · {formatBytes(art.size_bytes)}
                            </span>
                          </div>
                          <button
                            type="button"
                            className="rv-btn rv-btn-secondary rv-btn-sm"
                            onClick={() => handleDownload(art.artifact_id, art.filename)}
                          >
                            <Download size={14} /> Download
                          </button>
                        </div>
                      ))}
                    </div>
                  )}
                </section>
              )}
            </>
          )}

          {job.status !== 'COMPLETED' && artifacts.length > 0 && (
            <section className="rv-card">
              <div className="rv-card-header">
                <h3 className="rv-card-title">Partial Artifacts</h3>
                <span className="rv-badge rv-badge-warning">{job.status}</span>
              </div>
              <div className="rv-artifact-list">
                {artifacts.map((art) => (
                  <div key={art.artifact_id} className="rv-artifact-row">
                    <div className="rv-artifact-info">
                      <span className="rv-artifact-name">{art.filename}</span>
                      <span className="rv-artifact-meta">{art.artifact_id} · {formatBytes(art.size_bytes)}</span>
                    </div>
                    <button
                      type="button"
                      className="rv-btn rv-btn-secondary rv-btn-sm"
                      onClick={() => handleDownload(art.artifact_id, art.filename)}
                    >
                      <Download size={14} /> Download
                    </button>
                  </div>
                ))}
              </div>
            </section>
          )}

          <ScientificDisclaimer text={job.scientific_disclaimer} />
        </>
      )}
    </div>
  );
}
