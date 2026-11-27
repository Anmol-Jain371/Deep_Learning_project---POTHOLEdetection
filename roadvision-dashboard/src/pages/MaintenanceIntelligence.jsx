import React, { useState, useEffect, useCallback } from 'react';
import {
  getMaintenanceOverview,
  getMaintenanceReviewQueue,
  getMaintenanceReview,
  createMaintenanceReview,
  updateMaintenanceReview,
  getJobSegments,
  getJobRoute,
  listJobs,
  compareInspections,
  getGeometrySnapshot,
} from '../api/client';
import {
  ClipboardCheck,
  ListChecks,
  Route,
  GitCompare,
  AlertTriangle,
  ExternalLink,
  CheckCircle2,
  XCircle,
  Search,
  MapPin,
} from 'lucide-react';
import InspectionSelector from '../components/InspectionSelector';
import {
  StatePanel,
  QualityBadge,
  ProvenanceBadge,
  QualityBreakdownCard,
  ScientificDisclaimer,
} from '../components/StateBanner';
import {
  QUALITY_CATEGORIES,
  REVIEW_PRIORITY_CATEGORIES,
  REVIEW_STATUSES,
  isValidUuid,
  formatNullable,
  formatRatePercent,
  formatDateTime,
  reviewPriorityTone,
  reviewStatusTone,
} from '../utils/formatters';

function PriorityBadge({ category }) {
  const label = category || 'UNAVAILABLE';
  return <span className={`rv-badge rv-badge-${reviewPriorityTone(label)}`}>{label}</span>;
}

function ReviewStatusBadge({ status }) {
  const label = status || 'PENDING_REVIEW';
  return <span className={`rv-badge rv-badge-${reviewStatusTone(label)}`}>{label}</span>;
}

function SegmentRouteSvg({ routePoints, segments }) {
  const pts = (routePoints || []).filter((p) => p.lat !== null && p.lon !== null);
  const segPts = (segments || []).filter((s) => s.centroid_lat !== null && s.centroid_lon !== null);
  const all = [
    ...pts.map((p) => ({ lat: p.lat, lon: p.lon })),
    ...segPts.map((s) => ({ lat: s.centroid_lat, lon: s.centroid_lon })),
  ];
  if (all.length === 0) return null;

  const lats = all.map((c) => c.lat);
  const lons = all.map((c) => c.lon);
  const minLat = Math.min(...lats);
  const maxLat = Math.max(...lats);
  const minLon = Math.min(...lons);
  const maxLon = Math.max(...lons);
  const dLat = Math.max(maxLat - minLat, 0.0004);
  const dLon = Math.max(maxLon - minLon, 0.0004);

  const W = 680;
  const H = 260;
  const PAD = 28;
  const proj = (lat, lon) => ({
    x: PAD + ((lon - minLon) / dLon) * (W - 2 * PAD),
    y: H - PAD - ((lat - minLat) / dLat) * (H - 2 * PAD),
  });

  const polyline = pts
    .map((p) => {
      const xy = proj(p.lat, p.lon);
      return `${xy.x.toFixed(1)},${xy.y.toFixed(1)}`;
    })
    .join(' ');

  return (
    <div className="rv-canvas-box" style={{ marginBottom: 14 }}>
      <div style={{ padding: '6px 12px', fontSize: 11, color: '#94a3b8', borderBottom: '1px solid var(--rv-border)' }}>
        Genuine Vehicle GPS Trajectory &amp; Distance Segment Centroids (WGS84 local bounds)
      </div>
      <svg viewBox={`0 0 ${W} ${H}`} style={{ width: '100%', height: 240, display: 'block', background: '#070b13' }}>
        {pts.length >= 2 && <polyline fill="none" stroke="#0891b2" strokeWidth="2.5" points={polyline} />}
        {segPts.map((seg) => {
          const xy = proj(seg.centroid_lat, seg.centroid_lon);
          const count = seg.session_candidate_track_count || 0;
          return (
            <g key={seg.segment_id}>
              <circle
                cx={xy.x}
                cy={xy.y}
                r={count > 0 ? 7 : 4}
                fill={count > 0 ? '#f59e0b' : '#22d3ee'}
                stroke="#0f172a"
                strokeWidth="1.5"
              >
                <title>{`${seg.segment_id} (${seg.start_m}–${seg.end_m}m): ${count} session tracks`}</title>
              </circle>
            </g>
          );
        })}
      </svg>
    </div>
  );
}

export default function MaintenanceIntelligence({ selectedJobId, onSelectJob, onNavigate }) {
  const [subTab, setSubTab] = useState('overview'); // overview | queue | segments | compare
  const [provenanceScope, setProvenanceScope] = useState('genuine'); // genuine | exploratory | all

  // Overview state
  const [overview, setOverview] = useState(null);
  const [loadingOverview, setLoadingOverview] = useState(false);

  // Queue & Review state
  const [queueData, setQueueData] = useState(null);
  const [loadingQueue, setLoadingQueue] = useState(false);
  const [priorityFilter, setPriorityFilter] = useState('');
  const [reviewFilter, setReviewFilter] = useState('');
  const [qualityFilter, setQualityFilter] = useState('');
  const [activeCandidate, setActiveCandidate] = useState(null);
  const [candidateCropUrl, setCandidateCropUrl] = useState(null);
  const [activeReviewDetail, setActiveReviewDetail] = useState(null);
  const [formStatus, setFormStatus] = useState('CONFIRMED_VISUALLY');
  const [formLabel, setFormLabel] = useState('');
  const [formNotes, setFormNotes] = useState('');
  const [savingReview, setSavingReview] = useState(false);
  const [reviewSaveMsg, setReviewSaveMsg] = useState(null);

  // Segments state
  const [segmentLen, setSegmentLen] = useState('50');
  const [segmentData, setSegmentData] = useState(null);
  const [routeData, setRouteData] = useState(null);
  const [loadingSegments, setLoadingSegments] = useState(false);

  // Historical comparison state
  const [completedJobs, setCompletedJobs] = useState([]);
  const [compareIds, setCompareIds] = useState([]);
  const [compareResult, setCompareResult] = useState(null);
  const [loadingCompare, setLoadingCompare] = useState(false);

  const [error, setError] = useState(null);

  const loadOverview = useCallback(async () => {
    setLoadingOverview(true);
    setError(null);
    try {
      const ov = await getMaintenanceOverview({ scope: provenanceScope });
      setOverview(ov);
    } catch (err) {
      setError(err);
    } finally {
      setLoadingOverview(false);
    }
  }, [provenanceScope]);

  const loadQueue = useCallback(async () => {
    setLoadingQueue(true);
    setError(null);
    try {
      const res = await getMaintenanceReviewQueue({
        scope: provenanceScope,
        ...(isValidUuid(selectedJobId) ? { job_id: selectedJobId } : {}),
        ...(priorityFilter ? { priority_category: priorityFilter } : {}),
        ...(reviewFilter ? { review_status: reviewFilter } : {}),
        ...(qualityFilter ? { quality_category: qualityFilter } : {}),
        page: 1,
        page_size: 50,
      });
      setQueueData(res);
    } catch (err) {
      setError(err);
    } finally {
      setLoadingQueue(false);
    }
  }, [selectedJobId, priorityFilter, reviewFilter, qualityFilter, provenanceScope]);

  const loadSegments = useCallback(async () => {
    if (!selectedJobId || !isValidUuid(selectedJobId)) return;
    setLoadingSegments(true);
    setError(null);
    try {
      const [segRes, routeRes] = await Promise.all([
        getJobSegments(selectedJobId, Number(segmentLen) || 50),
        getJobRoute(selectedJobId).catch(() => null),
      ]);
      setSegmentData(segRes);
      setRouteData(routeRes);
    } catch (err) {
      setError(err);
    } finally {
      setLoadingSegments(false);
    }
  }, [selectedJobId, segmentLen]);

  const loadCompletedForCompare = useCallback(async () => {
    try {
      const resp = await listJobs({ status: 'COMPLETED', page: 1, page_size: 40 });
      const list = resp?.jobs || [];
      setCompletedJobs(list);
      setCompareIds((prev) => {
        if (prev.length >= 2) return prev;
        return list.slice(0, 2).map((j) => j.job_id);
      });
    } catch {
      setCompletedJobs([]);
    }
  }, []);

  useEffect(() => {
    if (subTab === 'overview') loadOverview();
    if (subTab === 'queue') loadQueue();
    if (subTab === 'segments') loadSegments();
    if (subTab === 'compare') loadCompletedForCompare();
  }, [subTab, loadOverview, loadQueue, loadSegments, loadCompletedForCompare]);

  const selectCandidateForReview = async (cand) => {
    setActiveCandidate(cand);
    setReviewSaveMsg(null);
    setCandidateCropUrl(null);
    setActiveReviewDetail(null);
    setFormStatus(cand.review_status && cand.review_status !== 'PENDING_REVIEW' ? cand.review_status : 'CONFIRMED_VISUALLY');
    setFormLabel(cand.reviewer_label || '');
    setFormNotes(cand.review_notes || '');

    if (cand.representative_snapshot_id) {
      getGeometrySnapshot(cand.job_id, cand.representative_snapshot_id)
        .then((snap) => {
          setCandidateCropUrl(snap?.source_crop_data_url || null);
        })
        .catch(() => setCandidateCropUrl(null));
    }
    if (cand.review_id) {
      getMaintenanceReview(cand.review_id)
        .then((rev) => setActiveReviewDetail(rev))
        .catch(() => setActiveReviewDetail(null));
    }
  };

  const handleSubmitReview = async (e) => {
    e.preventDefault();
    if (!activeCandidate || savingReview) return;
    setSavingReview(true);
    setReviewSaveMsg(null);
    try {
      let saved;
      if (activeCandidate.review_id) {
        saved = await updateMaintenanceReview(activeCandidate.review_id, {
          review_status: formStatus,
          reviewer_label: formLabel,
          review_notes: formNotes,
        });
      } else {
        saved = await createMaintenanceReview({
          job_id: activeCandidate.job_id,
          track_id: activeCandidate.track_id,
          track_generation_id: activeCandidate.track_generation_id || 0,
          cluster_id: activeCandidate.cluster_id || null,
          review_status: formStatus,
          reviewer_label: formLabel,
          review_notes: formNotes,
        });
      }
      setActiveReviewDetail(saved);
      setActiveCandidate((prev) =>
        prev
          ? {
              ...prev,
              review_id: saved.review_id,
              review_status: saved.review_status,
              reviewer_label: saved.reviewer_label,
              review_notes: saved.review_notes,
            }
          : prev
      );
      setReviewSaveMsg('Manual review saved and persisted (original automated evidence preserved).');
      await loadQueue();
    } catch (err) {
      setReviewSaveMsg(`Error: ${err.message}`);
    } finally {
      setSavingReview(false);
    }
  };

  const handleRunCompare = async () => {
    if (compareIds.length < 2 || loadingCompare) return;
    setLoadingCompare(true);
    setError(null);
    try {
      const res = await compareInspections(compareIds);
      setCompareResult(res);
    } catch (err) {
      setError(err);
    } finally {
      setLoadingCompare(false);
    }
  };

  const toggleCompareJob = (jid) => {
    setCompareIds((prev) =>
      prev.includes(jid) ? prev.filter((id) => id !== jid) : prev.length < 8 ? [...prev, jid] : prev
    );
  };

  return (
    <div className="rv-page">
      <div className="rv-page-header">
        <div>
          <h2 className="rv-page-title">Road Maintenance Intelligence</h2>
          <p className="rv-page-subtitle">
            Explainable inspection-review prioritization, human-in-the-loop audit workflow, road segment analytics, and compatible session comparisons.
          </p>
        </div>
        <div className="rv-header-actions">
          <select
            aria-label="Maintenance provenance scope"
            className="rv-select"
            value={provenanceScope}
            onChange={(e) => setProvenanceScope(e.target.value)}
          >
            <option value="genuine">Genuine Only (Default)</option>
            <option value="exploratory">Include Exploratory</option>
            <option value="all">All (incl. Test/Synthetic)</option>
          </select>
          <button
            type="button"
            className={`rv-btn rv-btn-sm ${subTab === 'overview' ? 'rv-btn-primary' : 'rv-btn-ghost'}`}
            onClick={() => setSubTab('overview')}
          >
            <ClipboardCheck size={14} /> Overview
          </button>
          <button
            type="button"
            className={`rv-btn rv-btn-sm ${subTab === 'queue' ? 'rv-btn-primary' : 'rv-btn-ghost'}`}
            onClick={() => setSubTab('queue')}
          >
            <ListChecks size={14} /> Review Queue
          </button>
          <button
            type="button"
            className={`rv-btn rv-btn-sm ${subTab === 'segments' ? 'rv-btn-primary' : 'rv-btn-ghost'}`}
            onClick={() => setSubTab('segments')}
          >
            <Route size={14} /> Road Segments
          </button>
          <button
            type="button"
            className={`rv-btn rv-btn-sm ${subTab === 'compare' ? 'rv-btn-primary' : 'rv-btn-ghost'}`}
            onClick={() => setSubTab('compare')}
          >
            <GitCompare size={14} /> Historical Compare
          </button>
        </div>
      </div>

      {error && (
        <StatePanel
          variant={error.isNetworkError ? 'disconnected' : 'error'}
          title="Maintenance request error"
          message={error.message}
        />
      )}

      {/* SUBTAB 1: MAINTENANCE OVERVIEW */}
      {subTab === 'overview' && (
        <>
          {loadingOverview && !overview ? (
            <StatePanel variant="loading" title="Loading maintenance overview" message="Aggregating completed inspection records…" />
          ) : overview && (
            <>
              <div className="rv-kpi-grid">
                <div className="rv-kpi-card">
                  <span className="rv-kpi-label">Completed Inspections ({overview.provenance_scope})</span>
                  <strong className="rv-kpi-value">{overview.completed_inspection_count ?? 0}</strong>
                  <span className="rv-kpi-sub">
                    {overview.jobs_with_gps_count ?? 0} with GPS · {overview.excluded_completed_count ?? 0} non-genuine excluded
                  </span>
                </div>
                <div className="rv-kpi-card">
                  <span className="rv-kpi-label">Pending Review</span>
                  <strong className="rv-kpi-value">
                    {(overview.review_status_distribution?.PENDING_REVIEW ?? 0) +
                      (overview.review_status_distribution?.IN_REVIEW ?? 0)}
                  </strong>
                  <span className="rv-kpi-sub">{overview.total_session_candidate_tracks ?? 0} session candidate tracks</span>
                </div>
                <div className="rv-kpi-card">
                  <span className="rv-kpi-label">Confirmed Visually</span>
                  <strong className="rv-kpi-value">{overview.review_status_distribution?.CONFIRMED_VISUALLY ?? 0}</strong>
                  <span className="rv-kpi-sub">Visual appearance confirmed only</span>
                </div>
                <div className="rv-kpi-card">
                  <span className="rv-kpi-label">Rejected / Field Inspection</span>
                  <strong className="rv-kpi-value">
                    {overview.review_status_distribution?.REJECTED_FALSE_POSITIVE ?? 0} /{' '}
                    {overview.review_status_distribution?.NEEDS_FIELD_INSPECTION ?? 0}
                  </strong>
                  <span className="rv-kpi-sub">Rejected FP / Needs Field</span>
                </div>
              </div>

              <div className="rv-grid-2">
                <section className="rv-card">
                  <div className="rv-card-header">
                    <h3 className="rv-card-title">Review Priority Distribution</h3>
                    <span className="rv-card-meta">Review queue urgency (NOT structural damage)</span>
                  </div>
                  <div className="rv-bar-list">
                    {REVIEW_PRIORITY_CATEGORIES.map((cat) => {
                      const cnt = overview.priority_distribution?.[cat] ?? 0;
                      const tot = Math.max(1, overview.total_session_candidate_tracks || 1);
                      return (
                        <div key={cat} className="rv-bar-row">
                          <span className="rv-bar-label">{cat}</span>
                          <div className="rv-bar-track">
                            <div className="rv-bar-fill" style={{ width: `${Math.round((cnt / tot) * 100)}%` }} />
                          </div>
                          <span className="rv-bar-val">{cnt}</span>
                        </div>
                      );
                    })}
                  </div>
                </section>

                <section className="rv-card">
                  <div className="rv-card-header">
                    <h3 className="rv-card-title">Quality Evidence Distribution</h3>
                    <span className="rv-card-meta">Heuristic evidence quality</span>
                  </div>
                  <div className="rv-bar-list">
                    {Object.entries(overview.quality_distribution || {}).map(([qcat, cnt]) => {
                      const tot = Math.max(1, overview.total_session_candidate_tracks || 1);
                      return (
                        <div key={qcat} className="rv-bar-row">
                          <span className="rv-bar-label">{qcat}</span>
                          <div className="rv-bar-track">
                            <div className="rv-bar-fill" style={{ width: `${Math.round((cnt / tot) * 100)}%` }} />
                          </div>
                          <span className="rv-bar-val">{cnt}</span>
                        </div>
                      );
                    })}
                  </div>
                </section>
              </div>

              <section className="rv-card">
                <div className="rv-card-header">
                  <h3 className="rv-card-title">Inspection Sessions</h3>
                  <span className="rv-card-meta">{overview.prototype_note}</span>
                </div>
                {(overview.inspections || []).length === 0 ? (
                  <StatePanel variant="empty" title="No completed inspections" message="Run an inspection first to populate maintenance analytics." />
                ) : (
                  <div className="rv-table-wrap">
                    <table className="rv-table">
                      <thead>
                        <tr>
                          <th>Video</th>
                          <th>Mode</th>
                          <th>Session Tracks</th>
                          <th>Observations</th>
                          <th>GPS / Geom</th>
                          <th>Actions</th>
                        </tr>
                      </thead>
                      <tbody>
                        {overview.inspections.map((ins) => (
                          <tr key={ins.job_id}>
                            <td>
                              {ins.original_filename || ins.job_id.slice(0, 8)}{' '}
                              <ProvenanceBadge provenance={ins.provenance} />
                            </td>
                            <td><span className="rv-badge rv-badge-neutral">{(ins.mode || '').toUpperCase()}</span></td>
                            <td className="rv-mono">{ins.candidate_count}</td>
                            <td className="rv-mono">{ins.total_observations}</td>
                            <td>
                              {ins.gps_available ? 'GPS Yes' : 'No GPS'} · {ins.geometry_available ? '3D Yes' : 'No 3D'}
                            </td>
                            <td>
                              <button
                                type="button"
                                className="rv-btn rv-btn-secondary rv-btn-sm"
                                onClick={() => {
                                  onSelectJob(ins.job_id);
                                  setSubTab('queue');
                                }}
                              >
                                Review Queue
                              </button>
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                )}
              </section>
            </>
          )}
        </>
      )}

      {/* SUBTAB 2: REVIEW QUEUE & HUMAN-IN-THE-LOOP WORKFLOW */}
      {subTab === 'queue' && (
        <>
          <InspectionSelector
            selectedJobId={isValidUuid(selectedJobId) ? selectedJobId : ''}
            onSelectJob={onSelectJob}
            onNavigate={onNavigate}
            label="Filter Queue by Inspection (optional)"
          />

          <section className="rv-card">
            <div className="rv-filter-bar" style={{ marginBottom: 0 }}>
              <div className="rv-field">
                <label className="rv-field-label" htmlFor="rv-m-prio">Review Priority</label>
                <select
                  id="rv-m-prio"
                  className="rv-select"
                  value={priorityFilter}
                  onChange={(e) => setPriorityFilter(e.target.value)}
                >
                  <option value="">All Priorities</option>
                  {REVIEW_PRIORITY_CATEGORIES.map((p) => (
                    <option key={p} value={p}>{p}</option>
                  ))}
                </select>
              </div>
              <div className="rv-field">
                <label className="rv-field-label" htmlFor="rv-m-rstat">Manual Status</label>
                <select
                  id="rv-m-rstat"
                  className="rv-select"
                  value={reviewFilter}
                  onChange={(e) => setReviewFilter(e.target.value)}
                >
                  <option value="">All Review States</option>
                  {REVIEW_STATUSES.map((s) => (
                    <option key={s} value={s}>{s}</option>
                  ))}
                </select>
              </div>
              <div className="rv-field">
                <label className="rv-field-label" htmlFor="rv-m-qual">Evidence Quality</label>
                <select
                  id="rv-m-qual"
                  className="rv-select"
                  value={qualityFilter}
                  onChange={(e) => setQualityFilter(e.target.value)}
                >
                  <option value="">All Quality Categories</option>
                  {QUALITY_CATEGORIES.map((q) => (
                    <option key={q} value={q}>{q}</option>
                  ))}
                </select>
              </div>
              {selectedJobId && (
                <button
                  type="button"
                  className="rv-btn rv-btn-ghost rv-btn-sm"
                  onClick={() => onSelectJob('')}
                >
                  Show All Inspections
                </button>
              )}
            </div>
          </section>

          {loadingQueue && !queueData ? (
            <StatePanel variant="loading" title="Loading review queue" message="Computing review priorities from stored track evidence…" />
          ) : (queueData?.items?.length || 0) === 0 ? (
            <StatePanel variant="empty" title="No candidates match filters" message="Adjust filters or select another completed inspection." />
          ) : (
            <div className="rv-grid-2">
              <section className="rv-card" style={{ overflow: 'hidden' }}>
                <div className="rv-card-header">
                  <h3 className="rv-card-title">Prioritized Review Queue ({queueData.total})</h3>
                  <span className="rv-card-meta">v{queueData.scoring_version}</span>
                </div>
                <div className="rv-table-wrap">
                  <table className="rv-table">
                    <thead>
                      <tr>
                        <th>Track</th>
                        <th>Priority</th>
                        <th>Quality</th>
                        <th>Manual Status</th>
                        <th>Obs</th>
                        <th>Conf</th>
                      </tr>
                    </thead>
                    <tbody>
                      {queueData.items.map((c) => {
                        const isSel =
                          activeCandidate?.job_id === c.job_id &&
                          activeCandidate?.track_id === c.track_id &&
                          activeCandidate?.track_generation_id === c.track_generation_id;
                        return (
                          <tr
                            key={`${c.job_id}-${c.track_id}-${c.track_generation_id}`}
                            className={isSel ? 'rv-row-active' : ''}
                            style={{ cursor: 'pointer' }}
                            onClick={() => selectCandidateForReview(c)}
                          >
                            <td className="rv-mono">
                              #{c.track_id}
                              <div style={{ fontSize: 10, color: 'var(--rv-text-secondary)' }}>
                                {c.original_filename || c.job_id.slice(0, 8)}
                              </div>
                            </td>
                            <td>
                              <PriorityBadge category={c.review_priority_category} />
                              <div className="rv-mono" style={{ fontSize: 10, marginTop: 2 }}>
                                {formatNullable(c.review_priority_index, 3)}
                              </div>
                            </td>
                            <td><QualityBadge category={c.quality_category} /></td>
                            <td><ReviewStatusBadge status={c.review_status} /></td>
                            <td className="rv-mono">{c.observation_count}</td>
                            <td className="rv-mono">{formatNullable(c.detector_confidence_median, 2)}</td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </div>
              </section>

              <section className="rv-card">
                {!activeCandidate ? (
                  <StatePanel
                    variant="empty"
                    title="Select a candidate track"
                    message="Click any row in the review queue to inspect automated evidence, quality explanations, and submit human review."
                  />
                ) : (
                  <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
                    <div className="rv-card-header" style={{ marginBottom: 0 }}>
                      <h3 className="rv-card-title">
                        Track #{activeCandidate.track_id} · {activeCandidate.original_filename || activeCandidate.job_id.slice(0, 8)}
                      </h3>
                      <div className="rv-header-actions">
                        <button
                          type="button"
                          className="rv-btn rv-btn-ghost rv-btn-sm"
                          onClick={() => onNavigate('workspace', activeCandidate.job_id)}
                        >
                          <ExternalLink size={13} /> Workspace
                        </button>
                        <button
                          type="button"
                          className="rv-btn rv-btn-ghost rv-btn-sm"
                          onClick={() => onNavigate('track', activeCandidate.job_id)}
                        >
                          Tracks
                        </button>
                        {activeCandidate.representative_snapshot_id && (
                          <button
                            type="button"
                            className="rv-btn rv-btn-ghost rv-btn-sm"
                            onClick={() => onNavigate('relative3d', activeCandidate.job_id)}
                          >
                            3D View
                          </button>
                        )}
                      </div>
                    </div>

                    {candidateCropUrl && (
                      <img
                        src={candidateCropUrl}
                        alt={`Track ${activeCandidate.track_id} crop`}
                        className="rv-crop-thumb"
                      />
                    )}

                    <div className="rv-kv-list">
                      <div className="rv-kv-item">
                        <span className="rv-kv-label">Review Priority</span>
                        <span className="rv-kv-value">
                          <PriorityBadge category={activeCandidate.review_priority_category} /> ({formatNullable(activeCandidate.review_priority_index, 3)})
                        </span>
                      </div>
                      <div className="rv-kv-item">
                        <span className="rv-kv-label">Fresh / Cached / Invalid</span>
                        <span className="rv-kv-value rv-mono">
                          {activeCandidate.fresh_valid_count} / {activeCandidate.cached_valid_count} / {activeCandidate.fresh_invalid_count}
                        </span>
                      </div>
                      <div className="rv-kv-item">
                        <span className="rv-kv-label">Relative Score (Uncal.)</span>
                        <span className="rv-kv-value rv-mono">
                          {formatNullable(activeCandidate.median_relative_score, 4)} ({activeCandidate.depth_provenance?.depth_model_id || 'none'})
                        </span>
                      </div>
                      <div className="rv-kv-item">
                        <span className="rv-kv-label">Evidence Sufficiency</span>
                        <span className="rv-kv-value">{activeCandidate.evidence_sufficiency}</span>
                      </div>
                    </div>

                    <div style={{ background: '#090f1d', padding: '8px 10px', borderRadius: 6, border: '1px solid var(--rv-border)' }}>
                      <strong style={{ fontSize: 12 }}>Priority Explanations:</strong>
                      <ul style={{ margin: '6px 0 0 18px', padding: 0, fontSize: 12, color: 'var(--rv-text-secondary)' }}>
                        {(activeCandidate.priority_reasons || []).map((r, idx) => (
                          <li key={idx}>{r}</li>
                        ))}
                      </ul>
                    </div>

                    {activeCandidate.quality_summary && (
                      <QualityBreakdownCard
                        assessment={{
                          quality_index: activeCandidate.median_fresh_quality_index,
                          quality_category: activeCandidate.quality_category,
                          evidence_coverage_ratio: activeCandidate.fresh_measurement_coverage,
                          components: activeCandidate.quality_summary.component_medians || {},
                          missing_evidence:
                            activeCandidate.quality_summary.evidence_sufficiency === 'INSUFFICIENT'
                              ? [activeCandidate.quality_summary.insufficient_evidence_reason || 'insufficient_fresh_history']
                              : [],
                          diagnostic_explanations: activeCandidate.quality_summary.dominant_rejection_reasons || [],
                        }}
                      />
                    )}

                    <form onSubmit={handleSubmitReview} style={{ borderTop: '1px solid var(--rv-border)', paddingTop: 12, display: 'flex', flexDirection: 'column', gap: 10 }}>
                      <strong style={{ fontSize: 13 }}>Human-in-the-Loop Review Decision</strong>
                      <div style={{ display: 'flex', gap: 10, flexWrap: 'wrap' }}>
                        <div className="rv-field" style={{ flex: 1, minWidth: 180 }}>
                          <label className="rv-field-label" htmlFor="rv-form-status">Review Outcome</label>
                          <select
                            id="rv-form-status"
                            className="rv-select"
                            value={formStatus}
                            onChange={(e) => setFormStatus(e.target.value)}
                          >
                            {REVIEW_STATUSES.map((st) => (
                              <option key={st} value={st}>{st}</option>
                            ))}
                          </select>
                        </div>
                        <div className="rv-field" style={{ flex: 1, minWidth: 160 }}>
                          <label className="rv-field-label" htmlFor="rv-form-label">Reviewer Label (Optional)</label>
                          <input
                            id="rv-form-label"
                            type="text"
                            className="rv-input"
                            placeholder="Local inspector note / tag"
                            maxLength={80}
                            value={formLabel}
                            onChange={(e) => setFormLabel(e.target.value)}
                          />
                        </div>
                      </div>

                      <div className="rv-field">
                        <label className="rv-field-label" htmlFor="rv-form-notes">Review Notes</label>
                        <textarea
                          id="rv-form-notes"
                          rows={2}
                          className="rv-input"
                          maxLength={1000}
                          placeholder="Confirm visual pothole appearance or false-positive cause (original automated evidence remains immutable)…"
                          value={formNotes}
                          onChange={(e) => setFormNotes(e.target.value)}
                        />
                      </div>

                      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 10, flexWrap: 'wrap' }}>
                        <button type="submit" className="rv-btn rv-btn-primary rv-btn-sm" disabled={savingReview}>
                          {activeCandidate.review_id ? 'Update Manual Review' : 'Save Manual Review'}
                        </button>
                        {reviewSaveMsg && (
                          <span style={{ fontSize: 12, color: reviewSaveMsg.startsWith('Error') ? '#f87171' : '#34d399' }}>
                            {reviewSaveMsg}
                          </span>
                        )}
                      </div>
                    </form>

                    {activeReviewDetail?.history?.length > 0 && (
                      <div style={{ fontSize: 11, color: 'var(--rv-text-secondary)', borderTop: '1px solid var(--rv-border)', paddingTop: 8 }}>
                        <strong>Audit History ({activeReviewDetail.history.length}):</strong>
                        {activeReviewDetail.history.map((h) => (
                          <div key={h.history_id} style={{ marginTop: 3 }}>
                            {formatDateTime(h.changed_at)}: {h.previous_status || 'NEW'} → <strong>{h.new_status}</strong>
                            {h.reviewer_label ? ` (${h.reviewer_label})` : ''}
                          </div>
                        ))}
                      </div>
                    )}
                  </div>
                )}
              </section>
            </div>
          )}
        </>
      )}

      {/* SUBTAB 3: ROAD SEGMENT ANALYTICS */}
      {subTab === 'segments' && (
        <>
          <InspectionSelector
            selectedJobId={isValidUuid(selectedJobId) ? selectedJobId : ''}
            onSelectJob={onSelectJob}
            onNavigate={onNavigate}
            label="Inspection for Road Segment Analytics"
          />

          {!selectedJobId ? (
            <StatePanel variant="empty" title="Select an inspection" message="Choose a completed inspection to analyze route segments." />
          ) : loadingSegments && !segmentData ? (
            <StatePanel variant="loading" title="Computing segment analytics" message="Grouping synchronized GPS tracks along route chainage…" />
          ) : segmentData && (
            <>
              <section className="rv-card">
                <div className="rv-card-header">
                  <h3 className="rv-card-title"><MapPin size={16} className="rv-icon-cyan" /> Segment Configuration &amp; Availability</h3>
                  <div className="rv-header-actions">
                    <label htmlFor="rv-seg-len" style={{ fontSize: 12, color: 'var(--rv-text-secondary)' }}>Segment Length:</label>
                    <select
                      id="rv-seg-len"
                      className="rv-select"
                      value={segmentLen}
                      onChange={(e) => setSegmentLen(e.target.value)}
                    >
                      <option value="25">25 m</option>
                      <option value="50">50 m</option>
                      <option value="100">100 m</option>
                      <option value="250">250 m</option>
                    </select>
                  </div>
                </div>

                {!segmentData.gps_available ? (
                  <StatePanel
                    variant="unavailable"
                    title="Geographical segment analysis unavailable"
                    message={segmentData.reason || 'No genuine GPS telemetry was provided for this inspection. Session-level track counts appear below without fabricated coordinates.'}
                    action={
                      <button
                        type="button"
                        className="rv-btn rv-btn-secondary rv-btn-sm"
                        onClick={() => onNavigate('geospatial', selectedJobId)}
                      >
                        Upload GPS Telemetry
                      </button>
                    }
                  />
                ) : (
                  <>
                    <div className="rv-kv-list" style={{ marginBottom: 14 }}>
                      <div className="rv-kv-item">
                        <span className="rv-kv-label">Observed Route Length</span>
                        <span className="rv-kv-value rv-mono">{formatNullable(segmentData.observed_route_length_m, 1, ' m')}</span>
                      </div>
                      <div className="rv-kv-item">
                        <span className="rv-kv-label">Geolocated Tracks</span>
                        <span className="rv-kv-value rv-mono">{segmentData.geolocated_track_count} ({segmentData.unlocated_track_count} unlocated)</span>
                      </div>
                      <div className="rv-kv-item">
                        <span className="rv-kv-label">GPS Provenance</span>
                        <span className="rv-kv-value">{segmentData.gps_metadata?.original_filename} ({segmentData.gps_metadata?.source_format?.toUpperCase()})</span>
                      </div>
                    </div>

                    <SegmentRouteSvg routePoints={routeData?.route_points} segments={segmentData.segments} />

                    <div className="rv-table-wrap">
                      <table className="rv-table">
                        <thead>
                          <tr>
                            <th>Segment</th>
                            <th>Chainage (m)</th>
                            <th>Session Tracks</th>
                            <th>Observations</th>
                            <th>Fresh Coverage</th>
                            <th>Max GPS Acc</th>
                            <th>Tracks</th>
                          </tr>
                        </thead>
                        <tbody>
                          {segmentData.segments.map((seg) => (
                            <tr key={seg.segment_id}>
                              <td className="rv-mono">{seg.segment_id}</td>
                              <td className="rv-mono">{seg.start_m} – {seg.end_m}</td>
                              <td className="rv-mono">{seg.session_candidate_track_count}</td>
                              <td className="rv-mono">{seg.candidate_observation_count}</td>
                              <td className="rv-mono">{formatRatePercent(seg.fresh_measurement_coverage)}</td>
                              <td className="rv-mono">{formatNullable(seg.max_horizontal_accuracy_m, 1, ' m')}</td>
                              <td className="rv-mono">{(seg.track_ids || []).join(', ') || '—'}</td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </>
                )}
              </section>

              <section className="rv-card">
                <div className="rv-card-header">
                  <h3 className="rv-card-title">Session-Level Candidate Summary</h3>
                  <span className="rv-card-meta">{segmentData.session_level_summary?.note}</span>
                </div>
                <div className="rv-kv-list">
                  <div className="rv-kv-item">
                    <span className="rv-kv-label">Session Candidate Tracks</span>
                    <span className="rv-kv-value rv-mono">{segmentData.session_level_summary?.session_track_count ?? 0}</span>
                  </div>
                  <div className="rv-kv-item">
                    <span className="rv-kv-label">Candidate Observations</span>
                    <span className="rv-kv-value rv-mono">{segmentData.session_level_summary?.total_candidate_observations ?? 0}</span>
                  </div>
                </div>
              </section>
            </>
          )}
        </>
      )}

      {/* SUBTAB 4: HISTORICAL COMPARISON */}
      {subTab === 'compare' && (
        <>
          <section className="rv-card">
            <div className="rv-card-header">
              <h3 className="rv-card-title">Select 2–8 Completed Inspections to Compare</h3>
              <button
                type="button"
                className="rv-btn rv-btn-primary rv-btn-sm"
                disabled={compareIds.length < 2 || loadingCompare}
                onClick={handleRunCompare}
              >
                <Search size={14} /> {loadingCompare ? 'Comparing…' : `Compare (${compareIds.length})`}
              </button>
            </div>
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8 }}>
              {completedJobs.map((j) => {
                const checked = compareIds.includes(j.job_id);
                return (
                  <label
                    key={j.job_id}
                    style={{
                      display: 'inline-flex',
                      alignItems: 'center',
                      gap: 6,
                      padding: '6px 10px',
                      borderRadius: 6,
                      border: `1px solid ${checked ? '#06b6d4' : 'var(--rv-border)'}`,
                      background: checked ? 'rgba(6, 182, 212, 0.12)' : '#090f1d',
                      fontSize: 12,
                      cursor: 'pointer',
                    }}
                  >
                    <input type="checkbox" checked={checked} onChange={() => toggleCompareJob(j.job_id)} />
                    <span>{j.original_filename || j.job_id.slice(0, 8)}</span>
                    <span className="rv-badge rv-badge-neutral">{(j.mode || '').toUpperCase()}</span>
                  </label>
                );
              })}
            </div>
          </section>

          {compareResult && (
            <>
              {(compareResult.compatibility_warnings || []).length > 0 && (
                <div
                  role="alert"
                  style={{
                    padding: '10px 14px',
                    borderRadius: 8,
                    border: '1px solid rgba(251, 191, 36, 0.45)',
                    background: 'rgba(120, 53, 15, 0.22)',
                    color: '#fde68a',
                    fontSize: 12,
                  }}
                >
                  <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 4 }}>
                    <AlertTriangle size={16} color="#fbbf24" />
                    <strong>Comparison Compatibility Warnings</strong>
                  </div>
                  <ul style={{ margin: '4px 0 0 20px', padding: 0 }}>
                    {compareResult.compatibility_warnings.map((w, i) => (
                      <li key={i}>{w}</li>
                    ))}
                  </ul>
                </div>
              )}

              <section className="rv-card">
                <div className="rv-card-header">
                  <h3 className="rv-card-title">Normalized Inspection Comparison</h3>
                  <span className="rv-card-meta">
                    Depth Compatible:{' '}
                    {compareResult.relative_depth_compatible ? (
                      <span style={{ color: '#34d399' }}><CheckCircle2 size={12} /> Yes</span>
                    ) : (
                      <span style={{ color: '#f87171' }}><XCircle size={12} /> No</span>
                    )}{' '}
                    · Shared GPS Clusters: {compareResult.shared_geospatial_cluster_count}
                  </span>
                </div>

                <div className="rv-table-wrap">
                  <table className="rv-table">
                    <thead>
                      <tr>
                        <th>Inspection</th>
                        <th>Mode / Depth</th>
                        <th>Frames</th>
                        <th>Tracks / 100f</th>
                        <th>Obs / 100f</th>
                        <th>Fresh Rate</th>
                        <th>Eff. Avail</th>
                        <th>FPS</th>
                        <th>Comp. Rel Score</th>
                      </tr>
                    </thead>
                    <tbody>
                      {compareResult.inspections.map((it) => (
                        <tr key={it.job_id}>
                          <td>{it.original_filename || it.job_id.slice(0, 8)}</td>
                          <td className="rv-mono" style={{ fontSize: 11 }}>
                            {(it.mode || '').toUpperCase()} · {it.depth_provenance?.depth_model_id} ({it.depth_provenance?.depth_input_size || '—'})
                          </td>
                          <td className="rv-mono">{it.processed_frames}</td>
                          <td className="rv-mono">{formatNullable(it.tracks_per_100_frames, 2)}</td>
                          <td className="rv-mono">{formatNullable(it.observations_per_100_frames, 2)}</td>
                          <td className="rv-mono">{formatRatePercent(it.fresh_valid_rate_over_observations)}</td>
                          <td className="rv-mono">{formatRatePercent(it.effective_measurement_availability)}</td>
                          <td className="rv-mono">{formatNullable(it.throughput_fps, 1)}</td>
                          <td className="rv-mono">{formatNullable(it.comparable_median_relative_score, 4)}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </section>
            </>
          )}
        </>
      )}

      <ScientificDisclaimer text={overview?.scientific_disclaimer || queueData?.scientific_disclaimer} />
    </div>
  );
}
