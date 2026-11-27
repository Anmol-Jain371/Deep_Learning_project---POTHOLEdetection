import React, { useEffect, useState, useCallback } from 'react';
import { getHealth, listJobs, getCapabilities } from '../api/client';
import {
  Layers,
  CheckCircle2,
  Clock,
  Crosshair,
  GitCommitHorizontal,
  RefreshCw,
  PlusCircle,
  ArrowRight,
} from 'lucide-react';
import {
  StatePanel,
  StatusBadge,
  ProvenanceBadge,
  ScientificDisclaimer,
  MeasurementStateDistribution,
} from '../components/StateBanner';
import { formatDateTime, formatNullable } from '../utils/formatters';

export default function Overview({ onNavigate }) {
  const [health, setHealth] = useState(null);
  const [jobs, setJobs] = useState([]);
  const [totalJobsCount, setTotalJobsCount] = useState(0);
  const [provenanceScope, setProvenanceScope] = useState('genuine');
  const [capabilities, setCapabilities] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  const loadData = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const [h, j, cap] = await Promise.all([
        getHealth(),
        listJobs({ page: 1, page_size: 20, scope: provenanceScope }),
        getCapabilities().catch(() => null),
      ]);
      setHealth(h);
      setJobs(j?.jobs || []);
      setTotalJobsCount(j?.total ?? (j?.jobs?.length || 0));
      setCapabilities(cap);
    } catch (err) {
      setError(err);
    } finally {
      setLoading(false);
    }
  }, [provenanceScope]);

  useEffect(() => {
    loadData();
  }, [loadData]);

  if (loading && !health) {
    return (
      <div className="rv-page">
        <StatePanel variant="loading" title="Loading system overview" message="Querying RoadVision AI 2.0 backend health and recent inspections…" />
      </div>
    );
  }

  if (error?.isApiDisabled || (health && !health.api_enabled)) {
    return (
      <div className="rv-page">
        <StatePanel
          variant="api-disabled"
          title="RoadVision V2 API is disabled"
          message={error?.message || 'Start server.py with ROADVISION_V2_ENABLED=true or enable it in config/roadvision_api_v2.yaml.'}
          action={
            <button type="button" className="rv-btn rv-btn-secondary rv-btn-sm" onClick={loadData}>
              <RefreshCw size={14} /> Retry
            </button>
          }
        />
      </div>
    );
  }

  if (error?.isNetworkError || error) {
    return (
      <div className="rv-page">
        <StatePanel
          variant={error?.isNetworkError ? 'disconnected' : 'error'}
          title={error?.isNetworkError ? 'Backend server disconnected' : 'Failed to load overview'}
          message={error.message}
          action={
            <button type="button" className="rv-btn rv-btn-secondary rv-btn-sm" onClick={loadData}>
              <RefreshCw size={14} /> Retry Connection
            </button>
          }
        />
      </div>
    );
  }

  const queueCounts =
    provenanceScope === 'genuine'
      ? (health?.queue_counts || {})
      : jobs.reduce((acc, j) => {
          acc[j.status] = (acc[j.status] || 0) + 1;
          return acc;
        }, {});
  const provCounts = health?.provenance_counts || {};
  const excludedNonGenuine = health?.non_genuine_excluded_count ?? 0;
  const completedCount = queueCounts.COMPLETED || 0;
  const failedCount = queueCounts.FAILED || 0;
  const cancelledCount = queueCounts.CANCELLED || 0;
  const activeCount =
    (queueCounts.QUEUED || 0) +
    (queueCounts.INITIALIZING || 0) +
    (queueCounts.RUNNING || 0) +
    (queueCounts.CANCELLING || 0);

  const recentCompleted = jobs.filter((j) => j.status === 'COMPLETED');
  const totalRecentObservations = recentCompleted.reduce(
    (sum, j) => sum + (Number(j.candidate_observation_count) || 0),
    0
  );
  const totalRecentTracks = recentCompleted.reduce(
    (sum, j) => sum + (Number(j.unique_candidate_track_count) || 0),
    0
  );

  const aggStateCounts = {};
  for (const job of recentCompleted) {
    const counts = job.measurement_state_counts || {};
    for (const [k, v] of Object.entries(counts)) {
      aggStateCounts[k] = (aggStateCounts[k] || 0) + (Number(v) || 0);
    }
  }

  return (
    <div className="rv-page">
      <div className="rv-page-header">
        <div>
          <h2 className="rv-page-title">Inspection Command Overview</h2>
          <p className="rv-page-subtitle">
            Bounded GPU worker status, provenance-isolated aggregates, and recent relative-depth audits.
          </p>
        </div>
        <div className="rv-header-actions">
          <select
            aria-label="Provenance aggregate filter"
            className="rv-select"
            value={provenanceScope}
            onChange={(e) => setProvenanceScope(e.target.value)}
          >
            <option value="genuine">Genuine Inspections Only ({provCounts.USER_INSPECTION ?? 0})</option>
            <option value="exploratory">Include Exploratory Research ({(provCounts.USER_INSPECTION ?? 0) + (provCounts.RESEARCH_EXPLORATORY ?? 0)})</option>
            <option value="all">All Stored Records (incl. {excludedNonGenuine} Non-Genuine)</option>
          </select>
          <button type="button" className="rv-btn rv-btn-ghost rv-btn-sm" onClick={loadData} disabled={loading}>
            <RefreshCw size={14} className={loading ? 'rv-spin' : ''} /> Refresh
          </button>
          <button type="button" className="rv-btn rv-btn-primary rv-btn-sm" onClick={() => onNavigate('new')}>
            <PlusCircle size={14} /> New Inspection
          </button>
        </div>
      </div>

      {provenanceScope === 'genuine' && excludedNonGenuine > 0 && (
        <div className="rv-disclaimer" role="note" style={{ marginBottom: 14 }}>
          <span>
            Default genuine aggregate excludes <strong>{excludedNonGenuine}</strong> non-genuine/exploratory record(s):{' '}
            Exploratory: {provCounts.RESEARCH_EXPLORATORY ?? 0} · Test Fixtures: {provCounts.TEST_FIXTURE ?? 0} ·{' '}
            Synthetic: {provCounts.SYNTHETIC_EXPERIMENT ?? 0} · Unknown: {provCounts.UNKNOWN ?? 0}.
          </span>
        </div>
      )}

      <div className="rv-kpi-grid">
        <div className="rv-kpi-card">
          <div className="rv-kpi-top">
            <span>Inspections in Scope</span>
            <Layers size={16} className="rv-icon-cyan" />
          </div>
          <div className="rv-kpi-val">{totalJobsCount}</div>
          <div className="rv-kpi-sub">
            {provenanceScope === 'genuine' ? 'USER_INSPECTION only' : `Scope: ${provenanceScope.toUpperCase()}`}
          </div>
        </div>

        <div className="rv-kpi-card">
          <div className="rv-kpi-top">
            <span>Completed</span>
            <CheckCircle2 size={16} style={{ color: '#34d399' }} />
          </div>
          <div className="rv-kpi-val">{completedCount}</div>
          <div className="rv-kpi-sub">Failed: {failedCount} · Cancelled: {cancelledCount}</div>
        </div>

        <div className="rv-kpi-card">
          <div className="rv-kpi-top">
            <span>Queue / Active</span>
            <Clock size={16} style={{ color: '#fbbf24' }} />
          </div>
          <div className="rv-kpi-val">{activeCount}</div>
          <div className="rv-kpi-sub">
            Worker: {health?.worker_running ? 'Online' : 'Offline'} ({(health?.device || 'cpu').toUpperCase()})
          </div>
        </div>

        <div className="rv-kpi-card">
          <div className="rv-kpi-top">
            <span>Recent Observations</span>
            <Crosshair size={16} className="rv-icon-cyan" />
          </div>
          <div className="rv-kpi-val">{totalRecentObservations}</div>
          <div className="rv-kpi-sub">Across {recentCompleted.length} recent completed jobs</div>
        </div>

        <div className="rv-kpi-card">
          <div className="rv-kpi-top">
            <span>Session Tracks</span>
            <GitCommitHorizontal size={16} className="rv-icon-cyan" />
          </div>
          <div className="rv-kpi-val">{totalRecentTracks}</div>
          <div className="rv-kpi-sub">Unique ByteTrack IDs (not verified pothole count)</div>
        </div>
      </div>

      <div className="rv-grid-2">
        <section className="rv-card">
          <div className="rv-card-header">
            <h3 className="rv-card-title">Measurement Coverage (Recent Completed Jobs)</h3>
            <span className="rv-card-meta">{totalRecentObservations} total observations</span>
          </div>
          {recentCompleted.length === 0 ? (
            <StatePanel
              variant="empty"
              title="No completed inspections yet"
              message="Submit a road survey video to view fresh valid and cached measurement distributions."
            />
          ) : (
            <MeasurementStateDistribution
              countsByState={aggStateCounts}
              totalObservations={totalRecentObservations}
            />
          )}
        </section>

        <section className="rv-card">
          <div className="rv-card-header">
            <h3 className="rv-card-title">Runtime Environment</h3>
            <span className="rv-card-meta">API v{health?.api_version || '2.0'}</span>
          </div>
          <div className="rv-kv-list">
            <div className="rv-kv-item">
              <span className="rv-kv-label">Compute Device</span>
              <span className="rv-kv-value">{capabilities?.gpu_name || (health?.device || 'cpu').toUpperCase()}</span>
            </div>
            <div className="rv-kv-item">
              <span className="rv-kv-label">GPU Worker State</span>
              <span className="rv-kv-value">{health?.gpu_busy ? 'Busy (1 job active)' : 'Idle / Ready'}</span>
            </div>
            <div className="rv-kv-item">
              <span className="rv-kv-label">Detector Model</span>
              <span className="rv-kv-value">{capabilities?.models?.detector || 'Unavailable'}</span>
            </div>
            <div className="rv-kv-item">
              <span className="rv-kv-label">Max Queue / GPU</span>
              <span className="rv-kv-value">
                {capabilities?.limits
                  ? `${capabilities.limits.max_queued_jobs} queued / ${capabilities.limits.max_gpu_concurrency} GPU`
                  : 'Unavailable'}
              </span>
            </div>
          </div>
        </section>
      </div>

      <section className="rv-card">
        <div className="rv-card-header">
          <h3 className="rv-card-title">Recent Inspections</h3>
          <button type="button" className="rv-btn rv-btn-ghost rv-btn-sm" onClick={() => onNavigate('history')}>
            <span>View Full History</span>
            <ArrowRight size={14} />
          </button>
        </div>

        {jobs.length === 0 ? (
          <StatePanel
            variant="empty"
            title="No inspections recorded"
            message="Upload a dashcam or road survey video in New Inspection to start processing."
            action={
              <button type="button" className="rv-btn rv-btn-primary rv-btn-sm" onClick={() => onNavigate('new')}>
                <PlusCircle size={14} /> Start Inspection
              </button>
            }
          />
        ) : (
          <div className="rv-table-wrap">
            <table className="rv-table">
              <thead>
                <tr>
                  <th>Inspection File</th>
                  <th>Provenance</th>
                  <th>Mode</th>
                  <th>Status</th>
                  <th>Frames</th>
                  <th>Observations</th>
                  <th>Session Tracks</th>
                  <th>Created</th>
                  <th>Action</th>
                </tr>
              </thead>
              <tbody>
                {jobs.slice(0, 8).map((job) => (
                  <tr
                    key={job.job_id}
                    className="rv-row-clickable"
                    onClick={() => onNavigate('workspace', job.job_id)}
                  >
                    <td>
                      <div style={{ fontWeight: 600 }}>{job.original_filename || 'video.mp4'}</div>
                      <div className="rv-mono" style={{ color: 'var(--rv-text-muted)', fontSize: 11 }}>
                        {job.job_id}
                      </div>
                    </td>
                    <td>
                      <ProvenanceBadge provenance={job.provenance} />
                    </td>
                    <td>
                      <span className="rv-badge rv-badge-neutral">{(job.mode || '').toUpperCase()}</span>
                    </td>
                    <td>
                      <StatusBadge status={job.status} />
                    </td>
                    <td className="rv-mono">
                      {job.frames_processed ?? 0} / {formatNullable(job.total_frames)}
                    </td>
                    <td className="rv-mono">{job.candidate_observation_count ?? 0}</td>
                    <td className="rv-mono">{job.unique_candidate_track_count ?? 0}</td>
                    <td>{formatDateTime(job.created_at)}</td>
                    <td>
                      <button
                        type="button"
                        className="rv-btn rv-btn-secondary rv-btn-sm"
                        onClick={(e) => {
                          e.stopPropagation();
                          onNavigate('workspace', job.job_id);
                        }}
                      >
                        Open
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </section>

      <ScientificDisclaimer text={health?.scientific_disclaimer} />
    </div>
  );
}
