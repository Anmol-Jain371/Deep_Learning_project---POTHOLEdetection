import React, { useEffect, useState, useCallback } from 'react';
import { listJobs } from '../api/client';
import { RefreshCw, ChevronLeft, ChevronRight, PlusCircle } from 'lucide-react';
import { StatePanel, StatusBadge } from '../components/StateBanner';
import { formatDateTime, formatNullable } from '../utils/formatters';

const JOB_STATUSES = ['QUEUED', 'INITIALIZING', 'RUNNING', 'CANCELLING', 'COMPLETED', 'FAILED', 'CANCELLED'];

export default function History({ onNavigate, selectedJobId }) {
  const [jobs, setJobs] = useState([]);
  const [total, setTotal] = useState(0);
  const [page, setPage] = useState(1);
  const [pageSize] = useState(15);
  const [statusFilter, setStatusFilter] = useState('');
  const [modeFilter, setModeFilter] = useState('');
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  const loadHistory = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const resp = await listJobs({
        page,
        page_size: pageSize,
        ...(statusFilter ? { status: statusFilter } : {}),
      });
      setJobs(resp?.jobs || []);
      setTotal(resp?.total ?? 0);
    } catch (err) {
      setError(err);
    } finally {
      setLoading(false);
    }
  }, [page, pageSize, statusFilter]);

  useEffect(() => {
    loadHistory();
  }, [loadHistory]);

  const visibleJobs = modeFilter
    ? jobs.filter((j) => (j.mode || '').toLowerCase() === modeFilter.toLowerCase())
    : jobs;

  const totalPages = Math.max(1, Math.ceil(total / pageSize));

  return (
    <div className="rv-page">
      <div className="rv-page-header">
        <div>
          <h2 className="rv-page-title">Inspection History</h2>
          <p className="rv-page-subtitle">
            Select any inspection row to open its Workspace, observations, tracks, and artifacts.
          </p>
        </div>
        <div className="rv-header-actions">
          <button type="button" className="rv-btn rv-btn-ghost rv-btn-sm" onClick={loadHistory} disabled={loading}>
            <RefreshCw size={14} className={loading ? 'rv-spin' : ''} /> Refresh
          </button>
          <button type="button" className="rv-btn rv-btn-primary rv-btn-sm" onClick={() => onNavigate('new')}>
            <PlusCircle size={14} /> New Inspection
          </button>
        </div>
      </div>

      <section className="rv-card">
        <div className="rv-filter-bar">
          <div className="rv-field">
            <label className="rv-field-label" htmlFor="rv-hist-status">Status</label>
            <select
              id="rv-hist-status"
              className="rv-select"
              value={statusFilter}
              onChange={(e) => {
                setStatusFilter(e.target.value);
                setPage(1);
              }}
            >
              <option value="">All Statuses</option>
              {JOB_STATUSES.map((st) => (
                <option key={st} value={st}>{st}</option>
              ))}
            </select>
          </div>

          <div className="rv-field">
            <label className="rv-field-label" htmlFor="rv-hist-mode">Mode (current page)</label>
            <select
              id="rv-hist-mode"
              className="rv-select"
              value={modeFilter}
              onChange={(e) => setModeFilter(e.target.value)}
            >
              <option value="">All Modes</option>
              <option value="live">LIVE</option>
              <option value="balanced">BALANCED</option>
              <option value="survey">SURVEY</option>
            </select>
          </div>
        </div>

        {loading && jobs.length === 0 ? (
          <StatePanel variant="loading" title="Loading inspection history" message="Querying SQLite job queue…" />
        ) : error ? (
          <StatePanel
            variant={error.isApiDisabled ? 'api-disabled' : error.isNetworkError ? 'disconnected' : 'error'}
            title={error.isNetworkError ? 'Backend disconnected' : 'Unable to load inspection history'}
            message={error.message}
          />
        ) : visibleJobs.length === 0 ? (
          <StatePanel
            variant="empty"
            title="No matching inspections"
            message="Adjust filters or submit a new video inspection."
          />
        ) : (
          <>
            <div className="rv-table-wrap">
              <table className="rv-table">
                <thead>
                  <tr>
                    <th>Filename</th>
                    <th>Mode</th>
                    <th>Status</th>
                    <th>Frames</th>
                    <th>FPS</th>
                    <th>Observations</th>
                    <th>Tracks</th>
                    <th>Created</th>
                    <th>Actions</th>
                  </tr>
                </thead>
                <tbody>
                  {visibleJobs.map((job) => (
                    <tr
                      key={job.job_id}
                      className={`rv-row-clickable ${selectedJobId === job.job_id ? 'rv-row-selected' : ''}`}
                      onClick={() => onNavigate('workspace', job.job_id)}
                    >
                      <td>
                        <div style={{ fontWeight: 600 }}>{job.original_filename || 'video.mp4'}</div>
                        <div className="rv-mono" style={{ fontSize: 11, color: 'var(--rv-text-muted)' }}>
                          {job.job_id}
                        </div>
                      </td>
                      <td><span className="rv-badge rv-badge-neutral">{(job.mode || '').toUpperCase()}</span></td>
                      <td><StatusBadge status={job.status} /></td>
                      <td className="rv-mono">
                        {job.frames_processed ?? 0} / {formatNullable(job.total_frames)}
                      </td>
                      <td className="rv-mono">{formatNullable(job.throughput_fps, 1, ' fps')}</td>
                      <td className="rv-mono">{job.candidate_observation_count ?? 0}</td>
                      <td className="rv-mono">{job.unique_candidate_track_count ?? 0}</td>
                      <td>{formatDateTime(job.created_at)}</td>
                      <td>
                        <div className="rv-header-actions" style={{ gap: 6 }}>
                          <button
                            type="button"
                            className="rv-btn rv-btn-secondary rv-btn-sm"
                            onClick={(e) => {
                              e.stopPropagation();
                              onNavigate('workspace', job.job_id);
                            }}
                          >
                            Workspace
                          </button>
                          {job.status === 'COMPLETED' && (
                            <button
                              type="button"
                              className="rv-btn rv-btn-ghost rv-btn-sm"
                              onClick={(e) => {
                                e.stopPropagation();
                                onNavigate('reports', job.job_id);
                              }}
                            >
                              Report
                            </button>
                          )}
                        </div>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>

            <div className="rv-pagination">
              <span>
                Showing {visibleJobs.length} of {total} inspections (Page {page} of {totalPages})
              </span>
              <div className="rv-pagination-btns">
                <button
                  type="button"
                  className="rv-btn rv-btn-ghost rv-btn-sm"
                  disabled={page <= 1 || loading}
                  onClick={() => setPage((p) => Math.max(1, p - 1))}
                >
                  <ChevronLeft size={14} /> Prev
                </button>
                <button
                  type="button"
                  className="rv-btn rv-btn-ghost rv-btn-sm"
                  disabled={page >= totalPages || loading}
                  onClick={() => setPage((p) => p + 1)}
                >
                  Next <ChevronRight size={14} />
                </button>
              </div>
            </div>
          </>
        )}
      </section>
    </div>
  );
}
