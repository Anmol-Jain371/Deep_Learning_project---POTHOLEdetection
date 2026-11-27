import React, { useEffect, useState, useCallback } from 'react';
import { listJobs } from '../api/client';
import { RefreshCw, FolderSearch } from 'lucide-react';
import { StatusBadge } from './StateBanner';

export default function InspectionSelector({
  selectedJobId,
  onSelectJob,
  filterStatus = null,
  onNavigate,
  label = 'Active Inspection',
}) {
  const [jobs, setJobs] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  const loadJobs = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const resp = await listJobs({ page: 1, page_size: 50, ...(filterStatus ? { status: filterStatus } : {}) });
      const items = resp?.jobs || [];
      setJobs(items);
      if (!selectedJobId && items.length > 0 && onSelectJob) {
        onSelectJob(items[0].job_id);
      }
    } catch (err) {
      setError(err);
    } finally {
      setLoading(false);
    }
  }, [filterStatus, selectedJobId, onSelectJob]);

  useEffect(() => {
    loadJobs();
  }, [loadJobs]);

  const currentJob = jobs.find((j) => j.job_id === selectedJobId);

  return (
    <div className="rv-selector-bar">
      <div className="rv-selector-left">
        <FolderSearch size={16} className="rv-icon-cyan" />
        <span className="rv-selector-label">{label}</span>
        <select
          className="rv-select rv-selector-dropdown"
          value={selectedJobId || ''}
          onChange={(e) => onSelectJob && onSelectJob(e.target.value)}
          disabled={loading || jobs.length === 0}
          aria-label={label}
        >
          {jobs.length === 0 ? (
            <option value="">{loading ? 'Loading inspections…' : 'No inspections available'}</option>
          ) : (
            <>
              {!currentJob && selectedJobId && (
                <option value={selectedJobId}>Selected: {selectedJobId.slice(0, 8)}…</option>
              )}
              {jobs.map((job) => (
                <option key={job.job_id} value={job.job_id}>
                  {job.original_filename || 'video'} — {job.mode?.toUpperCase()} — {job.status} ({job.job_id.slice(0, 8)})
                </option>
              ))}
            </>
          )}
        </select>
        {currentJob && <StatusBadge status={currentJob.status} />}
      </div>
      <div className="rv-selector-actions">
        <button
          type="button"
          className="rv-btn rv-btn-ghost rv-btn-sm"
          onClick={loadJobs}
          disabled={loading}
          title="Refresh inspection list"
        >
          <RefreshCw size={14} className={loading ? 'rv-spin' : ''} />
          <span>Refresh</span>
        </button>
        {onNavigate && (
          <button
            type="button"
            className="rv-btn rv-btn-secondary rv-btn-sm"
            onClick={() => onNavigate('new')}
          >
            New Inspection
          </button>
        )}
      </div>
      {error && (
        <span className="rv-selector-error">{error.message}</span>
      )}
    </div>
  );
}
