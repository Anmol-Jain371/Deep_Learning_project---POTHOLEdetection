import React, { useEffect, useState, useCallback } from 'react';
import { getHealth, getCapabilities, getModes, getApiBaseUrl } from '../api/client';
import { Cpu, RefreshCw, ShieldAlert, Server, HardDrive } from 'lucide-react';
import { StatePanel, StatusBadge } from '../components/StateBanner';
import { formatBytes, formatNullable } from '../utils/formatters';

export default function Settings() {
  const [health, setHealth] = useState(null);
  const [capabilities, setCapabilities] = useState(null);
  const [modes, setModes] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  const loadSettings = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const [h, c, m] = await Promise.all([
        getHealth(),
        getCapabilities(),
        getModes().catch(() => null),
      ]);
      setHealth(h);
      setCapabilities(c);
      setModes(m?.modes || null);
    } catch (err) {
      setError(err);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    loadSettings();
  }, [loadSettings]);

  if (loading && !health) {
    return (
      <div className="rv-page">
        <StatePanel variant="loading" title="Loading system configuration" message="Reading capabilities, hardware status, and safety limits…" />
      </div>
    );
  }

  if (error) {
    return (
      <div className="rv-page">
        <StatePanel
          variant={error.isApiDisabled ? 'api-disabled' : error.isNetworkError ? 'disconnected' : 'error'}
          title={error.isNetworkError ? 'Backend disconnected' : 'Failed to load system settings'}
          message={error.message}
          action={
            <button type="button" className="rv-btn rv-btn-secondary rv-btn-sm" onClick={loadSettings}>
              <RefreshCw size={14} /> Retry
            </button>
          }
        />
      </div>
    );
  }

  const limits = capabilities?.limits || {};
  const models = capabilities?.models || {};
  const queueCounts = health?.queue_counts || {};

  return (
    <div className="rv-page">
      <div className="rv-page-header">
        <div>
          <h2 className="rv-page-title">System Settings &amp; Capabilities</h2>
          <p className="rv-page-subtitle">
            Read-only backend hardware inventory, bounded worker limits, and configured V2 operating modes.
          </p>
        </div>
        <button type="button" className="rv-btn rv-btn-ghost rv-btn-sm" onClick={loadSettings} disabled={loading}>
          <RefreshCw size={14} className={loading ? 'rv-spin' : ''} /> Refresh
        </button>
      </div>

      <div className="rv-grid-2">
        <section className="rv-card">
          <div className="rv-card-header">
            <h3 className="rv-card-title"><Server size={16} className="rv-icon-cyan" /> Backend & Worker Status</h3>
            <StatusBadge status={health?.api_enabled ? 'COMPLETED' : 'CANCELLED'} />
          </div>
          <div className="rv-kv-list">
            <div className="rv-kv-item">
              <span className="rv-kv-label">Configured API Base</span>
              <span className="rv-kv-value">{getApiBaseUrl()}</span>
            </div>
            <div className="rv-kv-item">
              <span className="rv-kv-label">API Version</span>
              <span className="rv-kv-value">{formatNullable(health?.api_version)}</span>
            </div>
            <div className="rv-kv-item">
              <span className="rv-kv-label">Worker Thread</span>
              <span className="rv-kv-value">{health?.worker_running ? 'Running' : 'Stopped'}</span>
            </div>
            <div className="rv-kv-item">
              <span className="rv-kv-label">GPU Lock / Active Job</span>
              <span className="rv-kv-value">
                {health?.gpu_busy ? `Busy (${health.active_job_id?.slice(0, 8) || 'active'})` : 'Idle'}
              </span>
            </div>
            <div className="rv-kv-item">
              <span className="rv-kv-label">Queue Counts</span>
              <span className="rv-kv-value">
                Q:{queueCounts.QUEUED || 0} · R:{queueCounts.RUNNING || 0} · C:{queueCounts.COMPLETED || 0} · F:{queueCounts.FAILED || 0}
              </span>
            </div>
          </div>
        </section>

        <section className="rv-card">
          <div className="rv-card-header">
            <h3 className="rv-card-title"><Cpu size={16} className="rv-icon-cyan" /> Hardware & Model Readiness</h3>
          </div>
          <div className="rv-kv-list">
            <div className="rv-kv-item">
              <span className="rv-kv-label">CUDA GPU Available</span>
              <span className="rv-kv-value">{capabilities?.gpu_available ? 'Yes' : 'No (CPU)'}</span>
            </div>
            <div className="rv-kv-item">
              <span className="rv-kv-label">GPU Device Name</span>
              <span className="rv-kv-value">{capabilities?.gpu_name || 'Unavailable'}</span>
            </div>
            <div className="rv-kv-item">
              <span className="rv-kv-label">Detector Checkpoint</span>
              <span className="rv-kv-value">
                {models.detector || 'Unavailable'} ({capabilities?.detector_checkpoint_available ? 'Ready' : 'Missing'})
              </span>
            </div>
            <div className="rv-kv-item">
              <span className="rv-kv-label">Depth Models Supported</span>
              <span className="rv-kv-value">
                DAv2 Small: {models.depth_anything_v2_small ? 'Yes' : 'No'} · MiDaS Small: {models.midas_small ? 'Yes' : 'No'}
              </span>
            </div>
          </div>
        </section>
      </div>

      <section className="rv-card">
        <div className="rv-card-header">
          <h3 className="rv-card-title"><HardDrive size={16} className="rv-icon-cyan" /> Upload & Execution Safety Limits</h3>
        </div>
        <div className="rv-kv-list">
          <div className="rv-kv-item">
            <span className="rv-kv-label">Max Upload Size</span>
            <span className="rv-kv-value">{formatBytes(limits.maximum_upload_size_bytes)}</span>
          </div>
          <div className="rv-kv-item">
            <span className="rv-kv-label">Allowed Video Formats</span>
            <span className="rv-kv-value">{(limits.allowed_video_formats || []).join(', ') || 'Unavailable'}</span>
          </div>
          <div className="rv-kv-item">
            <span className="rv-kv-label">Max Video Resolution</span>
            <span className="rv-kv-value">
              {limits.maximum_video_resolution ? `${limits.maximum_video_resolution[0]} × ${limits.maximum_video_resolution[1]}` : 'Unavailable'}
            </span>
          </div>
          <div className="rv-kv-item">
            <span className="rv-kv-label">Max Video Duration</span>
            <span className="rv-kv-value">{formatNullable(limits.maximum_video_duration_sec, 0, ' s')}</span>
          </div>
          <div className="rv-kv-item">
            <span className="rv-kv-label">Max Processed Frames</span>
            <span className="rv-kv-value">{formatNullable(limits.maximum_processed_frames)}</span>
          </div>
          <div className="rv-kv-item">
            <span className="rv-kv-label">Queue & GPU Concurrency</span>
            <span className="rv-kv-value">
              Max queued: {formatNullable(limits.max_queued_jobs)} · Max GPU jobs: {formatNullable(limits.max_gpu_concurrency)}
            </span>
          </div>
          <div className="rv-kv-item">
            <span className="rv-kv-label">Pagination Bounds</span>
            <span className="rv-kv-value">
              Default: {formatNullable(limits.default_page_size)} · Max: {formatNullable(limits.max_page_size)}
            </span>
          </div>
        </div>
      </section>

      {modes && (
        <section className="rv-card">
          <div className="rv-card-header">
            <h3 className="rv-card-title">Configured Pipeline Modes</h3>
          </div>
          <div className="rv-table-wrap">
            <table className="rv-table">
              <thead>
                <tr>
                  <th>Mode</th>
                  <th>Depth Enabled</th>
                  <th>Depth Model & Input</th>
                  <th>Cavity Strategy</th>
                  <th>Scheduler</th>
                  <th>Description</th>
                </tr>
              </thead>
              <tbody>
                {Object.entries(modes).map(([key, m]) => (
                  <tr key={key}>
                    <td className="rv-mono" style={{ fontWeight: 700 }}>{key.toUpperCase()}</td>
                    <td>{m.depth?.enabled ? 'Yes' : 'No'}</td>
                    <td className="rv-mono">{m.depth?.model_id} ({m.depth?.input_size}px, {m.depth?.precision})</td>
                    <td className="rv-mono">{m.cavity?.strategy}</td>
                    <td className="rv-mono">{m.scheduler?.strategy}</td>
                    <td style={{ color: 'var(--rv-text-secondary)' }}>{m.description}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>
      )}

      {capabilities?.security_note && (
        <div className="rv-state-panel rv-state-api-disabled" role="note">
          <ShieldAlert size={18} className="rv-icon-warning" />
          <div className="rv-state-body">
            <h4 className="rv-state-title">Local Prototype Security Boundary</h4>
            <p className="rv-state-message">{capabilities.security_note}</p>
          </div>
        </div>
      )}
    </div>
  );
}
