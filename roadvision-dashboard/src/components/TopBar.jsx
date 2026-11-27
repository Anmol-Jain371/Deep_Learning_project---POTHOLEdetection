import React, { useEffect, useState } from 'react';
import { getHealth, getJob } from '../api/client';
import { Cpu, Wifi, WifiOff, ShieldAlert, PlusCircle, MonitorPlay } from 'lucide-react';
import { StatusBadge } from './StateBanner';

const PAGE_TITLES = {
  overview: 'System Overview',
  new: 'New Video Inspection',
  history: 'Inspection History',
  workspace: 'Inspection Workspace',
  hazard: 'Hazard Explorer',
  track: 'Track Analysis',
  relative3d: 'Relative 3D Analysis',
  geospatial: 'Geospatial Intelligence',
  maintenance: 'Road Maintenance Intelligence',
  performance: 'Pipeline Performance',
  reports: 'Inspection Reports',
  settings: 'System Settings & Limits',
  legacy: 'Legacy V1 Suite',
};

export default function TopBar({ activeSection, selectedJobId, onNavigate, onHealthChange }) {
  const [health, setHealth] = useState(null);
  const [connError, setConnError] = useState(null);
  const [activeJob, setActiveJob] = useState(null);

  useEffect(() => {
    let mounted = true;

    async function pollHealth() {
      try {
        const h = await getHealth();
        if (!mounted) return;
        setHealth(h);
        setConnError(null);
        if (onHealthChange) onHealthChange({ connected: true, health: h, error: null });
      } catch (err) {
        if (!mounted) return;
        setHealth(null);
        setConnError(err);
        if (onHealthChange) onHealthChange({ connected: false, health: null, error: err });
      }
    }

    pollHealth();
    const timer = setInterval(pollHealth, 8000);
    return () => {
      mounted = false;
      clearInterval(timer);
    };
  }, [onHealthChange]);

  useEffect(() => {
    let mounted = true;
    if (!selectedJobId) {
      setActiveJob(null);
      return undefined;
    }
    getJob(selectedJobId)
      .then((job) => {
        if (mounted) setActiveJob(job);
      })
      .catch(() => {
        if (mounted) setActiveJob(null);
      });
    return () => {
      mounted = false;
    };
  }, [selectedJobId, activeSection]);

  const renderConnectionPill = () => {
    if (connError?.isApiDisabled) {
      return (
        <span className="rv-conn-pill rv-conn-warning" title={connError.message}>
          <ShieldAlert size={14} />
          <span>V2 API Disabled</span>
        </span>
      );
    }
    if (connError || !health) {
      return (
        <span className="rv-conn-pill rv-conn-offline" title={connError?.message || 'Backend disconnected'}>
          <WifiOff size={14} />
          <span>Backend Disconnected</span>
        </span>
      );
    }
    if (!health.api_enabled) {
      return (
        <span className="rv-conn-pill rv-conn-warning">
          <ShieldAlert size={14} />
          <span>V2 API Disabled</span>
        </span>
      );
    }
    return (
      <span
        className="rv-conn-pill rv-conn-online"
        title={`Device: ${health.device?.toUpperCase()} | Worker: ${health.worker_running ? 'Running' : 'Stopped'}`}
      >
        <Wifi size={14} />
        <span>Connected</span>
        <span className="rv-conn-divider">·</span>
        <Cpu size={13} />
        <span>{(health.device || 'cpu').toUpperCase()}</span>
        {health.active_job_id && <span className="rv-worker-busy">Worker Busy</span>}
      </span>
    );
  };

  return (
    <header className="rv-topbar">
      <div className="rv-topbar-left">
        <span className="rv-topbar-brand">RoadVision AI 2.0</span>
        <span className="rv-topbar-sep">/</span>
        <h1 className="rv-topbar-title">{PAGE_TITLES[activeSection] || 'Dashboard'}</h1>
      </div>

      <div className="rv-topbar-right">
        {activeJob && (
          <button
            type="button"
            className="rv-active-job-pill"
            onClick={() => onNavigate('workspace', activeJob.job_id)}
            title={`Open Workspace for ${activeJob.original_filename}`}
          >
            <MonitorPlay size={14} className="rv-icon-cyan" />
            <span className="rv-active-job-name">{activeJob.original_filename || activeJob.job_id.slice(0, 8)}</span>
            <StatusBadge status={activeJob.status} />
          </button>
        )}

        {renderConnectionPill()}

        <button
          type="button"
          className="rv-btn rv-btn-primary rv-btn-sm"
          onClick={() => onNavigate('new')}
        >
          <PlusCircle size={14} />
          <span>New Inspection</span>
        </button>
      </div>
    </header>
  );
}
