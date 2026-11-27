import React, { useState, useEffect, useCallback } from 'react';
import Sidebar from './components/Sidebar';
import TopBar from './components/TopBar';
import Hero from './components/Hero';
import KPICards from './components/KPICards';
import AuditTable from './components/AuditTable';
import PipelineSection from './components/PipelineSection';
import BenchmarkComparison from './components/BenchmarkComparison';
import VideoUploadAnalysis from './components/VideoUploadAnalysis';
import Footer from './components/Footer';
import Overview from './pages/Overview';
import NewInspection from './pages/NewInspection';
import History from './pages/History';
import Workspace from './pages/Workspace';
import HazardExplorer from './pages/HazardExplorer';
import TrackAnalysis from './pages/TrackAnalysis';
import Performance from './pages/Performance';
import Reports from './pages/Reports';
import Settings from './pages/Settings';
import Relative3DAnalysis from './pages/Relative3DAnalysis';
import GeospatialIntelligence from './pages/GeospatialIntelligence';
import MaintenanceIntelligence from './pages/MaintenanceIntelligence';
import { isValidUuid } from './utils/formatters';
import './App.css';
import './styles/dashboard.css';

const VALID_SECTIONS = new Set([
  'overview',
  'new',
  'history',
  'workspace',
  'hazard',
  'track',
  'relative3d',
  'geospatial',
  'maintenance',
  'performance',
  'reports',
  'settings',
  'legacy',
]);

const JOB_SCOPED_SECTIONS = new Set([
  'workspace',
  'hazard',
  'track',
  'relative3d',
  'geospatial',
  'maintenance',
  'performance',
  'reports',
]);

const SELECTED_JOB_STORAGE_KEY = 'rv2_selected_job_id';

function parseHashRoute() {
  const raw = (window.location.hash || '').replace(/^#\/?/, '');
  if (!raw) {
    return { section: 'overview', jobId: null };
  }
  const parts = raw.split('/').filter(Boolean);
  const candidateSection = parts[0]?.toLowerCase() || 'overview';
  const section = VALID_SECTIONS.has(candidateSection) ? candidateSection : 'overview';
  const hashJobId = parts[1] ? decodeURIComponent(parts[1]) : null;
  return { section, jobId: hashJobId };
}

function buildHash(section, jobId) {
  if (JOB_SCOPED_SECTIONS.has(section) && jobId) {
    return `#/${section}/${encodeURIComponent(jobId)}`;
  }
  return `#/${section}`;
}

function LegacySuite({ onOpenV2 }) {
  const [legacyData, setLegacyData] = useState(null);
  return (
    <div className="editorial-page-canvas">
      <Hero onExplore={onOpenV2} />
      <VideoUploadAnalysis onAnalysisComplete={(data) => setLegacyData(data)} />
      {legacyData && (
        <div className="dashboard-container" style={{ maxWidth: 1280, margin: '0 auto', padding: '0 24px 48px' }}>
          <KPICards stats={legacyData.stats} />
          <AuditTable records={legacyData.auditRecords} hasGPS={legacyData.hasGPS} />
        </div>
      )}
      <BenchmarkComparison />
      <PipelineSection />
    </div>
  );
}

export default function App() {
  const initialRoute = parseHashRoute();
  const storedJob = window.localStorage.getItem(SELECTED_JOB_STORAGE_KEY);
  const initialJobId = initialRoute.jobId || (isValidUuid(storedJob) ? storedJob : null);

  const [activeSection, setActiveSection] = useState(initialRoute.section);
  const [selectedJobId, setSelectedJobId] = useState(initialJobId);
  const [newInspectionPreset, setNewInspectionPreset] = useState(null);

  const persistJobId = useCallback((jobId) => {
    setSelectedJobId(jobId || null);
    if (jobId && isValidUuid(jobId)) {
      window.localStorage.setItem(SELECTED_JOB_STORAGE_KEY, jobId);
    }
  }, []);

  const navigate = useCallback((section, nextJobId = undefined, options = null) => {
    const targetSection = VALID_SECTIONS.has(section) ? section : 'overview';
    const resolvedJobId = nextJobId !== undefined ? nextJobId : selectedJobId;
    if (nextJobId !== undefined) {
      persistJobId(nextJobId);
    }
    setNewInspectionPreset(targetSection === 'new' ? options : null);
    setActiveSection(targetSection);
    const nextHash = buildHash(targetSection, resolvedJobId);
    if (window.location.hash !== nextHash) {
      window.location.hash = nextHash;
    }
  }, [selectedJobId, persistJobId]);

  const handleSelectJob = useCallback((jobId) => {
    persistJobId(jobId);
    if (JOB_SCOPED_SECTIONS.has(activeSection)) {
      const nextHash = buildHash(activeSection, jobId);
      if (window.location.hash !== nextHash) {
        window.location.hash = nextHash;
      }
    }
  }, [activeSection, persistJobId]);

  useEffect(() => {
    if (!window.location.hash) {
      window.location.hash = buildHash(initialRoute.section, initialJobId);
    }
    const onHashChange = () => {
      const parsed = parseHashRoute();
      setActiveSection(parsed.section);
      if (parsed.jobId) {
        persistJobId(parsed.jobId);
      }
    };
    window.addEventListener('hashchange', onHashChange);
    return () => window.removeEventListener('hashchange', onHashChange);
  }, [initialRoute.section, initialJobId, persistJobId]);

  const renderSection = () => {
    switch (activeSection) {
      case 'overview':
        return <Overview onNavigate={navigate} />;
      case 'new':
        return <NewInspection onNavigate={navigate} initialPreset={newInspectionPreset} />;
      case 'history':
        return <History onNavigate={navigate} selectedJobId={selectedJobId} />;
      case 'workspace':
        return (
          <Workspace
            selectedJobId={selectedJobId}
            onSelectJob={handleSelectJob}
            onNavigate={navigate}
          />
        );
      case 'hazard':
        return (
          <HazardExplorer
            selectedJobId={selectedJobId}
            onSelectJob={handleSelectJob}
            onNavigate={navigate}
          />
        );
      case 'track':
        return (
          <TrackAnalysis
            selectedJobId={selectedJobId}
            onSelectJob={handleSelectJob}
            onNavigate={navigate}
          />
        );
      case 'relative3d':
        return (
          <Relative3DAnalysis
            selectedJobId={selectedJobId}
            onSelectJob={handleSelectJob}
            onNavigate={navigate}
          />
        );
      case 'geospatial':
        return (
          <GeospatialIntelligence
            selectedJobId={selectedJobId}
            onSelectJob={handleSelectJob}
            onNavigate={navigate}
          />
        );
      case 'maintenance':
        return (
          <MaintenanceIntelligence
            selectedJobId={selectedJobId}
            onSelectJob={handleSelectJob}
            onNavigate={navigate}
          />
        );
      case 'performance':
        return (
          <Performance
            selectedJobId={selectedJobId}
            onSelectJob={handleSelectJob}
            onNavigate={navigate}
          />
        );
      case 'reports':
        return (
          <Reports
            selectedJobId={selectedJobId}
            onSelectJob={handleSelectJob}
            onNavigate={navigate}
          />
        );
      case 'settings':
        return <Settings />;
      case 'legacy':
        return <LegacySuite onOpenV2={() => navigate('overview')} />;
      default:
        return <Overview onNavigate={navigate} />;
    }
  };

  return (
    <div className="rv-shell">
      <TopBar
        activeSection={activeSection}
        selectedJobId={selectedJobId}
        onNavigate={navigate}
      />
      <div className="rv-main-layout">
        <Sidebar active={activeSection} onNavigate={(sec) => navigate(sec)} />
        <main className={`rv-content ${activeSection === 'legacy' ? 'rv-content-legacy' : ''}`}>
          {renderSection()}
        </main>
      </div>
      <Footer />
    </div>
  );
}
