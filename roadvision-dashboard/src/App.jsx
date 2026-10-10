import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import Navbar from './components/Navbar';
import Hero from './components/Hero';
import KPICards from './components/KPICards';
import AuditTable from './components/AuditTable';
import PipelineSection from './components/PipelineSection';
import VideoUploadAnalysis from './components/VideoUploadAnalysis';
import Footer from './components/Footer';
import './App.css';

export default function App() {
  const [activeSection, setActiveSection] = useState('home');

  // Real analysis results from backend — null until an actual video is processed
  const [analysisResults, setAnalysisResults] = useState(null);

  const handleAnalysisComplete = (data) => {
    // data = { stats, auditRecords, videoUrl, filename }
    setAnalysisResults(data);
  };

  // The entire application uses the unified warm editorial design system
  const isEditorial = true;

  return (
    <div className="app-shell theme-editorial">
      <Navbar
        active={activeSection}
        onNavigate={setActiveSection}
        variant="editorial"
      />

      <AnimatePresence mode="wait">
        {activeSection === 'home' && (
          <motion.div
            key="home"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.4 }}
            className="editorial-page-canvas"
          >
            <Hero onExplore={() => setActiveSection('dashboard')} />
            <PipelineSection />
          </motion.div>
        )}

        {activeSection === 'dashboard' && (
          <motion.div
            key="dashboard"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
            transition={{ duration: 0.35, ease: [0.22, 1, 0.36, 1] }}
            className="editorial-page-canvas dashboard-editorial-canvas"
          >
            {/* Dashboard Workspace Header */}
            <div className="dashboard-editorial-header">
              <span className="dashboard-eyebrow">MUNICIPAL AUDIT WORKSPACE</span>
              <h1 className="dashboard-main-title">Road Condition Assessment</h1>
              <p className="dashboard-main-lead">
                Review detected defects, estimated severity, and survey records.
              </p>
            </div>

            {/* 1. Video Upload & Player */}
            <div className="dashboard-video-section">
              <VideoUploadAnalysis onAnalysisComplete={handleAnalysisComplete} />
            </div>

            {/* 2. Results Section — Only shown after REAL analysis completes */}
            {analysisResults && (
              <motion.div
                className="analysis-results-section"
                initial={{ opacity: 0, y: 30 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.5, delay: 0.2, ease: [0.22, 1, 0.36, 1] }}
              >
                {/* Summary KPI Cards from real analysis data */}
                <KPICards data={analysisResults.auditRecords || []} stats={analysisResults.stats || {}} />

                {/* Audit Table with real detection records */}
                <AuditTable data={analysisResults.auditRecords || []} />
              </motion.div>
            )}
          </motion.div>
        )}

        {activeSection === 'architecture' && (
          <motion.div
            key="architecture"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
            transition={{ duration: 0.35, ease: [0.22, 1, 0.36, 1] }}
            className="editorial-page-canvas architecture-canvas"
          >
            <PipelineSection expanded />
          </motion.div>
        )}
      </AnimatePresence>

      <Footer />
    </div>
  );
}
