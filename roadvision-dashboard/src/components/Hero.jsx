import React from 'react';
import { motion } from 'framer-motion';
import { ChevronDown, ArrowRight, ShieldCheck, Activity, Layers, FileSpreadsheet } from 'lucide-react';

export default function Hero({ onExplore }) {
  const handleDiscoverClick = () => {
    const el = document.getElementById('discover-intro');
    if (el) {
      const prefersReducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
      el.scrollIntoView({ behavior: prefersReducedMotion ? 'auto' : 'smooth' });
    }
  };

  return (
    <div className="landing-editorial-wrap">
      {/* ── CINEMATIC PHOTOGRAPHIC HERO ── */}
      <section className="editorial-hero" aria-label="RoadVision Autonomous Road Intelligence">
        <div className="editorial-hero-media">
          <img
            src="/road-inspection-hero.jpg"
            alt="Cinematic perspective of weathered asphalt highway surface stretching toward distant mountains"
            className="editorial-hero-img"
            loading="eager"
          />
          <div className="editorial-hero-scrim" />
        </div>

        <motion.div
          className="editorial-hero-body"
          initial={{ opacity: 0, y: 24 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.75, ease: [0.22, 1, 0.36, 1] }}
        >
          <p className="editorial-eyebrow">
            MUNICIPAL INFRASTRUCTURE TELEMETRY
          </p>

          <h1 className="editorial-hero-headline">
            AUTONOMOUS ROAD INTELLIGENCE
          </h1>

          <div className="editorial-hero-actions">
            <button
              className="editorial-pill-btn"
              onClick={handleDiscoverClick}
              aria-label="Discover RoadVision AI platform capabilities"
            >
              <span>DISCOVER</span>
              <ChevronDown size={15} />
            </button>
          </div>
        </motion.div>
      </section>

      {/* ── INTRODUCTORY SECTION ── */}
      <section id="discover-intro" className="editorial-intro-section" aria-label="Platform Overview">
        <div className="intro-container">
          <div className="intro-header">
            <span className="intro-tag">PLATFORM CAPABILITIES</span>
            <h2 className="intro-title">
              Millimeter-calibrated road surface auditing from standard edge video.
            </h2>
            <p className="intro-lead">
              RoadVision AI bridges optical computer vision and physical infrastructure engineering.
              By combining fine-tuned spatial detection, Kalman-filter tracking, and monocular 3D depth geometry,
              it transforms raw dashcam recordings into automated municipal maintenance work orders without expensive LiDAR arrays.
            </p>
          </div>

          <div className="intro-grid">
            <div className="intro-card">
              <div className="intro-card-icon">
                <ShieldCheck size={20} />
              </div>
              <h3>Spatial Localization</h3>
              <p>
                Fine-tuned YOLOv8 neural network detects road cavities, asphalt cracking, and surface depression contours in real time.
              </p>
            </div>

            <div className="intro-card">
              <div className="intro-card-icon">
                <Activity size={20} />
              </div>
              <h3>ByteTrack Trajectory Matching</h3>
              <p>
                Associates high- and low-confidence detections across consecutive frames to maintain persistent defect IDs and eliminate duplicate recounting.
              </p>
            </div>

            <div className="intro-card">
              <div className="intro-card-icon">
                <Layers size={20} />
              </div>
              <h3>MiDaS 3D Metric Calibration</h3>
              <p>
                Calculates relative disparity maps, calibrated against camera optics and annular road-rim baselines to estimate physical centimeter depth.
              </p>
            </div>

            <div className="intro-card">
              <div className="intro-card-icon">
                <FileSpreadsheet size={20} />
              </div>
              <h3>Automated Municipal Auditing</h3>
              <p>
                Classifies hazards into Mild, Moderate, and Severe tiers, logging timestamps, depth metrics, and downloadable CSV records for public works dispatch.
              </p>
            </div>
          </div>

          <div className="intro-footer-action">
            <button className="btn-editorial-explore" onClick={onExplore}>
              <span>Launch Command Center</span>
              <ArrowRight size={16} />
            </button>
          </div>
        </div>
      </section>
    </div>
  );
}
