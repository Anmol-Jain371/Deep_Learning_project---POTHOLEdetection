import React from 'react';
import { motion } from 'framer-motion';
import { Target, AlertTriangle, Gauge, ArrowDown, Cpu, Droplets } from 'lucide-react';

export default function KPICards({ data = [], stats = {} }) {
  const safeData = data || [];
  const total = stats.total_hazards ?? safeData.length;
  const severe = stats.severe_count ?? safeData.filter(d => (d.severity || '').includes('SEVERE')).length;
  const moderate = stats.moderate_count ?? safeData.filter(d => (d.severity || '').includes('MODERATE')).length;
  const waterHazards = stats.water_count ?? safeData.filter(d => d.water || (d.severity || '').includes('WATER')).length;

  const depths = safeData
    .map((d) => (d.depth_cm === null || d.depth_cm === undefined ? null : Number(d.depth_cm)))
    .filter((val) => val !== null && !Number.isNaN(val) && Number.isFinite(val));
  const hasDepth = depths.length > 0;
  const maxDepth = hasDepth ? Math.max(...depths) : null;

  // Legacy experimental summary index (not calibrated engineering PCI)
  const rqi = safeData.length === 0
    ? 100
    : hasDepth
      ? Math.max(0, Math.min(100, Math.round(100 - (severe * 14) - (moderate * 6) - (maxDepth * 0.7))))
      : null;

  const surveyDuration = safeData.length > 0
    ? Math.round(Number(safeData[safeData.length - 1]?.timestamp ?? 0))
    : (stats.processing_time_sec !== null && stats.processing_time_sec !== undefined ? Math.round(stats.processing_time_sec) : 0);

  const hasValidConf = stats?.avg_confidence !== null && stats?.avg_confidence !== undefined && !Number.isNaN(Number(stats.avg_confidence));

  const cards = [
    {
      icon: Target,
      value: total,
      label: 'Total Legacy Detections',
      sub: total === 0 ? 'No defects registered' : `Logged across ${surveyDuration}s survey`,
      color: '#e8651a',
      bg: 'rgba(232, 101, 26, 0.08)',
    },
    {
      icon: AlertTriangle,
      value: severe,
      label: 'Legacy Experimental Tier (Severe)',
      sub: severe > 0 ? `${severe} legacy tier flags — physical accuracy not established` : 'No legacy severe tier flags',
      color: severe > 0 ? '#dc2626' : '#16a34a',
      bg: severe > 0 ? 'rgba(220, 38, 38, 0.08)' : 'rgba(22, 163, 74, 0.08)',
    },
    {
      icon: Gauge,
      value: rqi !== null ? `${rqi}/100` : 'Unavailable',
      label: 'Legacy Experimental Index',
      sub: 'Uncalibrated experimental estimate — physical accuracy not established.',
      color: rqi === null ? '#64748b' : rqi < 65 ? '#dc2626' : rqi < 85 ? '#d97706' : '#16a34a',
      bg: 'rgba(100, 116, 139, 0.08)',
    },
    {
      icon: ArrowDown,
      value: hasDepth ? `${maxDepth.toFixed(1)} cm (uncal.)` : 'Unavailable',
      label: 'Legacy Peak Estimate',
      sub: 'Uncalibrated experimental estimate — physical accuracy not established.',
      color: !hasDepth ? '#64748b' : maxDepth > 8.0 ? '#dc2626' : maxDepth > 4.0 ? '#d97706' : '#16a34a',
      bg: 'rgba(232, 101, 26, 0.08)',
    },
  ];

  return (
    <div className="kpi-section-wrap">
      <div className="kpi-grid">
        {cards.map((card, i) => {
          const Icon = card.icon;
          return (
            <motion.div
              key={i}
              className="kpi-card"
              initial={{ opacity: 0, y: 16 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.35, delay: i * 0.06, ease: [0.22, 1, 0.36, 1] }}
            >
              <div className="kpi-indicator" style={{ background: card.color }} />
              <div className="kpi-icon" style={{ background: card.bg, color: card.color }}>
                <Icon size={18} />
              </div>
              <div className="kpi-value" style={{ color: card.color }}>{card.value}</div>
              <div className="kpi-label">{card.label}</div>
              <div className="kpi-sub">{card.sub}</div>
            </motion.div>
          );
        })}
      </div>

      {/* Model Performance Ribbon when stats are available */}
      {stats && stats.avg_fps && (
        <div className="model-benchmark-ribbon">
          <div className="bench-pill">
            <Cpu size={13} style={{ color: '#e8651a' }} />
            <span>Inference: <strong>{Number(stats.avg_fps).toFixed(1)} FPS</strong></span>
          </div>
          <div className="bench-pill">
            <span>Frames: <strong>{stats.total_frames || 0}</strong></span>
          </div>
          <div className="bench-pill">
            <span>
              AI Confidence:{' '}
              <strong>
                {hasValidConf ? `${Math.round(Number(stats.avg_confidence) * 100)}%` : 'Unavailable'}
              </strong>
            </span>
          </div>
          {waterHazards > 0 && (
            <div className="bench-pill water-alert">
              <Droplets size={13} style={{ color: '#0284c7' }} />
              <span>Aquaplane Risks: <strong>{waterHazards}</strong></span>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
