import React from 'react';
import {
  AlertTriangle,
  WifiOff,
  Loader2,
  Inbox,
  ShieldAlert,
  Info,
} from 'lucide-react';
import {
  MEASUREMENT_STATES,
  gpsProvenanceTone,
  inspectionProvenanceTone,
  matchStatusTone,
  measurementStateTone,
  qualityCategoryTone,
  statusTone,
} from '../utils/formatters';

export function ProvenanceBadge({ provenance }) {
  const label = provenance || 'UNKNOWN';
  return <span className={`rv-badge rv-badge-${inspectionProvenanceTone(label)}`}>{label}</span>;
}

export function GpsProvenanceBadge({ gpsProvenance, isSynthetic = false }) {
  const label = isSynthetic ? 'SYNTHETIC' : (gpsProvenance || 'UNKNOWN');
  return <span className={`rv-badge rv-badge-${gpsProvenanceTone(label, isSynthetic)}`}>{label}</span>;
}

export function StatusBadge({ status }) {
  const label = status || 'UNKNOWN';
  return <span className={`rv-badge rv-badge-${statusTone(label)}`}>{label}</span>;
}

export function MeasurementStateBadge({ state }) {
  const label = state || 'UNKNOWN';
  return <span className={`rv-badge rv-badge-${measurementStateTone(label)}`}>{label}</span>;
}

export function QualityBadge({ category }) {
  const label = category || 'UNAVAILABLE';
  return <span className={`rv-badge rv-badge-${qualityCategoryTone(label)}`}>{label}</span>;
}

export function MatchStatusBadge({ status }) {
  const label = status || 'UNMATCHED';
  return <span className={`rv-badge rv-badge-${matchStatusTone(label)}`}>{label}</span>;
}

const COMPONENT_LABELS = [
  { key: 'detection_quality', label: 'Detection Quality' },
  { key: 'geometric_reference_quality', label: 'Geometric Ref Plane' },
  { key: 'cavity_extraction_quality', label: 'Cavity Extraction' },
  { key: 'temporal_consistency', label: 'Temporal Consistency' },
  { key: 'measurement_freshness', label: 'Measurement Freshness' },
  { key: 'environmental_risk', label: 'Environmental Risk' },
];

export function QualityBreakdownCard({ assessment }) {
  if (!assessment) {
    return (
      <StatePanel
        variant="unavailable"
        title="Measurement quality unavailable"
        message="This observation or historical job does not include Phase 9 quality diagnostics."
      />
    );
  }
  const comps = assessment.components || {};
  const explanations = assessment.diagnostic_explanations || [];
  const missing = assessment.missing_evidence || [];

  return (
    <div className="rv-quality-card">
      <div className="rv-card-header" style={{ marginBottom: 10 }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 10, flexWrap: 'wrap' }}>
          <QualityBadge category={assessment.quality_category} />
          <span className="rv-mono" style={{ fontSize: 13 }}>
            Heuristic Quality Index:{' '}
            <strong>
              {assessment.quality_index !== null && assessment.quality_index !== undefined
                ? assessment.quality_index.toFixed(3)
                : 'Unavailable'}
            </strong>
          </span>
        </div>
        <span className="rv-card-meta">
          Experimental heuristic (not a probability) · Coverage:{' '}
          {assessment.evidence_coverage_ratio !== null && assessment.evidence_coverage_ratio !== undefined
            ? `${(assessment.evidence_coverage_ratio * 100).toFixed(0)}%`
            : 'Unavailable'}
        </span>
      </div>

      <div className="rv-bar-list">
        {COMPONENT_LABELS.map(({ key, label }) => {
          const val = comps[key];
          const pct = val !== null && val !== undefined ? Math.round(val * 100) : null;
          return (
            <div key={key} className="rv-bar-row">
              <span className="rv-bar-label">{label}</span>
              <div className="rv-bar-track">
                <div className="rv-bar-fill" style={{ width: pct !== null ? `${pct}%` : '0%' }} />
              </div>
              <span className="rv-bar-val">{val !== null && val !== undefined ? val.toFixed(2) : 'Unavailable'}</span>
            </div>
          );
        })}
      </div>

      {assessment.is_cached_inherited && (
        <p style={{ fontSize: 12, color: 'var(--rv-cyan-light)', marginTop: 10 }}>
          Inherited from fresh measurement at {assessment.source_fresh_timestamp_sec?.toFixed?.(2) ?? '—'}s with freshness penalty (age {assessment.measurement_age_sec?.toFixed?.(2) ?? '—'}s).
        </p>
      )}

      {missing.length > 0 && (
        <p style={{ fontSize: 12, color: 'var(--rv-amber)', marginTop: 8 }}>
          Missing evidence: {missing.join(', ')}
        </p>
      )}

      {explanations.length > 0 && (
        <div style={{ marginTop: 10, display: 'flex', flexDirection: 'column', gap: 6 }}>
          {explanations.map((ex) => (
            <div key={ex.code} className="rv-kv-item" style={{ padding: '7px 10px' }}>
              <span className="rv-kv-label">{ex.title} ({ex.code})</span>
              <span style={{ fontSize: 12, color: 'var(--rv-text-secondary)' }}>{ex.explanation}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

export function StatePanel({
  variant = 'empty', // loading | empty | error | disconnected | api-disabled | unavailable
  title,
  message,
  action,
}) {
  const iconMap = {
    loading: <Loader2 size={22} className="rv-spin rv-icon-cyan" />,
    disconnected: <WifiOff size={22} className="rv-icon-danger" />,
    'api-disabled': <ShieldAlert size={22} className="rv-icon-warning" />,
    error: <AlertTriangle size={22} className="rv-icon-danger" />,
    unavailable: <Info size={22} className="rv-icon-muted" />,
    empty: <Inbox size={22} className="rv-icon-muted" />,
  };

  return (
    <div className={`rv-state-panel rv-state-${variant}`} role="status">
      <div className="rv-state-icon">{iconMap[variant] || iconMap.empty}</div>
      <div className="rv-state-body">
        {title && <h3 className="rv-state-title">{title}</h3>}
        {message && <p className="rv-state-message">{message}</p>}
      </div>
      {action && <div className="rv-state-action">{action}</div>}
    </div>
  );
}

export function ScientificDisclaimer({ text }) {
  const defaultText =
    'RoadVision AI 2.0 produces uncalibrated relative inverse-depth residuals only (not metric cm/mm, physical volume, or physical severity classes). Session track counts reflect ByteTrack IDs, not independently confirmed physical potholes.';
  return (
    <div className="rv-disclaimer" role="note">
      <Info size={16} className="rv-disclaimer-icon" />
      <span>{text || defaultText}</span>
    </div>
  );
}

export function MeasurementStateDistribution({ countsByState = {}, totalObservations = 0 }) {
  const total = Math.max(0, Number(totalObservations) || 0);
  return (
    <div className="rv-dist-wrap">
      <div className="rv-stacked-bar" aria-label="Measurement state distribution">
        {total === 0 ? (
          <div className="rv-stacked-empty" />
        ) : (
          MEASUREMENT_STATES.map((state) => {
            const count = Number(countsByState?.[state] || 0);
            if (count <= 0) return null;
            const pct = (count / total) * 100;
            return (
              <div
                key={state}
                className={`rv-stack-seg rv-seg-${measurementStateTone(state)}`}
                style={{ width: `${pct}%` }}
                title={`${state}: ${count} (${pct.toFixed(1)}%)`}
              />
            );
          })
        )}
      </div>
      <div className="rv-dist-legend">
        {MEASUREMENT_STATES.map((state) => {
          const count = Number(countsByState?.[state] || 0);
          const pct = total > 0 ? ((count / total) * 100).toFixed(1) : '0.0';
          return (
            <div key={state} className="rv-legend-item">
              <span className={`rv-dot rv-dot-${measurementStateTone(state)}`} />
              <span className="rv-legend-name">{state}</span>
              <span className="rv-legend-val">
                {count} ({pct}%)
              </span>
            </div>
          );
        })}
      </div>
    </div>
  );
}
