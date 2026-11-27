import React, { useEffect, useState, useCallback } from 'react';
import { getObservations } from '../api/client';
import { ChevronLeft, ChevronRight, RefreshCw, Box } from 'lucide-react';
import InspectionSelector from '../components/InspectionSelector';
import {
  StatePanel,
  MeasurementStateBadge,
  QualityBadge,
  QualityBreakdownCard,
  ScientificDisclaimer,
} from '../components/StateBanner';
import { MEASUREMENT_STATES, isValidUuid, formatNullable } from '../utils/formatters';

export default function HazardExplorer({ selectedJobId, onSelectJob, onNavigate }) {
  const [observations, setObservations] = useState([]);
  const [selectedObs, setSelectedObs] = useState(null);
  const [total, setTotal] = useState(0);
  const [totalPages, setTotalPages] = useState(0);
  const [page, setPage] = useState(1);
  const [pageSize] = useState(25);
  const [disclaimer, setDisclaimer] = useState('');

  const [stateFilter, setStateFilter] = useState('');
  const [trackFilter, setTrackFilter] = useState('');
  const [minConf, setMinConf] = useState('');

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const loadObservations = useCallback(async () => {
    if (!selectedJobId || !isValidUuid(selectedJobId)) return;
    setLoading(true);
    setError(null);
    try {
      const resp = await getObservations(selectedJobId, {
        page,
        page_size: pageSize,
        ...(stateFilter ? { measurement_state: stateFilter } : {}),
        ...(trackFilter.trim() !== '' ? { track_id: trackFilter.trim() } : {}),
        ...(minConf.trim() !== '' ? { min_conf: minConf.trim() } : {}),
      });
      const items = resp?.observations || [];
      setObservations(items);
      setSelectedObs(items[0] || null);
      setTotal(resp?.total ?? 0);
      setTotalPages(resp?.total_pages ?? 0);
      setDisclaimer(resp?.scientific_disclaimer || '');
    } catch (err) {
      setObservations([]);
      setSelectedObs(null);
      setError(err);
    } finally {
      setLoading(false);
    }
  }, [selectedJobId, page, pageSize, stateFilter, trackFilter, minConf]);

  useEffect(() => {
    setPage(1);
  }, [selectedJobId]);

  useEffect(() => {
    loadObservations();
  }, [loadObservations]);

  return (
    <div className="rv-page">
      <div className="rv-page-header">
        <div>
          <h2 className="rv-page-title">Hazard Explorer &amp; Measurement Quality</h2>
          <p className="rv-page-subtitle">
            Inspect per-frame candidate detections, heuristic measurement quality breakdowns, and explainable rejection diagnostics.
          </p>
        </div>
      </div>

      <InspectionSelector
        selectedJobId={isValidUuid(selectedJobId) ? selectedJobId : ''}
        onSelectJob={onSelectJob}
        onNavigate={onNavigate}
        label="Completed Inspection"
      />

      {!selectedJobId ? (
        <StatePanel
          variant="empty"
          title="Select an inspection"
          message="Choose a completed inspection from the selector above to explore hazard observations."
        />
      ) : (
        <>
          <section className="rv-card">
            <div className="rv-filter-bar">
              <div className="rv-field">
                <label className="rv-field-label" htmlFor="rv-obs-state">Measurement State</label>
                <select
                  id="rv-obs-state"
                  className="rv-select"
                  value={stateFilter}
                  onChange={(e) => {
                    setStateFilter(e.target.value);
                    setPage(1);
                  }}
                >
                  <option value="">All States</option>
                  {MEASUREMENT_STATES.map((st) => (
                    <option key={st} value={st}>{st}</option>
                  ))}
                </select>
              </div>

              <div className="rv-field">
                <label className="rv-field-label" htmlFor="rv-obs-track">Track ID</label>
                <input
                  id="rv-obs-track"
                  type="number"
                  placeholder="Any track"
                  className="rv-input"
                  style={{ width: 130 }}
                  value={trackFilter}
                  onChange={(e) => {
                    setTrackFilter(e.target.value);
                    setPage(1);
                  }}
                />
              </div>

              <div className="rv-field">
                <label className="rv-field-label" htmlFor="rv-obs-conf">Min Confidence (0–1)</label>
                <input
                  id="rv-obs-conf"
                  type="number"
                  step="0.05"
                  min="0"
                  max="1"
                  placeholder="e.g. 0.30"
                  className="rv-input"
                  style={{ width: 140 }}
                  value={minConf}
                  onChange={(e) => {
                    setMinConf(e.target.value);
                    setPage(1);
                  }}
                />
              </div>

              <button type="button" className="rv-btn rv-btn-ghost rv-btn-sm" onClick={loadObservations} disabled={loading}>
                <RefreshCw size={14} className={loading ? 'rv-spin' : ''} /> Reload
              </button>
            </div>

            {loading ? (
              <StatePanel variant="loading" title="Loading observations" message="Querying paginated HazardObservation records…" />
            ) : error ? (
              <StatePanel
                variant={error.status === 409 ? 'unavailable' : error.isNetworkError ? 'disconnected' : 'error'}
                title={error.status === 409 ? 'Observations not ready' : 'Failed to load observations'}
                message={error.message}
                action={
                  <button type="button" className="rv-btn rv-btn-secondary rv-btn-sm" onClick={() => onNavigate('workspace', selectedJobId)}>
                    Open Workspace
                  </button>
                }
              />
            ) : observations.length === 0 ? (
              <StatePanel
                variant="empty"
                title="No matching observations"
                message="No hazard observations match the current filter criteria for this inspection."
              />
            ) : (
              <>
                <div className="rv-table-wrap">
                  <table className="rv-table">
                    <thead>
                      <tr>
                        <th>Frame</th>
                        <th>Time (s)</th>
                        <th>Track</th>
                        <th>Conf.</th>
                        <th>State</th>
                        <th>Quality Category</th>
                        <th>Quality Idx</th>
                        <th>Raw Rel.</th>
                        <th>Smoothed Rel.</th>
                        <th>3D Snap</th>
                      </tr>
                    </thead>
                    <tbody>
                      {observations.map((obs, i) => {
                        const fc = obs.frame_context || {};
                        const det = obs.detection || {};
                        const meas = obs.measurement || {};
                        const qa = meas.quality_assessment;
                        const isSel = selectedObs === obs;
                        return (
                          <tr
                            key={`${fc.frame_idx}-${det.track_id}-${i}`}
                            className={`rv-row-clickable ${isSel ? 'rv-row-selected' : ''}`}
                            onClick={() => setSelectedObs(obs)}
                          >
                            <td className="rv-mono">{formatNullable(fc.frame_idx)}</td>
                            <td className="rv-mono">{formatNullable(fc.timestamp_sec, 2)}</td>
                            <td className="rv-mono">#{formatNullable(det.track_id)}</td>
                            <td className="rv-mono">
                              {det.detector_confidence !== null && det.detector_confidence !== undefined
                                ? `${(det.detector_confidence * 100).toFixed(1)}%`
                                : 'Unavailable'}
                            </td>
                            <td><MeasurementStateBadge state={meas.measurement_state} /></td>
                            <td>{qa ? <QualityBadge category={qa.quality_category} /> : 'Unavailable'}</td>
                            <td className="rv-mono">{formatNullable(qa?.quality_index, 3)}</td>
                            <td className="rv-mono">{formatNullable(meas.raw_relative_cavity_score, 4)}</td>
                            <td className="rv-mono">{formatNullable(meas.smoothed_relative_score, 4)}</td>
                            <td>
                              {meas.geometry_snapshot_id ? (
                                <button
                                  type="button"
                                  className="rv-btn rv-btn-ghost rv-btn-sm"
                                  onClick={(e) => {
                                    e.stopPropagation();
                                    onNavigate('relative3d', selectedJobId);
                                  }}
                                >
                                  <Box size={13} /> 3D
                                </button>
                              ) : (
                                '—'
                              )}
                            </td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </div>

                <div className="rv-pagination">
                  <span>
                    {total} total matching observations · Page {page} of {Math.max(1, totalPages)}
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

          {selectedObs && (
            <section className="rv-card">
              <div className="rv-card-header">
                <h3 className="rv-card-title">
                  Selected Observation Details — Frame {selectedObs.frame_context?.frame_idx} · Track #{selectedObs.detection?.track_id}
                </h3>
                <span className="rv-card-meta">
                  Inlier Ratio: {formatNullable(selectedObs.measurement?.ref_diagnostics?.inlier_ratio, 3)} · Water: {selectedObs.measurement?.is_water_suspected ? 'Suspected' : 'No'}
                </span>
              </div>
              <QualityBreakdownCard assessment={selectedObs.measurement?.quality_assessment} />
            </section>
          )}
        </>
      )}

      <ScientificDisclaimer text={disclaimer} />
    </div>
  );
}
