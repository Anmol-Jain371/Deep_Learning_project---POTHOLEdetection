import React, { useEffect, useState, useCallback } from 'react';
import { getTracks, getObservations, getJobLocations, getGeometrySnapshots } from '../api/client';
import { ChevronLeft, ChevronRight, GitCommitHorizontal, Box, MapPin } from 'lucide-react';
import InspectionSelector from '../components/InspectionSelector';
import {
  StatePanel,
  MeasurementStateBadge,
  QualityBadge,
  MatchStatusBadge,
  ScientificDisclaimer,
} from '../components/StateBanner';
import { isValidUuid, formatNullable, formatRatePercent } from '../utils/formatters';

export default function TrackAnalysis({ selectedJobId, onSelectJob, onNavigate }) {
  const [tracks, setTracks] = useState([]);
  const [totalTracks, setTotalTracks] = useState(0);
  const [totalPages, setTotalPages] = useState(0);
  const [page, setPage] = useState(1);
  const [pageSize] = useState(20);
  const [trackNote, setTrackNote] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const [selectedTrackId, setSelectedTrackId] = useState(null);
  const [trackObs, setTrackObs] = useState([]);
  const [loadingObs, setLoadingObs] = useState(false);
  const [obsError, setObsError] = useState(null);

  const [locMap, setLocMap] = useState({});
  const [geomTracks, setGeomTracks] = useState(new Set());

  const loadTracks = useCallback(async () => {
    if (!selectedJobId || !isValidUuid(selectedJobId)) return;
    setLoading(true);
    setError(null);
    try {
      const [resp, locResp, geomResp] = await Promise.all([
        getTracks(selectedJobId, { page, page_size: pageSize }),
        getJobLocations(selectedJobId).catch(() => null),
        getGeometrySnapshots(selectedJobId).catch(() => null),
      ]);
      const items = resp?.tracks || [];
      setTracks(items);
      setTotalTracks(resp?.unique_session_track_count ?? resp?.total ?? items.length);
      setTotalPages(resp?.total_pages ?? 0);
      setTrackNote(resp?.note || '');

      const nextLocMap = {};
      for (const l of locResp?.locations || []) {
        nextLocMap[l.track_id] = l;
      }
      setLocMap(nextLocMap);

      const gSet = new Set((geomResp?.snapshots || []).map((s) => s.track_id));
      setGeomTracks(gSet);

      if (items.length > 0 && !items.some((t) => t.track_id === selectedTrackId)) {
        setSelectedTrackId(items[0].track_id);
      } else if (items.length === 0) {
        setSelectedTrackId(null);
      }
    } catch (err) {
      setTracks([]);
      setSelectedTrackId(null);
      setError(err);
    } finally {
      setLoading(false);
    }
  }, [selectedJobId, page, pageSize, selectedTrackId]);

  useEffect(() => {
    setPage(1);
    setSelectedTrackId(null);
    setTrackObs([]);
  }, [selectedJobId]);

  useEffect(() => {
    loadTracks();
  }, [loadTracks]);

  useEffect(() => {
    let mounted = true;
    if (!selectedJobId || !isValidUuid(selectedJobId) || selectedTrackId === null) {
      setTrackObs([]);
      return undefined;
    }
    setLoadingObs(true);
    setObsError(null);
    getObservations(selectedJobId, { track_id: selectedTrackId, page: 1, page_size: 200 })
      .then((resp) => {
        if (mounted) setTrackObs(resp?.observations || []);
      })
      .catch((err) => {
        if (mounted) {
          setTrackObs([]);
          setObsError(err.message);
        }
      })
      .finally(() => {
        if (mounted) setLoadingObs(false);
      });
    return () => {
      mounted = false;
    };
  }, [selectedJobId, selectedTrackId]);

  const activeTrack = tracks.find((t) => t.track_id === selectedTrackId) || null;
  const activeQuality = activeTrack?.quality_summary || null;
  const activeLoc = selectedTrackId !== null ? locMap[selectedTrackId] : null;

  const validMeasurements = trackObs.filter((o) => {
    const m = o.measurement;
    return m && m.smoothed_relative_score !== null && m.smoothed_relative_score !== undefined;
  });

  const maxSmoothed = validMeasurements.reduce(
    (mx, o) => Math.max(mx, Math.abs(Number(o.measurement.smoothed_relative_score) || 0)),
    0.0001
  );

  return (
    <div className="rv-page">
      <div className="rv-page-header">
        <div>
          <h2 className="rv-page-title">Track Analysis &amp; Stability</h2>
          <p className="rv-page-subtitle">
            Evaluate track-level measurement quality, fresh vs cached evidence coverage, relative-depth variability, and approximate location.
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
        <StatePanel variant="empty" title="Select an inspection" message="Choose an inspection above to inspect session tracks." />
      ) : loading ? (
        <StatePanel variant="loading" title="Loading session tracks" message="Fetching ByteTrack records…" />
      ) : error ? (
        <StatePanel
          variant={error.status === 409 ? 'unavailable' : error.isNetworkError ? 'disconnected' : 'error'}
          title={error.status === 409 ? 'Tracks unavailable' : 'Failed to load tracks'}
          message={error.message}
        />
      ) : tracks.length === 0 ? (
        <StatePanel variant="empty" title="No tracks found" message="No candidate tracks were recorded in this inspection." />
      ) : (
        <>
          <div className="rv-grid-2">
            <section className="rv-card">
              <div className="rv-card-header">
                <h3 className="rv-card-title">
                  <GitCommitHorizontal size={16} className="rv-icon-cyan" /> Session Candidate Tracks ({totalTracks})
                </h3>
              </div>
              {trackNote && (
                <p style={{ fontSize: 12, color: 'var(--rv-text-secondary)', marginBottom: 12 }}>
                  {trackNote}
                </p>
              )}
              <div className="rv-table-wrap">
                <table className="rv-table">
                  <thead>
                    <tr>
                      <th>Track ID</th>
                      <th>Frames</th>
                      <th>Obs.</th>
                      <th>Latest State</th>
                      <th>Quality</th>
                    </tr>
                  </thead>
                  <tbody>
                    {tracks.map((t) => (
                      <tr
                        key={`${t.track_id}-${t.track_generation_id}`}
                        className={`rv-row-clickable ${selectedTrackId === t.track_id ? 'rv-row-selected' : ''}`}
                        onClick={() => setSelectedTrackId(t.track_id)}
                      >
                        <td className="rv-mono">#{t.track_id}</td>
                        <td className="rv-mono">{t.first_seen_frame_idx} → {t.last_seen_frame_idx}</td>
                        <td className="rv-mono">{t.observation_count}</td>
                        <td><MeasurementStateBadge state={t.latest_state} /></td>
                        <td>
                          {t.quality_summary ? (
                            <QualityBadge category={t.quality_summary.track_quality_category} />
                          ) : (
                            'Unavailable'
                          )}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>

              <div className="rv-pagination">
                <span>Page {page} of {Math.max(1, totalPages)}</span>
                <div className="rv-pagination-btns">
                  <button
                    type="button"
                    className="rv-btn rv-btn-ghost rv-btn-sm"
                    disabled={page <= 1}
                    onClick={() => setPage((p) => Math.max(1, p - 1))}
                  >
                    <ChevronLeft size={14} /> Prev
                  </button>
                  <button
                    type="button"
                    className="rv-btn rv-btn-ghost rv-btn-sm"
                    disabled={page >= totalPages}
                    onClick={() => setPage((p) => p + 1)}
                  >
                    Next <ChevronRight size={14} />
                  </button>
                </div>
              </div>
            </section>

            <section className="rv-card">
              <div className="rv-card-header">
                <h3 className="rv-card-title">
                  {selectedTrackId !== null ? `Track #${selectedTrackId} Quality, Stability & Location` : 'Select a Track'}
                </h3>
                <div className="rv-header-actions">
                  {geomTracks.has(selectedTrackId) && (
                    <button
                      type="button"
                      className="rv-btn rv-btn-secondary rv-btn-sm"
                      onClick={() => onNavigate('relative3d', selectedJobId)}
                    >
                      <Box size={14} /> View 3D Surface
                    </button>
                  )}
                  {activeLoc?.lat !== null && activeLoc?.lat !== undefined && (
                    <button
                      type="button"
                      className="rv-btn rv-btn-ghost rv-btn-sm"
                      onClick={() => onNavigate('geospatial', selectedJobId)}
                    >
                      <MapPin size={14} /> Map
                    </button>
                  )}
                </div>
              </div>

              {activeQuality ? (
                <div style={{ marginBottom: 14 }}>
                  <div className="rv-kv-list">
                    <div className="rv-kv-item">
                      <span className="rv-kv-label">Quality Category / Median Fresh Index</span>
                      <span className="rv-kv-value">
                        {activeQuality.track_quality_category} · {formatNullable(activeQuality.median_fresh_quality_index, 3)}
                      </span>
                    </div>
                    <div className="rv-kv-item">
                      <span className="rv-kv-label">Fresh Valid / Invalid / Cached</span>
                      <span className="rv-kv-value">
                        {activeQuality.fresh_valid_count} / {activeQuality.fresh_invalid_count} / {activeQuality.cached_measurement_count} ({formatRatePercent(activeQuality.fresh_measurement_coverage)} fresh)
                      </span>
                    </div>
                    <div className="rv-kv-item">
                      <span className="rv-kv-label">Median Score / MAD / IQR</span>
                      <span className="rv-kv-value">
                        {formatNullable(activeQuality.median_relative_score, 4)} / {formatNullable(activeQuality.mad_relative_score, 4)} / {formatNullable(activeQuality.iqr_relative_score, 4)}
                      </span>
                    </div>
                    <div className="rv-kv-item">
                      <span className="rv-kv-label">Temporal Variability ({activeQuality.evidence_sufficiency})</span>
                      <span className="rv-kv-value">{formatNullable(activeQuality.relative_depth_variability, 3)}</span>
                    </div>
                  </div>
                  {activeQuality.insufficient_evidence_reason && (
                    <p style={{ fontSize: 12, color: '#fbbf24', marginTop: 8 }}>
                      {activeQuality.insufficient_evidence_reason}
                    </p>
                  )}
                </div>
              ) : (
                <p style={{ fontSize: 12, color: 'var(--rv-text-secondary)', marginBottom: 12 }}>
                  Track quality summary unavailable (historical inspection job).
                </p>
              )}

              {activeLoc && (
                <div className="rv-kv-list" style={{ marginBottom: 14 }}>
                  <div className="rv-kv-item">
                    <span className="rv-kv-label">Approx. Vehicle GPS ({activeLoc.provenance_label})</span>
                    <span className="rv-kv-value">
                      {activeLoc.lat !== null && activeLoc.lon !== null
                        ? `${activeLoc.lat.toFixed(6)}, ${activeLoc.lon.toFixed(6)}`
                        : 'Unavailable'}
                    </span>
                  </div>
                  <div className="rv-kv-item">
                    <span className="rv-kv-label">GPS Accuracy / Sync / Cluster</span>
                    <span className="rv-kv-value" style={{ display: 'flex', alignItems: 'center', gap: 6, flexWrap: 'wrap' }}>
                      <span>±{formatNullable(activeLoc.horizontal_accuracy_m, 1, ' m')} ({activeLoc.interpolation_status})</span>
                      {activeLoc.cluster && <MatchStatusBadge status={activeLoc.cluster.match_status} />}
                    </span>
                  </div>
                </div>
              )}

              {loadingObs ? (
                <StatePanel variant="loading" title="Loading track timeline" message="Fetching per-frame measurements…" />
              ) : obsError ? (
                <StatePanel variant="error" title="Failed to load track observations" message={obsError} />
              ) : trackObs.length === 0 ? (
                <StatePanel variant="empty" title="No observations" message="Select a track row on the left to inspect its measurements." />
              ) : (
                <div className="rv-bar-list">
                  {trackObs.slice(0, 12).map((obs, idx) => {
                    const fc = obs.frame_context || {};
                    const meas = obs.measurement || {};
                    const score = meas.smoothed_relative_score;
                    const widthPct =
                      score !== null && score !== undefined
                        ? Math.min(100, (Math.abs(score) / maxSmoothed) * 100)
                        : 0;
                    return (
                      <div key={`${fc.frame_idx}-${idx}`} className="rv-bar-row">
                        <span className="rv-bar-label rv-mono">
                          F{fc.frame_idx} · {meas.measurement_state}
                        </span>
                        <div className="rv-bar-track">
                          <div className="rv-bar-fill" style={{ width: `${widthPct}%` }} />
                        </div>
                        <span className="rv-bar-val">{formatNullable(score, 4)}</span>
                      </div>
                    );
                  })}
                </div>
              )}
            </section>
          </div>
          <ScientificDisclaimer />
        </>
      )}
    </div>
  );
}
