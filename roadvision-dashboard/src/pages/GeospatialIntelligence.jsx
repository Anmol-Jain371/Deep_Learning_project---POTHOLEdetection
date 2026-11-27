import React, { useEffect, useState, useRef, useCallback } from 'react';
import L from 'leaflet';
import 'leaflet/dist/leaflet.css';
import {
  uploadJobGps,
  getJobRoute,
  getJobLocations,
  runGeospatialMatch,
  listGeospatialClusters,
  getGeospatialCluster,
  reviewGeospatialCluster,
} from '../api/client';
import {
  MapPin,
  Upload,
  RefreshCw,
  Download,
  GitMerge,
  Check,
  X,
  AlertTriangle,
  Loader2,
} from 'lucide-react';
import InspectionSelector from '../components/InspectionSelector';
import {
  StatePanel,
  MatchStatusBadge,
  MeasurementStateBadge,
  QualityBadge,
  ScientificDisclaimer,
} from '../components/StateBanner';
import {
  MEASUREMENT_STATES,
  QUALITY_CATEGORIES,
  isValidUuid,
  formatNullable,
  downloadBlobFile,
} from '../utils/formatters';

function OfflineFallbackMap({ routePoints, locations, selectedLoc, onSelectLoc }) {
  const coords = [
    ...(routePoints || []).map((p) => ({ lat: p.lat, lon: p.lon })),
    ...(locations || []).filter((l) => l.lat !== null && l.lon !== null).map((l) => ({ lat: l.lat, lon: l.lon })),
  ];
  if (coords.length === 0) {
    return (
      <StatePanel
        variant="empty"
        title="No WGS84 coordinates to plot"
        message="No synchronized coordinates fall within telemetry bounds."
      />
    );
  }
  const lats = coords.map((c) => c.lat);
  const lons = coords.map((c) => c.lon);
  const minLat = Math.min(...lats);
  const maxLat = Math.max(...lats);
  const minLon = Math.min(...lons);
  const maxLon = Math.max(...lons);
  const dLat = Math.max(maxLat - minLat, 0.0004);
  const dLon = Math.max(maxLon - minLon, 0.0004);

  const W = 680;
  const H = 390;
  const PAD = 32;
  const proj = (lat, lon) => ({
    x: PAD + ((lon - minLon) / dLon) * (W - 2 * PAD),
    y: H - PAD - ((lat - minLat) / dLat) * (H - 2 * PAD),
  });

  const routeLine = (routePoints || [])
    .map((p) => {
      const pt = proj(p.lat, p.lon);
      return `${pt.x.toFixed(1)},${pt.y.toFixed(1)}`;
    })
    .join(' ');

  return (
    <div className="rv-map-frame" style={{ display: 'flex', flexDirection: 'column' }}>
      <div style={{ padding: '6px 12px', fontSize: 11, color: '#94a3b8', borderBottom: '1px solid var(--rv-border)' }}>
        Offline Vector Coordinate Projection (WGS84 local bounds — no internet basemap tiles required)
      </div>
      <svg viewBox={`0 0 ${W} ${H}`} style={{ width: '100%', flex: 1 }}>
        {routePoints.length >= 2 && (
          <polyline fill="none" stroke="#0891b2" strokeWidth="2.5" points={routeLine} />
        )}
        {(locations || [])
          .filter((l) => l.lat !== null && l.lon !== null)
          .map((loc) => {
            const pt = proj(loc.lat, loc.lon);
            const isSel = selectedLoc?.track_id === loc.track_id;
            return (
              <circle
                key={`${loc.track_id}-${loc.track_generation_id}`}
                cx={pt.x}
                cy={pt.y}
                r={isSel ? 8 : 6}
                fill={isSel ? '#f59e0b' : '#22d3ee'}
                stroke="#0f172a"
                strokeWidth="2"
                style={{ cursor: 'pointer' }}
                onClick={() => onSelectLoc(loc)}
              >
                <title>{`Track #${loc.track_id} (${loc.lat}, ${loc.lon})`}</title>
              </circle>
            );
          })}
      </svg>
    </div>
  );
}

export default function GeospatialIntelligence({ selectedJobId, onSelectJob, onNavigate }) {
  const [routeData, setRouteData] = useState(null);
  const [locData, setLocData] = useState(null);
  const [selectedLoc, setSelectedLoc] = useState(null);
  const [stateFilter, setStateFilter] = useState('');
  const [qualityFilter, setQualityFilter] = useState('');
  const [offlineMapMode, setOfflineMapMode] = useState(false);

  // Optional GPS upload form
  const [gpsFile, setGpsFile] = useState(null);
  const [timeMode, setTimeMode] = useState('auto');
  const [syncOffset, setSyncOffset] = useState('0.0');
  const [maxSyncGap, setMaxSyncGap] = useState('2.0');
  const [uploadingGps, setUploadingGps] = useState(false);
  const [gpsError, setGpsError] = useState(null);

  // Cross-trip clustering state
  const [clusters, setClusters] = useState([]);
  const [selectedCluster, setSelectedCluster] = useState(null);
  const [maxMatchDist, setMaxMatchDist] = useState('15');
  const [maxHorizAcc, setMaxHorizAcc] = useState('25');
  const [runningMatch, setRunningMatch] = useState(false);
  const [matchMsg, setMatchMsg] = useState(null);

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const mapContainerRef = useRef(null);
  const leafletMapRef = useRef(null);

  const loadGeoData = useCallback(async () => {
    if (!selectedJobId || !isValidUuid(selectedJobId)) {
      setRouteData(null);
      setLocData(null);
      return;
    }
    setLoading(true);
    setError(null);
    try {
      const [rResp, lResp, cResp] = await Promise.all([
        getJobRoute(selectedJobId),
        getJobLocations(selectedJobId, {
          ...(stateFilter ? { measurement_state: stateFilter } : {}),
          ...(qualityFilter ? { quality_category: qualityFilter } : {}),
        }),
        listGeospatialClusters({ page: 1, page_size: 50 }).catch(() => ({ clusters: [] })),
      ]);
      setRouteData(rResp);
      setLocData(lResp);
      setClusters(cResp?.clusters || []);
    } catch (err) {
      setError(err);
    } finally {
      setLoading(false);
    }
  }, [selectedJobId, stateFilter, qualityFilter]);

  useEffect(() => {
    loadGeoData();
  }, [loadGeoData]);

  // Initialize & update Leaflet map when online map mode is active
  useEffect(() => {
    if (offlineMapMode || !mapContainerRef.current || !routeData?.gps_available) {
      if (leafletMapRef.current) {
        leafletMapRef.current.remove();
        leafletMapRef.current = null;
      }
      return undefined;
    }

    const routePts = routeData.route_points || [];
    const locs = (locData?.locations || []).filter((l) => l.lat !== null && l.lon !== null);
    const allCoords = [
      ...routePts.map((p) => [p.lat, p.lon]),
      ...locs.map((l) => [l.lat, l.lon]),
    ];
    if (allCoords.length === 0) return undefined;

    if (leafletMapRef.current) {
      leafletMapRef.current.remove();
      leafletMapRef.current = null;
    }

    const map = L.map(mapContainerRef.current, { zoomControl: true });
    leafletMapRef.current = map;

    const tileLayer = L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
      attribution: '&copy; OpenStreetMap contributors',
      maxZoom: 19,
    });
    let tileErrors = 0;
    tileLayer.on('tileerror', () => {
      tileErrors += 1;
      if (tileErrors >= 3) {
        setOfflineMapMode(true);
      }
    });
    tileLayer.addTo(map);

    if (routePts.length >= 2) {
      L.polyline(
        routePts.map((p) => [p.lat, p.lon]),
        { color: '#06b6d4', weight: 3.5, opacity: 0.85 }
      ).addTo(map);
    }

    for (const loc of locs) {
      const marker = L.circleMarker([loc.lat, loc.lon], {
        radius: 7,
        color: '#0f172a',
        weight: 2,
        fillColor: selectedLoc?.track_id === loc.track_id ? '#f59e0b' : '#22d3ee',
        fillOpacity: 0.92,
      });
      marker.bindTooltip(
        `Track #${loc.track_id} · ${loc.interpolation_status} (${loc.provenance_label})`
      );
      marker.on('click', () => setSelectedLoc(loc));
      marker.addTo(map);
    }

    const bounds = L.latLngBounds(allCoords);
    map.fitBounds(bounds, { padding: [28, 28], maxZoom: 18 });

    return () => {
      if (leafletMapRef.current) {
        leafletMapRef.current.remove();
        leafletMapRef.current = null;
      }
    };
  }, [offlineMapMode, routeData, locData, selectedLoc]);

  const handleUploadGps = async (e) => {
    e.preventDefault();
    if (!gpsFile || !selectedJobId || uploadingGps) return;
    setUploadingGps(true);
    setGpsError(null);
    const fd = new FormData();
    fd.append('gps', gpsFile);
    fd.append('time_mode', timeMode);
    fd.append('sync_offset_sec', syncOffset);
    fd.append('max_sync_gap_sec', maxSyncGap);
    try {
      await uploadJobGps(selectedJobId, fd);
      setGpsFile(null);
      await loadGeoData();
    } catch (err) {
      setGpsError(err.message);
    } finally {
      setUploadingGps(false);
    }
  };

  const handleExportGeoJson = async () => {
    if (!selectedJobId) return;
    const gj = await getJobLocations(selectedJobId, { format: 'geojson' });
    const blob = new Blob([JSON.stringify(gj, null, 2)], { type: 'application/geo+json' });
    downloadBlobFile(blob, `roadvision_geo_${selectedJobId.slice(0, 8)}.geojson`);
  };

  const handleRunMatching = async () => {
    setRunningMatch(true);
    setMatchMsg(null);
    try {
      const res = await runGeospatialMatch({
        thresholds: {
          max_match_distance_m: Number(maxMatchDist) || 15,
          max_horizontal_accuracy_m: Number(maxHorizAcc) || 25,
        },
      });
      setMatchMsg(
        `Evaluated ${res.total_candidates} candidates → ${res.status_counts?.POTENTIAL_MATCH || 0} potential matches, ${res.status_counts?.REVIEW_REQUIRED || 0} review required, ${res.status_counts?.UNMATCHED || 0} unmatched.`
      );
      const cResp = await listGeospatialClusters({ page: 1, page_size: 50 });
      setClusters(cResp?.clusters || []);
    } catch (err) {
      setMatchMsg(`Matching failed: ${err.message}`);
    } finally {
      setRunningMatch(false);
    }
  };

  const handleInspectCluster = async (clusterId) => {
    try {
      const det = await getGeospatialCluster(clusterId);
      setSelectedCluster(det);
    } catch (err) {
      setMatchMsg(err.message);
    }
  };

  const handleReviewCluster = async (clusterId, reviewStatus) => {
    try {
      const updated = await reviewGeospatialCluster(clusterId, reviewStatus);
      setSelectedCluster(updated);
      setClusters((prev) =>
        prev.map((c) => (c.cluster_id === clusterId ? { ...c, review_status: updated.review_status } : c))
      );
    } catch (err) {
      setMatchMsg(err.message);
    }
  };

  return (
    <div className="rv-page">
      <div className="rv-page-header">
        <div>
          <h2 className="rv-page-title">Geospatial Intelligence &amp; Cross-Trip Matching</h2>
          <p className="rv-page-subtitle">
            Associate external GPS telemetry via video timestamps and evaluate conservative cross-trip candidate clusters.
          </p>
        </div>
        {routeData?.gps_available && (
          <div className="rv-header-actions">
            <button
              type="button"
              className="rv-btn rv-btn-ghost rv-btn-sm"
              onClick={() => setOfflineMapMode((m) => !m)}
            >
              {offlineMapMode ? 'Try Leaflet Tiles' : 'Offline Vector View'}
            </button>
            <button type="button" className="rv-btn rv-btn-secondary rv-btn-sm" onClick={handleExportGeoJson}>
              <Download size={14} /> Export GeoJSON
            </button>
          </div>
        )}
      </div>

      <InspectionSelector
        selectedJobId={isValidUuid(selectedJobId) ? selectedJobId : ''}
        onSelectJob={onSelectJob}
        onNavigate={onNavigate}
      />

      {selectedJobId && (
        <section className="rv-card">
          <div className="rv-card-header">
            <h3 className="rv-card-title"><Upload size={16} className="rv-icon-cyan" /> External GPS Telemetry Upload</h3>
            <span className="rv-card-meta">CSV, GPX, or GeoJSON · Approximate vehicle/camera path only</span>
          </div>
          <form onSubmit={handleUploadGps} className="rv-filter-bar" style={{ marginBottom: 0 }}>
            <div className="rv-field" style={{ flex: 1, minWidth: 220 }}>
              <label className="rv-field-label" htmlFor="rv-gps-file">Telemetry File</label>
              <input
                id="rv-gps-file"
                type="file"
                accept=".csv,.gpx,.geojson,.json"
                className="rv-input"
                onChange={(e) => setGpsFile(e.target.files?.[0] || null)}
              />
            </div>
            <div className="rv-field">
              <label className="rv-field-label" htmlFor="rv-gps-tmode">Timestamp Mode</label>
              <select
                id="rv-gps-tmode"
                className="rv-select"
                value={timeMode}
                disabled={!gpsFile}
                onChange={(e) => setTimeMode(e.target.value)}
              >
                <option value="auto">Auto (Numeric or ISO-8601 TZ)</option>
                <option value="video_relative">Video-Relative Seconds</option>
                <option value="iso8601">ISO-8601 with Timezone</option>
              </select>
            </div>
            <div className="rv-field">
              <label className="rv-field-label" htmlFor="rv-gps-off">Sync Offset (s)</label>
              <input
                id="rv-gps-off"
                type="number"
                step="0.1"
                className="rv-input"
                style={{ width: 105 }}
                value={syncOffset}
                disabled={!gpsFile}
                onChange={(e) => setSyncOffset(e.target.value)}
              />
            </div>
            <div className="rv-field">
              <label className="rv-field-label" htmlFor="rv-gps-gap">Max Sync Gap (s)</label>
              <input
                id="rv-gps-gap"
                type="number"
                step="0.1"
                min="0.1"
                max="60"
                className="rv-input"
                style={{ width: 115 }}
                value={maxSyncGap}
                disabled={!gpsFile}
                onChange={(e) => setMaxSyncGap(e.target.value)}
              />
            </div>
            <button type="submit" className="rv-btn rv-btn-primary rv-btn-sm" disabled={!gpsFile || uploadingGps}>
              {uploadingGps ? <Loader2 size={14} className="rv-spin" /> : <Upload size={14} />}
              <span>{uploadingGps ? 'Synchronizing…' : 'Upload & Sync GPS'}</span>
            </button>
          </form>
          {gpsError && <p style={{ fontSize: 12, color: '#fb7185', marginTop: 8 }}>{gpsError}</p>}
        </section>
      )}

      {!selectedJobId ? (
        <StatePanel variant="empty" title="Select an inspection" message="Choose an inspection above to inspect or attach GPS telemetry." />
      ) : loading && !routeData ? (
        <StatePanel variant="loading" title="Loading geospatial state" message="Querying route and track synchronization…" />
      ) : error ? (
        <StatePanel variant="error" title="Geospatial query failed" message={error.message} />
      ) : !routeData?.gps_available ? (
        <StatePanel
          variant="empty"
          title="No genuine GPS telemetry available for this inspection"
          message="RoadVision AI 2.0 never fabricates GPS coordinates from video. Upload a CSV (timestamp, latitude, longitude, horizontal_accuracy), GPX, or timestamped GeoJSON file above to synchronize approximate vehicle observation locations."
        />
      ) : (
        <>
          <div className="rv-kpi-grid">
            <div className="rv-kpi-card">
              <div className="rv-kpi-top"><span>GPS Source</span><MapPin size={15} className="rv-icon-cyan" /></div>
              <div className="rv-kpi-val" style={{ fontSize: 15 }}>{routeData.metadata?.original_filename}</div>
              <div className="rv-kpi-sub">{routeData.metadata?.source_format?.toUpperCase()} · {routeData.metadata?.time_mode}</div>
            </div>
            <div className="rv-kpi-card">
              <div className="rv-kpi-top"><span>Route Samples</span></div>
              <div className="rv-kpi-val">{routeData.metadata?.sample_count ?? 0}</div>
              <div className="rv-kpi-sub">Offset: {routeData.metadata?.sync_offset_sec ?? 0}s</div>
            </div>
            <div className="rv-kpi-card">
              <div className="rv-kpi-top"><span>Synced Tracks</span></div>
              <div className="rv-kpi-val">{locData?.synchronized_count ?? 0}</div>
              <div className="rv-kpi-sub">Unavailable: {locData?.unavailable_count ?? 0}</div>
            </div>
          </div>

          <section className="rv-card">
            <div className="rv-card-header">
              <h3 className="rv-card-title">Inspection Route &amp; Approximate Observation Locations</h3>
              <div className="rv-filter-bar" style={{ marginBottom: 0 }}>
                <select className="rv-select" value={stateFilter} onChange={(e) => setStateFilter(e.target.value)}>
                  <option value="">All Measurement States</option>
                  {MEASUREMENT_STATES.map((s) => <option key={s} value={s}>{s}</option>)}
                </select>
                <select className="rv-select" value={qualityFilter} onChange={(e) => setQualityFilter(e.target.value)}>
                  <option value="">All Quality Categories</option>
                  {QUALITY_CATEGORIES.map((q) => <option key={q} value={q}>{q}</option>)}
                </select>
              </div>
            </div>

            {offlineMapMode ? (
              <OfflineFallbackMap
                routePoints={routeData.route_points || []}
                locations={locData?.locations || []}
                selectedLoc={selectedLoc}
                onSelectLoc={setSelectedLoc}
              />
            ) : (
              <div ref={mapContainerRef} className="rv-map-frame" />
            )}

            {selectedLoc && (
              <div className="rv-kv-list" style={{ marginTop: 12 }}>
                <div className="rv-kv-item">
                  <span className="rv-kv-label">Selected Track</span>
                  <span className="rv-kv-value">Track #{selectedLoc.track_id} (ts {selectedLoc.observation_timestamp_sec}s)</span>
                </div>
                <div className="rv-kv-item">
                  <span className="rv-kv-label">Approx. Vehicle Position (WGS84)</span>
                  <span className="rv-kv-value">{formatNullable(selectedLoc.lat, 6)}, {formatNullable(selectedLoc.lon, 6)}</span>
                </div>
                <div className="rv-kv-item">
                  <span className="rv-kv-label">Sync &amp; Provenance</span>
                  <span className="rv-kv-value">
                    {selectedLoc.interpolation_status} ({selectedLoc.sync_quality}, Δt={formatNullable(selectedLoc.time_diff_sec, 2, 's')}) · {selectedLoc.provenance_label}
                  </span>
                </div>
                <div className="rv-kv-item">
                  <span className="rv-kv-label">Accuracy / Quality</span>
                  <span className="rv-kv-value">
                    ±{formatNullable(selectedLoc.horizontal_accuracy_m, 1, ' m')} · {selectedLoc.quality_category || 'Unavailable'}
                  </span>
                </div>
              </div>
            )}
          </section>

          <section className="rv-card">
            <div className="rv-card-header">
              <h3 className="rv-card-title">Synchronized Track Locations ({locData?.locations?.length || 0})</h3>
            </div>
            {(locData?.locations?.length || 0) === 0 ? (
              <StatePanel variant="empty" title="No track locations match current filters" message="Complete the inspection or adjust filters." />
            ) : (
              <div className="rv-table-wrap">
                <table className="rv-table">
                  <thead>
                    <tr>
                      <th>Track</th>
                      <th>Obs. Time</th>
                      <th>Lat, Lon (Approx. Vehicle)</th>
                      <th>Sync Status</th>
                      <th>Accuracy</th>
                      <th>State</th>
                      <th>Quality</th>
                      <th>Cluster</th>
                    </tr>
                  </thead>
                  <tbody>
                    {locData.locations.map((loc) => (
                      <tr
                        key={`${loc.track_id}-${loc.track_generation_id}`}
                        className={`rv-row-clickable ${selectedLoc?.track_id === loc.track_id ? 'rv-row-selected' : ''}`}
                        onClick={() => setSelectedLoc(loc)}
                      >
                        <td className="rv-mono">#{loc.track_id}</td>
                        <td className="rv-mono">{formatNullable(loc.observation_timestamp_sec, 2, 's')}</td>
                        <td className="rv-mono">
                          {loc.lat !== null && loc.lon !== null
                            ? `${loc.lat.toFixed(6)}, ${loc.lon.toFixed(6)}`
                            : 'Unavailable'}
                        </td>
                        <td className="rv-mono">{loc.interpolation_status} ({loc.sync_quality})</td>
                        <td className="rv-mono">{formatNullable(loc.horizontal_accuracy_m, 1, ' m')}</td>
                        <td><MeasurementStateBadge state={loc.measurement_state} /></td>
                        <td><QualityBadge category={loc.quality_category} /></td>
                        <td>{loc.cluster ? <MatchStatusBadge status={loc.cluster.match_status} /> : '—'}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </section>
        </>
      )}

      <section className="rv-card">
        <div className="rv-card-header">
          <h3 className="rv-card-title"><GitMerge size={16} className="rv-icon-cyan" /> Cross-Trip Candidate Deduplication Clusters</h3>
          <div className="rv-header-actions">
            <button type="button" className="rv-btn rv-btn-primary rv-btn-sm" onClick={handleRunMatching} disabled={runningMatch}>
              {runningMatch ? <Loader2 size={14} className="rv-spin" /> : <RefreshCw size={14} />}
              <span>Run Cross-Trip Matching</span>
            </button>
          </div>
        </div>

        <div className="rv-filter-bar">
          <div className="rv-field">
            <label className="rv-field-label" htmlFor="rv-match-dist">Max Match Distance (m)</label>
            <input
              id="rv-match-dist"
              type="number"
              className="rv-input"
              style={{ width: 140 }}
              value={maxMatchDist}
              onChange={(e) => setMaxMatchDist(e.target.value)}
            />
          </div>
          <div className="rv-field">
            <label className="rv-field-label" htmlFor="rv-match-acc">Max Horizontal Accuracy (m)</label>
            <input
              id="rv-match-acc"
              type="number"
              className="rv-input"
              style={{ width: 150 }}
              value={maxHorizAcc}
              onChange={(e) => setMaxHorizAcc(e.target.value)}
            />
          </div>
        </div>

        {matchMsg && <p style={{ fontSize: 12, color: 'var(--rv-cyan-light)', marginBottom: 10 }}>{matchMsg}</p>}

        {clusters.length === 0 ? (
          <StatePanel
            variant="empty"
            title="No geospatial candidate clusters computed"
            message="Upload GPS telemetry to two or more inspections and run Cross-Trip Matching to evaluate potential proximity candidates."
          />
        ) : (
          <div className="rv-table-wrap">
            <table className="rv-table">
              <thead>
                <tr>
                  <th>Cluster</th>
                  <th>Match Status</th>
                  <th>Review</th>
                  <th>Members (Trips)</th>
                  <th>Diameter (m)</th>
                  <th>Centroid (WGS84)</th>
                  <th>Reasons / Flags</th>
                </tr>
              </thead>
              <tbody>
                {clusters.map((cl) => (
                  <tr
                    key={cl.cluster_id}
                    className={`rv-row-clickable ${selectedCluster?.cluster_id === cl.cluster_id ? 'rv-row-selected' : ''}`}
                    onClick={() => handleInspectCluster(cl.cluster_id)}
                  >
                    <td className="rv-mono">
                      {cl.cluster_id.slice(0, 8)}
                      {cl.contains_synthetic && (
                        <span className="rv-badge rv-badge-warning" style={{ marginLeft: 6 }}>SYNTHETIC</span>
                      )}
                    </td>
                    <td><MatchStatusBadge status={cl.match_status} /></td>
                    <td className="rv-mono">{cl.review_status}</td>
                    <td className="rv-mono">{cl.member_count} ({cl.distinct_trip_count} trips)</td>
                    <td className="rv-mono">{formatNullable(cl.diameter_m, 2, ' m')}</td>
                    <td className="rv-mono">
                      {cl.centroid_lat !== null && cl.centroid_lon !== null
                        ? `${cl.centroid_lat.toFixed(6)}, ${cl.centroid_lon.toFixed(6)}`
                        : 'Unavailable'}
                    </td>
                    <td style={{ fontSize: 12, color: 'var(--rv-text-secondary)' }}>
                      {(cl.match_reasons || []).join(', ')}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}

        {selectedCluster && (
          <div className="rv-card" style={{ marginTop: 14, background: 'var(--rv-bg-surface)' }}>
            <div className="rv-card-header">
              <h4 className="rv-card-title">
                Cluster {selectedCluster.cluster_id.slice(0, 8)} Details ({selectedCluster.member_count} tracks retained)
              </h4>
              <div className="rv-header-actions">
                <button
                  type="button"
                  className="rv-btn rv-btn-secondary rv-btn-sm"
                  onClick={() => handleReviewCluster(selectedCluster.cluster_id, 'CONFIRMED_REVIEW')}
                >
                  <Check size={14} /> Confirm Review
                </button>
                <button
                  type="button"
                  className="rv-btn rv-btn-ghost rv-btn-sm"
                  onClick={() => handleReviewCluster(selectedCluster.cluster_id, 'DISMISSED')}
                >
                  <X size={14} /> Dismiss Match
                </button>
              </div>
            </div>
            <div className="rv-table-wrap">
              <table className="rv-table">
                <thead>
                  <tr>
                    <th>Job ID</th>
                    <th>Track</th>
                    <th>Distance to Centroid</th>
                    <th>Accuracy</th>
                    <th>Quality</th>
                  </tr>
                </thead>
                <tbody>
                  {(selectedCluster.members || []).map((m, i) => (
                    <tr key={`${m.job_id}-${m.track_id}-${i}`}>
                      <td className="rv-mono">{m.job_id.slice(0, 8)}…</td>
                      <td className="rv-mono">#{m.track_id}</td>
                      <td className="rv-mono">{formatNullable(m.distance_to_centroid_m, 2, ' m')}</td>
                      <td className="rv-mono">{formatNullable(m.horizontal_accuracy_m, 1, ' m')}</td>
                      <td><QualityBadge category={m.quality_category} /></td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}
      </section>

      <div className="rv-state-panel rv-state-api-disabled" role="note">
        <AlertTriangle size={16} className="rv-icon-warning" />
        <div className="rv-state-body">
          <p className="rv-state-message">
            Locations shown here reflect approximate vehicle/camera GPS positions synchronized by video timestamp (not surveyed road-surface coordinates). Candidate clusters indicate proximity only and are not independently verified unique physical potholes.
          </p>
        </div>
      </div>

      <ScientificDisclaimer />
    </div>
  );
}
