import React, { useEffect, useState, useCallback } from 'react';
import {
  getJob,
  getSummary,
  getArtifacts,
  downloadArtifact,
  getJobRoute,
  getJobLocations,
  getJobMaintenance,
  downloadJobReport,
  getJobReportUrl,
} from '../api/client';
import { Download, FileBarChart2, Printer, FileJson, MapPin, FileSpreadsheet, FileCode2 } from 'lucide-react';
import InspectionSelector from '../components/InspectionSelector';
import {
  StatePanel,
  StatusBadge,
  ScientificDisclaimer,
  MeasurementStateDistribution,
} from '../components/StateBanner';
import {
  isValidUuid,
  formatNullable,
  formatRatePercent,
  formatBytes,
  formatSeconds,
  formatDateTime,
  downloadBlobFile,
} from '../utils/formatters';

export default function Reports({ selectedJobId, onSelectJob, onNavigate }) {
  const [job, setJob] = useState(null);
  const [summaryData, setSummaryData] = useState(null);
  const [artifacts, setArtifacts] = useState([]);
  const [partial, setPartial] = useState(false);
  const [routeData, setRouteData] = useState(null);
  const [locData, setLocData] = useState(null);
  const [maintData, setMaintData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [downloadError, setDownloadError] = useState(null);

  const loadReport = useCallback(async () => {
    if (!selectedJobId || !isValidUuid(selectedJobId)) return;
    setLoading(true);
    setError(null);
    setDownloadError(null);
    try {
      const j = await getJob(selectedJobId);
      setJob(j);
      const [artResp, rResp, lResp] = await Promise.all([
        getArtifacts(selectedJobId).catch(() => null),
        getJobRoute(selectedJobId).catch(() => null),
        getJobLocations(selectedJobId).catch(() => null),
      ]);
      setArtifacts(artResp?.artifacts || []);
      setPartial(Boolean(artResp?.partial));
      setRouteData(rResp);
      setLocData(lResp);

      if (j.status === 'COMPLETED') {
        const [s, m] = await Promise.all([
          getSummary(selectedJobId),
          getJobMaintenance(selectedJobId).catch(() => null),
        ]);
        setSummaryData(s);
        setMaintData(m);
      } else {
        setSummaryData(null);
        setMaintData(null);
      }
    } catch (err) {
      setJob(null);
      setSummaryData(null);
      setMaintData(null);
      setArtifacts([]);
      setRouteData(null);
      setLocData(null);
      setError(err);
    } finally {
      setLoading(false);
    }
  }, [selectedJobId]);

  useEffect(() => {
    loadReport();
  }, [loadReport]);

  const handleDownloadArtifact = async (artifactId, filename) => {
    setDownloadError(null);
    try {
      const blob = await downloadArtifact(selectedJobId, artifactId);
      downloadBlobFile(blob, filename || artifactId);
    } catch (err) {
      setDownloadError(err.message);
    }
  };

  const handleExportServerReport = async (format, ext) => {
    if (!selectedJobId) return;
    setDownloadError(null);
    try {
      const blob = await downloadJobReport(selectedJobId, format);
      downloadBlobFile(blob, `roadvision_report_${selectedJobId.slice(0, 8)}.${ext}`);
    } catch (err) {
      setDownloadError(err.message);
    }
  };

  const handleExportCompiledJson = () => {
    if (job?.status === 'COMPLETED') {
      handleExportServerReport('json', 'json');
      return;
    }
    if (!job) return;
    const payload = {
      exported_at: new Date().toISOString(),
      job,
      summary: summaryData,
      artifacts,
      geospatial_route_metadata: routeData?.metadata || null,
      geospatial_locations: locData?.locations || [],
    };
    const blob = new Blob([JSON.stringify(payload, null, 2)], { type: 'application/json' });
    downloadBlobFile(blob, `roadvision_report_${job.job_id.slice(0, 8)}.json`);
  };

  const handleExportGeoJson = async () => {
    if (!selectedJobId) return;
    const gj = await getJobLocations(selectedJobId, { format: 'geojson' });
    const blob = new Blob([JSON.stringify(gj, null, 2)], { type: 'application/geo+json' });
    downloadBlobFile(blob, `roadvision_geo_${selectedJobId.slice(0, 8)}.geojson`);
  };

  const sessionSummary = summaryData?.session_summary || {};
  const coverage = summaryData?.measurement_coverage || job?.measurement_coverage || {};
  const qualitySummary = summaryData?.quality_summary || sessionSummary?.quality_summary || null;

  return (
    <div className="rv-page">
      <div className="rv-page-header">
        <div>
          <h2 className="rv-page-title">Inspection Report Centre</h2>
          <p className="rv-page-subtitle">
            Review verified inspection metadata, heuristic measurement quality, maintenance reviews, and download structured reports.
          </p>
        </div>
        {job && (
          <div className="rv-header-actions">
            {job.status === 'COMPLETED' && (
              <>
                <button type="button" className="rv-btn rv-btn-primary rv-btn-sm" onClick={() => handleExportServerReport('html', 'html')}>
                  <FileCode2 size={14} /> Download HTML Report
                </button>
                <button type="button" className="rv-btn rv-btn-secondary rv-btn-sm" onClick={() => handleExportServerReport('csv', 'csv')}>
                  <FileSpreadsheet size={14} /> Export CSV Summary
                </button>
              </>
            )}
            <button type="button" className="rv-btn rv-btn-secondary rv-btn-sm" onClick={handleExportCompiledJson}>
              <FileJson size={14} /> Export JSON Report
            </button>
            {routeData?.gps_available && (
              <button type="button" className="rv-btn rv-btn-secondary rv-btn-sm" onClick={handleExportGeoJson}>
                <MapPin size={14} /> Export GeoJSON
              </button>
            )}
            {job.status === 'COMPLETED' ? (
              <a
                href={getJobReportUrl(job.job_id, 'html')}
                target="_blank"
                rel="noopener noreferrer"
                className="rv-btn rv-btn-ghost rv-btn-sm"
              >
                <Printer size={14} /> Printable HTML (PDF)
              </a>
            ) : (
              <button type="button" className="rv-btn rv-btn-ghost rv-btn-sm" onClick={() => window.print()}>
                <Printer size={14} /> Print View
              </button>
            )}
          </div>
        )}
      </div>

      <InspectionSelector
        selectedJobId={isValidUuid(selectedJobId) ? selectedJobId : ''}
        onSelectJob={onSelectJob}
        onNavigate={onNavigate}
        label="Inspection Report"
      />

      {!selectedJobId ? (
        <StatePanel variant="empty" title="No inspection selected" message="Choose an inspection above to view its report and artifacts." />
      ) : loading ? (
        <StatePanel variant="loading" title="Building inspection report" message="Loading summary and verified artifact manifest…" />
      ) : error ? (
        <StatePanel
          variant={error.isNetworkError ? 'disconnected' : 'error'}
          title="Unable to load report"
          message={error.message}
        />
      ) : job && (
        <>
          <section className="rv-card">
            <div className="rv-card-header">
              <h3 className="rv-card-title">
                <FileBarChart2 size={16} className="rv-icon-cyan" /> Inspection Metadata
              </h3>
              <StatusBadge status={job.status} />
            </div>

            <div className="rv-kv-list">
              <div className="rv-kv-item">
                <span className="rv-kv-label">Source File</span>
                <span className="rv-kv-value">{job.original_filename || 'Unavailable'}</span>
              </div>
              <div className="rv-kv-item">
                <span className="rv-kv-label">Job UUID</span>
                <span className="rv-kv-value">{job.job_id}</span>
              </div>
              <div className="rv-kv-item">
                <span className="rv-kv-label">Mode / Pipeline</span>
                <span className="rv-kv-value">{(job.mode || '').toUpperCase()} · v{job.pipeline_version || '2.0'}</span>
              </div>
              <div className="rv-kv-item">
                <span className="rv-kv-label">Created / Completed</span>
                <span className="rv-kv-value">
                  {formatDateTime(job.created_at)} → {formatDateTime(job.completed_at)}
                </span>
              </div>
              <div className="rv-kv-item">
                <span className="rv-kv-label">Frames Processed</span>
                <span className="rv-kv-value">{job.frames_processed ?? 0} / {formatNullable(job.total_frames)}</span>
              </div>
              <div className="rv-kv-item">
                <span className="rv-kv-label">Sustained FPS</span>
                <span className="rv-kv-value">{formatNullable(job.throughput_fps, 2)}</span>
              </div>
            </div>
          </section>

          {job.status !== 'COMPLETED' ? (
            <StatePanel
              variant="unavailable"
              title={`Full summary unavailable (${job.status})`}
              message="Session summary metrics are only finalized when a job reaches COMPLETED status. Any partial artifacts generated before cancellation or failure can be downloaded below."
            />
          ) : (
            <>
              <section className="rv-card">
                <div className="rv-card-header">
                  <h3 className="rv-card-title">Measurement State, Coverage &amp; Quality Summary</h3>
                  <span className="rv-card-meta">
                    {coverage?.denominator_total_observations ?? 0} observations · {sessionSummary.unique_tracks ?? 0} session tracks
                  </span>
                </div>

                <MeasurementStateDistribution
                  countsByState={coverage?.counts_by_state}
                  totalObservations={coverage?.denominator_total_observations ?? 0}
                />

                <div className="rv-kv-list" style={{ marginTop: 14 }}>
                  <div className="rv-kv-item">
                    <span className="rv-kv-label">Fresh Valid Rate (All Obs.)</span>
                    <span className="rv-kv-value">{formatRatePercent(coverage?.fresh_valid_rate_over_observations)}</span>
                  </div>
                  <div className="rv-kv-item">
                    <span className="rv-kv-label">Fresh Valid Rate (Keyframes)</span>
                    <span className="rv-kv-value">{formatRatePercent(coverage?.fresh_valid_rate_over_keyframes)}</span>
                  </div>
                  <div className="rv-kv-item">
                    <span className="rv-kv-label">Cached Valid Rate</span>
                    <span className="rv-kv-value">{formatRatePercent(coverage?.cached_valid_rate_over_observations)}</span>
                  </div>
                  <div className="rv-kv-item">
                    <span className="rv-kv-label">Median Fresh Quality (Heuristic)</span>
                    <span className="rv-kv-value">{formatNullable(qualitySummary?.median_fresh_quality_index, 3)}</span>
                  </div>
                  <div className="rv-kv-item">
                    <span className="rv-kv-label">Depth Model</span>
                    <span className="rv-kv-value">
                      {sessionSummary.depth_enabled
                        ? `${sessionSummary.depth_model_id} (${sessionSummary.depth_input_size}px)`
                        : 'Depth disabled'}
                    </span>
                  </div>
                  <div className="rv-kv-item">
                    <span className="rv-kv-label">Elapsed / Model Init</span>
                    <span className="rv-kv-value">
                      {formatSeconds(sessionSummary.elapsed_sec)} / {formatSeconds(sessionSummary.model_init_sec)}
                    </span>
                  </div>
                </div>

                {qualitySummary?.top_rejection_reasons?.length > 0 && (
                  <div style={{ marginTop: 14 }}>
                    <h4 className="rv-card-title" style={{ fontSize: 13, marginBottom: 8 }}>Top Explainable Diagnostics</h4>
                    <div className="rv-kv-list">
                      {qualitySummary.top_rejection_reasons.slice(0, 4).map((rr) => (
                        <div key={rr.code} className="rv-kv-item">
                          <span className="rv-kv-label">{rr.title} ({rr.count}×)</span>
                          <span style={{ fontSize: 12, color: 'var(--rv-text-secondary)' }}>{rr.explanation}</span>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </section>

              <section className="rv-card">
                <div className="rv-card-header">
                  <h3 className="rv-card-title"><MapPin size={16} className="rv-icon-cyan" /> Geospatial Availability &amp; Route Summary</h3>
                </div>
                {routeData?.gps_available ? (
                  <div className="rv-kv-list">
                    <div className="rv-kv-item">
                      <span className="rv-kv-label">GPS Source</span>
                      <span className="rv-kv-value">
                        {routeData.metadata?.original_filename} ({routeData.metadata?.source_format?.toUpperCase()})
                      </span>
                    </div>
                    <div className="rv-kv-item">
                      <span className="rv-kv-label">Route Samples</span>
                      <span className="rv-kv-value">{routeData.metadata?.sample_count ?? 0} points</span>
                    </div>
                    <div className="rv-kv-item">
                      <span className="rv-kv-label">Synchronized Tracks</span>
                      <span className="rv-kv-value">
                        {locData?.synchronized_count ?? 0} synced · {locData?.unavailable_count ?? 0} unavailable
                      </span>
                    </div>
                    <div className="rv-kv-item">
                      <span className="rv-kv-label">Location Provenance</span>
                      <span className="rv-kv-value">Approximate vehicle/camera GPS (NOT surveyed surface coordinates)</span>
                    </div>
                  </div>
                ) : (
                  <p style={{ fontSize: 13, color: 'var(--rv-text-secondary)' }}>
                    No genuine GPS telemetry was uploaded for this inspection. Approximate route and candidate coordinates are unavailable.
                  </p>
                )}
              </section>

              {maintData && (
                <section className="rv-card">
                  <div className="rv-card-header">
                    <h3 className="rv-card-title">Maintenance Priority &amp; Manual Review Summary</h3>
                    <span className="rv-card-meta">Scoring {maintData.scoring_version}</span>
                  </div>
                  <div className="rv-kv-list">
                    <div className="rv-kv-item">
                      <span className="rv-kv-label">Priority Review / Manual Verification</span>
                      <span className="rv-kv-value rv-mono">
                        {maintData.priority_distribution?.PRIORITY_REVIEW ?? 0} / {maintData.priority_distribution?.MANUAL_VERIFICATION_REQUIRED ?? 0}
                      </span>
                    </div>
                    <div className="rv-kv-item">
                      <span className="rv-kv-label">Standard / Insufficient / Reviewed</span>
                      <span className="rv-kv-value rv-mono">
                        {maintData.priority_distribution?.STANDARD_REVIEW ?? 0} / {maintData.priority_distribution?.INSUFFICIENT_EVIDENCE ?? 0} / {maintData.priority_distribution?.REVIEWED ?? 0}
                      </span>
                    </div>
                    <div className="rv-kv-item">
                      <span className="rv-kv-label">Confirmed Visually / Rejected FP / Needs Field</span>
                      <span className="rv-kv-value rv-mono">
                        {maintData.review_status_distribution?.CONFIRMED_VISUALLY ?? 0} / {maintData.review_status_distribution?.REJECTED_FALSE_POSITIVE ?? 0} / {maintData.review_status_distribution?.NEEDS_FIELD_INSPECTION ?? 0}
                      </span>
                    </div>
                    <div className="rv-kv-item">
                      <span className="rv-kv-label">Pending / In Review</span>
                      <span className="rv-kv-value rv-mono">
                        {maintData.review_status_distribution?.PENDING_REVIEW ?? 0} / {maintData.review_status_distribution?.IN_REVIEW ?? 0}
                      </span>
                    </div>
                  </div>
                </section>
              )}
            </>
          )}

          <section className="rv-card">
            <div className="rv-card-header">
              <h3 className="rv-card-title">Verified Artifacts</h3>
              {partial && <span className="rv-badge rv-badge-warning">Partial Output</span>}
            </div>

            {downloadError && (
              <div className="rv-state-panel rv-state-error" style={{ marginBottom: 12 }}>
                <div className="rv-state-body"><p className="rv-state-message">{downloadError}</p></div>
              </div>
            )}

            {artifacts.length === 0 ? (
              <StatePanel variant="empty" title="No artifacts published" message="No downloadable files exist for this inspection." />
            ) : (
              <div className="rv-artifact-list">
                {artifacts.map((art) => (
                  <div key={art.artifact_id} className="rv-artifact-row">
                    <div className="rv-artifact-info">
                      <span className="rv-artifact-name">{art.filename}</span>
                      <span className="rv-artifact-meta">
                        {art.artifact_id} · {art.mime_type} · {formatBytes(art.size_bytes)} · Created {formatDateTime(art.created_at)}
                      </span>
                    </div>
                    <button
                      type="button"
                      className="rv-btn rv-btn-secondary rv-btn-sm"
                      onClick={() => handleDownloadArtifact(art.artifact_id, art.filename)}
                    >
                      <Download size={14} /> Download
                    </button>
                  </div>
                ))}
              </div>
            )}
          </section>

          <ScientificDisclaimer text={summaryData?.scientific_disclaimer || job.scientific_disclaimer} />
        </>
      )}
    </div>
  );
}
