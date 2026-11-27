import React, { useEffect, useState } from 'react';
import { createJob, getModes, getCapabilities, uploadJobGps } from '../api/client';
import { Upload, Play, FileVideo, AlertTriangle, Loader2, CheckCircle2, MapPin } from 'lucide-react';
import { StatePanel, ScientificDisclaimer } from '../components/StateBanner';
import { formatBytes } from '../utils/formatters';

export default function NewInspection({ onNavigate, initialPreset = null }) {
  const [modesData, setModesData] = useState(null);
  const [capabilities, setCapabilities] = useState(null);
  const [loadingConfig, setLoadingConfig] = useState(true);
  const [configError, setConfigError] = useState(null);

  const [file, setFile] = useState(null);
  const [mode, setMode] = useState('balanced');
  const [enableDepth, setEnableDepth] = useState(
    initialPreset?.enableDepth !== undefined ? Boolean(initialPreset.enableDepth) : true
  );
  const [enableGeometry, setEnableGeometry] = useState(
    initialPreset?.enableGeometry !== undefined ? Boolean(initialPreset.enableGeometry) : false
  );
  const [diagnostics, setDiagnostics] = useState(false);
  const [saveVideo, setSaveVideo] = useState(true);
  const [saveJson, setSaveJson] = useState(true);
  const [saveCsv, setSaveCsv] = useState(true);
  const [maxFrames, setMaxFrames] = useState('');

  // Optional Phase 11 GPS telemetry
  const [gpsFile, setGpsFile] = useState(null);
  const [gpsTimeMode, setGpsTimeMode] = useState('auto');
  const [gpsSyncOffset, setGpsSyncOffset] = useState('0.0');
  const [gpsMaxGap, setGpsMaxGap] = useState('2.0');

  const [submitting, setSubmitting] = useState(false);
  const [submitError, setSubmitError] = useState(null);

  useEffect(() => {
    let mounted = true;
    async function load() {
      setLoadingConfig(true);
      setConfigError(null);
      try {
        const [mResp, cResp] = await Promise.all([getModes(), getCapabilities()]);
        if (!mounted) return;
        setModesData(mResp?.modes || {});
        setCapabilities(cResp);
        const defaultDepth = mResp?.modes?.balanced?.depth?.enabled;
        if (initialPreset?.enableDepth === undefined && typeof defaultDepth === 'boolean') {
          setEnableDepth(defaultDepth);
        }
      } catch (err) {
        if (mounted) setConfigError(err);
      } finally {
        if (mounted) setLoadingConfig(false);
      }
    }
    load();
    return () => {
      mounted = false;
    };
  }, [initialPreset?.enableDepth]);

  useEffect(() => {
    if (!initialPreset) return;
    if (initialPreset.enableDepth !== undefined) {
      setEnableDepth(Boolean(initialPreset.enableDepth));
    }
    if (initialPreset.enableGeometry !== undefined) {
      setEnableGeometry(Boolean(initialPreset.enableGeometry));
    }
  }, [initialPreset]);

  const handleModeSelect = (selectedMode) => {
    setMode(selectedMode);
    const modeDefaultDepth = modesData?.[selectedMode]?.depth?.enabled;
    if (typeof modeDefaultDepth === 'boolean') {
      setEnableDepth(modeDefaultDepth);
    }
  };

  const limits = capabilities?.limits || {};
  const allowedFormats = limits.allowed_video_formats || ['.mp4', '.avi', '.mov', '.mkv'];
  const maxUploadBytes = limits.maximum_upload_size_bytes || 268435456;
  const maxProcessedFrames = limits.maximum_processed_frames || 9000;

  const validateSelectedFile = (candidate) => {
    if (!candidate) return null;
    const lowerName = candidate.name.toLowerCase();
    const validExt = allowedFormats.some((ext) => lowerName.endsWith(ext.toLowerCase()));
    if (!validExt) {
      return `Unsupported video extension. Allowed formats: ${allowedFormats.join(', ')}`;
    }
    if (candidate.size <= 0) {
      return 'Selected file is empty (0 bytes).';
    }
    if (candidate.size > maxUploadBytes) {
      return `File size (${formatBytes(candidate.size)}) exceeds the ${formatBytes(maxUploadBytes)} upload limit.`;
    }
    return null;
  };

  const handleFileChange = (e) => {
    const picked = e.target.files?.[0] || null;
    setSubmitError(null);
    if (!picked) {
      setFile(null);
      return;
    }
    const validationMsg = validateSelectedFile(picked);
    if (validationMsg) {
      setFile(null);
      setSubmitError(validationMsg);
      return;
    }
    setFile(picked);
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!file || submitting) return;

    const fileErr = validateSelectedFile(file);
    if (fileErr) {
      setSubmitError(fileErr);
      return;
    }

    if (maxFrames.trim() !== '') {
      const parsed = Number(maxFrames);
      if (!Number.isInteger(parsed) || parsed < 1 || parsed > maxProcessedFrames) {
        setSubmitError(`Max frames must be an integer between 1 and ${maxProcessedFrames}.`);
        return;
      }
    }

    if (gpsFile) {
      const offsetVal = Number(gpsSyncOffset);
      const gapVal = Number(gpsMaxGap);
      if (!Number.isFinite(offsetVal)) {
        setSubmitError('GPS Sync Offset must be a valid number.');
        return;
      }
      if (!Number.isFinite(gapVal) || gapVal <= 0 || gapVal > 60) {
        setSubmitError('GPS Max Sync Gap must be greater than 0 and at most 60 seconds.');
        return;
      }
    }

    setSubmitting(true);
    setSubmitError(null);

    const formData = new FormData();
    formData.append('video', file);
    formData.append('mode', mode);
    formData.append('enable_depth', String(enableDepth));
    formData.append('enable_geometry', String(enableGeometry));
    formData.append('diagnostics', String(diagnostics));
    formData.append('save_video', String(saveVideo));
    formData.append('save_json', String(saveJson));
    formData.append('save_csv', String(saveCsv));
    if (maxFrames.trim() !== '') {
      formData.append('max_frames', String(Number(maxFrames)));
    }

    try {
      const resp = await createJob(formData);
      if (!resp?.job_id) {
        throw new Error('Backend accepted upload but did not return a job_id.');
      }
      if (gpsFile) {
        const gpsForm = new FormData();
        gpsForm.append('gps', gpsFile);
        gpsForm.append('time_mode', gpsTimeMode);
        gpsForm.append('sync_offset_sec', gpsSyncOffset);
        gpsForm.append('max_sync_gap_sec', gpsMaxGap);
        await uploadJobGps(resp.job_id, gpsForm);
      }
      onNavigate('workspace', resp.job_id);
    } catch (err) {
      setSubmitError(err.message || 'Failed to submit inspection job.');
    } finally {
      setSubmitting(false);
    }
  };

  if (loadingConfig) {
    return (
      <div className="rv-page">
        <StatePanel
          variant="loading"
          title="Loading pipeline modes and limits"
          message="Fetching /api/v2/modes and /api/v2/capabilities…"
        />
      </div>
    );
  }

  if (configError) {
    return (
      <div className="rv-page">
        <StatePanel
          variant={configError.isApiDisabled ? 'api-disabled' : configError.isNetworkError ? 'disconnected' : 'error'}
          title={configError.isNetworkError ? 'Backend server disconnected' : 'Cannot initialize inspection form'}
          message={configError.message}
        />
      </div>
    );
  }

  const modeOrder = capabilities?.supported_modes || Object.keys(modesData || {});

  return (
    <form className="rv-page" onSubmit={handleSubmit}>
      <div className="rv-page-header">
        <div>
          <h2 className="rv-page-title">Submit New Road Inspection</h2>
          <p className="rv-page-subtitle">
            Upload a dashcam or mobile survey clip to queue YOLOv8n + ByteTrack + relative depth analysis.
          </p>
        </div>
      </div>

      <section className="rv-card">
        <div className="rv-card-header">
          <h3 className="rv-card-title">1. Video Source</h3>
          <span className="rv-card-meta">
            Max {formatBytes(maxUploadBytes)} · Formats: {allowedFormats.join(', ')} · Max resolution:{' '}
            {limits.maximum_video_resolution ? `${limits.maximum_video_resolution[0]}x${limits.maximum_video_resolution[1]}` : '1920x1080'} · Max duration:{' '}
            {limits.maximum_video_duration_sec ? `${limits.maximum_video_duration_sec}s` : '300s'}
          </span>
        </div>

        <label className={`rv-upload-zone ${file ? 'has-file' : ''}`}>
          <input
            type="file"
            accept={allowedFormats.join(',')}
            onChange={handleFileChange}
            style={{ display: 'none' }}
          />
          {file ? <FileVideo size={28} className="rv-icon-cyan" /> : <Upload size={28} className="rv-icon-cyan" />}
          <div style={{ fontWeight: 600, fontSize: 14 }}>
            {file ? file.name : 'Click to select a road survey video file'}
          </div>
          <div style={{ fontSize: 12, color: 'var(--rv-text-secondary)' }}>
            {file ? `${formatBytes(file.size)} · Ready for upload` : 'Server verifies OpenCV stream decodability, dimensions, and duration before queuing.'}
          </div>
        </label>
      </section>

      <section className="rv-card">
        <div className="rv-card-header">
          <h3 className="rv-card-title">2. Operating Mode</h3>
          <span className="rv-card-meta">Configured via config/roadvision_v2.yaml</span>
        </div>

        <div className="rv-mode-grid">
          {modeOrder.map((mKey) => {
            const mCfg = modesData?.[mKey] || {};
            const isSelected = mode === mKey;
            return (
              <button
                key={mKey}
                type="button"
                className={`rv-mode-card ${isSelected ? 'selected' : ''}`}
                onClick={() => handleModeSelect(mKey)}
              >
                <div className="rv-mode-head">
                  <span>{mKey.toUpperCase()}</span>
                  {isSelected && <CheckCircle2 size={16} className="rv-icon-cyan" />}
                </div>
                <div className="rv-mode-desc">
                  {mCfg.description || 'RoadVision V2 inspection profile'}
                </div>
                <div className="rv-mode-specs">
                  <span>Depth: {mCfg.depth?.enabled ? `${mCfg.depth.model_id} (${mCfg.depth.input_size}px)` : 'Off by default'}</span>
                  <span>Scheduler: {mCfg.scheduler?.strategy || 'default'}</span>
                </div>
              </button>
            );
          })}
        </div>
      </section>

      <section className="rv-card">
        <div className="rv-card-header">
          <h3 className="rv-card-title">3. Execution & Artifact Options</h3>
          <span className="rv-card-meta">Validated server-side against API safety limits</span>
        </div>

        <div className="rv-checkbox-grid" style={{ marginBottom: 14 }}>
          <label className="rv-checkbox-item">
            <input type="checkbox" checked={enableDepth} onChange={(e) => setEnableDepth(e.target.checked)} />
            <span>Enable Depth Extraction</span>
          </label>
          <label className="rv-checkbox-item">
            <input type="checkbox" checked={enableGeometry} onChange={(e) => setEnableGeometry(e.target.checked)} />
            <span>Record 2.5D Geometry Snapshots</span>
          </label>
          <label className="rv-checkbox-item">
            <input type="checkbox" checked={diagnostics} onChange={(e) => setDiagnostics(e.target.checked)} />
            <span>Research Diagnostics</span>
          </label>
          <label className="rv-checkbox-item">
            <input type="checkbox" checked={saveVideo} onChange={(e) => setSaveVideo(e.target.checked)} />
            <span>Save Annotated Video</span>
          </label>
          <label className="rv-checkbox-item">
            <input type="checkbox" checked={saveJson} onChange={(e) => setSaveJson(e.target.checked)} />
            <span>Save Observations & Tracks JSON</span>
          </label>
          <label className="rv-checkbox-item">
            <input type="checkbox" checked={saveCsv} onChange={(e) => setSaveCsv(e.target.checked)} />
            <span>Save Performance CSV</span>
          </label>
        </div>

        <div className="rv-field" style={{ maxWidth: 260 }}>
          <label className="rv-field-label" htmlFor="rv-max-frames-input">
            Optional Max Frames (1–{maxProcessedFrames})
          </label>
          <input
            id="rv-max-frames-input"
            type="number"
            min={1}
            max={maxProcessedFrames}
            placeholder="All frames"
            value={maxFrames}
            onChange={(e) => setMaxFrames(e.target.value)}
            className="rv-input"
          />
        </div>
      </section>

      <section className="rv-card">
        <div className="rv-card-header">
          <h3 className="rv-card-title"><MapPin size={16} className="rv-icon-cyan" /> 4. Optional External GPS Telemetry</h3>
          <span className="rv-card-meta">Approximate camera/vehicle location only · Never fabricated from video</span>
        </div>
        <div className="rv-filter-bar" style={{ marginBottom: 0 }}>
          <div className="rv-field" style={{ flex: 1, minWidth: 220 }}>
            <label className="rv-field-label" htmlFor="rv-new-gps-file">GPS Log (.csv, .gpx, .geojson)</label>
            <input
              id="rv-new-gps-file"
              type="file"
              accept=".csv,.gpx,.geojson,.json"
              className="rv-input"
              onChange={(e) => setGpsFile(e.target.files?.[0] || null)}
            />
          </div>
          <div className="rv-field">
            <label className="rv-field-label" htmlFor="rv-new-gps-mode">Timestamp Mode</label>
            <select
              id="rv-new-gps-mode"
              className="rv-select"
              value={gpsTimeMode}
              disabled={!gpsFile}
              onChange={(e) => setGpsTimeMode(e.target.value)}
            >
              <option value="auto">Auto (Numeric or ISO-8601 TZ)</option>
              <option value="video_relative">Video-Relative Seconds</option>
              <option value="iso8601">ISO-8601 Timezone</option>
            </select>
          </div>
          <div className="rv-field">
            <label className="rv-field-label" htmlFor="rv-new-gps-off">Sync Offset (s)</label>
            <input
              id="rv-new-gps-off"
              type="number"
              step="0.1"
              className="rv-input"
              style={{ width: 105 }}
              value={gpsSyncOffset}
              disabled={!gpsFile}
              onChange={(e) => setGpsSyncOffset(e.target.value)}
            />
          </div>
          <div className="rv-field">
            <label className="rv-field-label" htmlFor="rv-new-gps-gap">Max Sync Gap (s)</label>
            <input
              id="rv-new-gps-gap"
              type="number"
              step="0.1"
              min="0.1"
              max="60"
              className="rv-input"
              style={{ width: 115 }}
              value={gpsMaxGap}
              disabled={!gpsFile}
              onChange={(e) => setGpsMaxGap(e.target.value)}
            />
          </div>
        </div>
      </section>

      {submitError && (
        <div className="rv-state-panel rv-state-error" role="alert">
          <AlertTriangle size={18} className="rv-icon-danger" />
          <div className="rv-state-body">
            <h4 className="rv-state-title">Submission Error</h4>
            <p className="rv-state-message">{submitError}</p>
          </div>
        </div>
      )}

      <div className="rv-header-actions">
        <button
          type="submit"
          className="rv-btn rv-btn-primary"
          disabled={!file || submitting || !capabilities?.detector_checkpoint_available}
        >
          {submitting ? <Loader2 size={16} className="rv-spin" /> : <Play size={16} />}
          <span>{submitting ? 'Uploading & Queuing Job…' : 'Submit Inspection Job'}</span>
        </button>
        {!capabilities?.detector_checkpoint_available && (
          <span style={{ fontSize: 12, color: '#fbbf24' }}>
            Detector checkpoint unavailable on backend server.
          </span>
        )}
      </div>

      <ScientificDisclaimer />
    </form>
  );
}
