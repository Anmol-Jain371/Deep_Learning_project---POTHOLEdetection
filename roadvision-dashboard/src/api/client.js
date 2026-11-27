// Centralized V2 API client for RoadVision AI 2.0
// Defaults to relative `/api/v2` (proxied by Vite during local development)
// and normalizes explicit base URLs without missing or duplicated `/api/v2` prefixes.

export function normalizeV2BaseUrl(raw) {
  const trimmed = (raw || '').trim();
  if (!trimmed) {
    return '/api/v2';
  }
  let clean = trimmed.replace(/\/+$/, '');
  if (/\/api\/v2(\/api\/v2)+$/i.test(clean)) {
    clean = clean.replace(/(\/api\/v2)+$/i, '/api/v2');
  } else if (/\/api\/v2$/i.test(clean)) {
    // Already ends with /api/v2
  } else if (/\/api$/i.test(clean)) {
    clean = `${clean}/v2`;
  } else {
    clean = `${clean}/api/v2`;
  }
  return clean;
}

const RAW_BASE =
  import.meta.env.VITE_API_BASE_URL ||
  import.meta.env.VITE_API_URL ||
  import.meta.env.VITE_BACKEND_URL ||
  '';

const BASE_URL = normalizeV2BaseUrl(RAW_BASE);

export function getApiBaseUrl() {
  return BASE_URL;
}

export class ApiClientError extends Error {
  constructor(message, { status = 0, errorCode = 'UNKNOWN_ERROR', details = null, isNetworkError = false, isApiDisabled = false } = {}) {
    super(message);
    this.name = 'ApiClientError';
    this.status = status;
    this.errorCode = errorCode;
    this.details = details;
    this.isNetworkError = isNetworkError;
    this.isApiDisabled = isApiDisabled;
  }
}

async function parseErrorResponse(response, path) {
  const contentType = response.headers.get('content-type') || '';
  let payload = null;
  if (contentType.includes('application/json')) {
    payload = await response.json().catch(() => null);
  }

  // Vite proxy returns 502/504 or 500 when the upstream Flask server is down
  if (response.status === 502 || response.status === 504) {
    return new ApiClientError('Backend server is unreachable (connection refused). Start server.py with ROADVISION_V2_ENABLED=true.', {
      status: response.status,
      errorCode: 'BACKEND_UNREACHABLE',
      isNetworkError: true,
    });
  }

  // If /api/v2/health returns 404 non-JSON, the V2 blueprint is likely disabled on the server
  if (path === '/health' && response.status === 404 && !payload) {
    return new ApiClientError('RoadVision V2 API is disabled on the backend (set ROADVISION_V2_ENABLED=true).', {
      status: 404,
      errorCode: 'V2_API_DISABLED',
      isApiDisabled: true,
    });
  }

  const msg =
    payload?.error ||
    payload?.message ||
    (response.statusText ? `HTTP ${response.status}: ${response.statusText}` : `HTTP ${response.status}`);

  const code = payload?.error_code || `HTTP_${response.status}`;
  return new ApiClientError(msg, {
    status: response.status,
    errorCode: code,
    details: payload?.details || null,
    isApiDisabled: code === 'V2_API_DISABLED' || (path === '/health' && response.status === 503),
  });
}

async function requestJson(path, options = {}) {
  let response;
  try {
    response = await fetch(`${BASE_URL}${path}`, options);
  } catch (err) {
    if (err?.name === 'AbortError') {
      throw err;
    }
    throw new ApiClientError(
      'Unable to reach the RoadVision backend. Ensure server.py is running on port 5000.',
      {
        status: 0,
        errorCode: 'BACKEND_UNREACHABLE',
        isNetworkError: true,
      }
    );
  }

  if (!response.ok) {
    throw await parseErrorResponse(response, path);
  }

  const contentType = response.headers.get('content-type') || '';
  if (contentType.includes('application/json')) {
    return response.json();
  }
  return null;
}

export async function getHealth(options = {}) {
  return requestJson('/health', options);
}

export async function getModes(options = {}) {
  return requestJson('/modes', options);
}

export async function getCapabilities(options = {}) {
  return requestJson('/capabilities', options);
}

export async function listJobs(params = {}, options = {}) {
  const clean = Object.fromEntries(
    Object.entries(params).filter(([, v]) => v !== undefined && v !== null && v !== '')
  );
  const qs = new URLSearchParams(clean).toString();
  return requestJson(`/jobs${qs ? `?${qs}` : ''}`, options);
}

export async function createJob(formData, options = {}) {
  return requestJson('/jobs', {
    method: 'POST',
    body: formData,
    ...options,
  });
}

export async function getJob(jobId, options = {}) {
  return requestJson(`/jobs/${encodeURIComponent(jobId)}`, options);
}

export async function cancelJob(jobId, options = {}) {
  return requestJson(`/jobs/${encodeURIComponent(jobId)}/cancel`, {
    method: 'POST',
    ...options,
  });
}

export async function getSummary(jobId, options = {}) {
  return requestJson(`/jobs/${encodeURIComponent(jobId)}/summary`, options);
}

export async function getObservations(jobId, params = {}, options = {}) {
  const clean = Object.fromEntries(
    Object.entries(params).filter(([, v]) => v !== undefined && v !== null && v !== '')
  );
  const qs = new URLSearchParams(clean).toString();
  return requestJson(`/jobs/${encodeURIComponent(jobId)}/observations${qs ? `?${qs}` : ''}`, options);
}

export async function getTracks(jobId, params = {}, options = {}) {
  const clean = Object.fromEntries(
    Object.entries(params).filter(([, v]) => v !== undefined && v !== null && v !== '')
  );
  const qs = new URLSearchParams(clean).toString();
  return requestJson(`/jobs/${encodeURIComponent(jobId)}/tracks${qs ? `?${qs}` : ''}`, options);
}

export async function getPerformance(jobId, params = {}, options = {}) {
  const clean = Object.fromEntries(
    Object.entries(params).filter(([, v]) => v !== undefined && v !== null && v !== '')
  );
  const qs = new URLSearchParams(clean).toString();
  return requestJson(`/jobs/${encodeURIComponent(jobId)}/performance${qs ? `?${qs}` : ''}`, options);
}

export async function getArtifacts(jobId, options = {}) {
  return requestJson(`/jobs/${encodeURIComponent(jobId)}/artifacts`, options);
}

export function getArtifactUrl(jobId, artifactName, { download = false } = {}) {
  const base = `${BASE_URL}/jobs/${encodeURIComponent(jobId)}/artifacts/${encodeURIComponent(artifactName)}`;
  return download ? `${base}?download=1` : base;
}

export async function downloadArtifact(jobId, artifactName) {
  let resp;
  try {
    resp = await fetch(getArtifactUrl(jobId, artifactName, { download: true }));
  } catch {
    throw new ApiClientError('Unable to reach backend to download artifact.', {
      status: 0,
      errorCode: 'BACKEND_UNREACHABLE',
      isNetworkError: true,
    });
  }
  if (!resp.ok) {
    throw await parseErrorResponse(resp, `/jobs/${jobId}/artifacts/${artifactName}`);
  }
  return resp.blob();
}

// Phase 10: Relative 2.5D Geometry Snapshots
export async function getGeometrySnapshots(jobId, params = {}, options = {}) {
  const clean = Object.fromEntries(
    Object.entries(params).filter(([, v]) => v !== undefined && v !== null && v !== '')
  );
  const qs = new URLSearchParams(clean).toString();
  return requestJson(`/jobs/${encodeURIComponent(jobId)}/geometry${qs ? `?${qs}` : ''}`, options);
}

export async function getGeometrySnapshot(jobId, snapshotId, options = {}) {
  return requestJson(
    `/jobs/${encodeURIComponent(jobId)}/geometry/${encodeURIComponent(snapshotId)}`,
    options
  );
}

// Phase 11: Geospatial Telemetry, Routes, Locations & Cross-Trip Matching
export async function uploadJobGps(jobId, formData, options = {}) {
  return requestJson(`/jobs/${encodeURIComponent(jobId)}/gps`, {
    method: 'POST',
    body: formData,
    ...options,
  });
}

export async function getJobRoute(jobId, options = {}) {
  return requestJson(`/jobs/${encodeURIComponent(jobId)}/route`, options);
}

export async function getJobLocations(jobId, params = {}, options = {}) {
  const clean = Object.fromEntries(
    Object.entries(params).filter(([, v]) => v !== undefined && v !== null && v !== '')
  );
  const qs = new URLSearchParams(clean).toString();
  return requestJson(`/jobs/${encodeURIComponent(jobId)}/locations${qs ? `?${qs}` : ''}`, options);
}

export async function runGeospatialMatch(payload = {}, options = {}) {
  return requestJson('/geospatial/match', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
    ...options,
  });
}

export async function listGeospatialClusters(params = {}, options = {}) {
  const clean = Object.fromEntries(
    Object.entries(params).filter(([, v]) => v !== undefined && v !== null && v !== '')
  );
  const qs = new URLSearchParams(clean).toString();
  return requestJson(`/geospatial/clusters${qs ? `?${qs}` : ''}`, options);
}

export async function getGeospatialCluster(clusterId, options = {}) {
  return requestJson(`/geospatial/clusters/${encodeURIComponent(clusterId)}`, options);
}

export async function reviewGeospatialCluster(clusterId, reviewStatus, options = {}) {
  return requestJson(`/geospatial/clusters/${encodeURIComponent(clusterId)}/review`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ review_status: reviewStatus }),
    ...options,
  });
}

// Phase 12: Road Maintenance Intelligence, Manual Reviews, Segments, Comparison & Reports
export async function getMaintenanceOverview(params = {}, options = {}) {
  const clean = Object.fromEntries(
    Object.entries(params).filter(([, v]) => v !== undefined && v !== null && v !== '')
  );
  const qs = new URLSearchParams(clean).toString();
  return requestJson(`/maintenance/overview${qs ? `?${qs}` : ''}`, options);
}

export async function getMaintenanceReviewQueue(params = {}, options = {}) {
  const clean = Object.fromEntries(
    Object.entries(params).filter(([, v]) => v !== undefined && v !== null && v !== '')
  );
  const qs = new URLSearchParams(clean).toString();
  return requestJson(`/maintenance/review-queue${qs ? `?${qs}` : ''}`, options);
}

export async function listMaintenanceReviews(params = {}, options = {}) {
  const clean = Object.fromEntries(
    Object.entries(params).filter(([, v]) => v !== undefined && v !== null && v !== '')
  );
  const qs = new URLSearchParams(clean).toString();
  return requestJson(`/maintenance/reviews${qs ? `?${qs}` : ''}`, options);
}

export async function getMaintenanceReview(reviewId, options = {}) {
  return requestJson(`/maintenance/reviews/${encodeURIComponent(reviewId)}`, options);
}

export async function createMaintenanceReview(payload, options = {}) {
  return requestJson('/maintenance/reviews', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
    ...options,
  });
}

export async function updateMaintenanceReview(reviewId, payload, options = {}) {
  return requestJson(`/maintenance/reviews/${encodeURIComponent(reviewId)}`, {
    method: 'PATCH',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
    ...options,
  });
}

export async function getJobMaintenance(jobId, options = {}) {
  return requestJson(`/jobs/${encodeURIComponent(jobId)}/maintenance`, options);
}

export async function getJobSegments(jobId, segmentLengthM = 50, options = {}) {
  return requestJson(
    `/jobs/${encodeURIComponent(jobId)}/segments?segment_length_m=${encodeURIComponent(segmentLengthM)}`,
    options
  );
}

export async function compareInspections(jobIds = [], options = {}) {
  const qs = new URLSearchParams({ job_ids: jobIds.join(',') }).toString();
  return requestJson(`/maintenance/compare?${qs}`, options);
}

export function getJobReportUrl(jobId, format = 'json', { download = false } = {}) {
  const params = new URLSearchParams({ format });
  if (download) params.set('download', '1');
  return `${BASE_URL}/jobs/${encodeURIComponent(jobId)}/report?${params.toString()}`;
}

export async function downloadJobReport(jobId, format = 'json') {
  let resp;
  try {
    resp = await fetch(getJobReportUrl(jobId, format, { download: true }));
  } catch {
    throw new ApiClientError('Unable to reach backend to download inspection report.', {
      status: 0,
      errorCode: 'BACKEND_UNREACHABLE',
      isNetworkError: true,
    });
  }
  if (!resp.ok) {
    throw await parseErrorResponse(resp, `/jobs/${jobId}/report`);
  }
  return resp.blob();
}
