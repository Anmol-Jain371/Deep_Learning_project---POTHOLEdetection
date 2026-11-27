export const TERMINAL_JOB_STATUSES = new Set(['COMPLETED', 'FAILED', 'CANCELLED']);
export const ACTIVE_JOB_STATUSES = new Set(['QUEUED', 'INITIALIZING', 'RUNNING', 'CANCELLING']);
export const MEASUREMENT_STATES = [
  'FRESH_VALID',
  'CACHED_VALID',
  'FRESH_INVALID',
  'EXPIRED',
  'NOT_EVALUATED',
];

export const QUALITY_CATEGORIES = [
  'HIGH_QUALITY',
  'MODERATE_QUALITY',
  'LOW_QUALITY',
  'INSUFFICIENT_EVIDENCE',
  'INVALID',
];

export const REVIEW_STATUSES = [
  'PENDING_REVIEW',
  'IN_REVIEW',
  'CONFIRMED_VISUALLY',
  'REJECTED_FALSE_POSITIVE',
  'NEEDS_FIELD_INSPECTION',
];

export const REVIEW_PRIORITY_CATEGORIES = [
  'PRIORITY_REVIEW',
  'MANUAL_VERIFICATION_REQUIRED',
  'STANDARD_REVIEW',
  'INSUFFICIENT_EVIDENCE',
  'REVIEWED',
];

export const INSPECTION_PROVENANCE_CATEGORIES = [
  'USER_INSPECTION',
  'RESEARCH_EXPLORATORY',
  'TEST_FIXTURE',
  'SYNTHETIC_EXPERIMENT',
  'DEMONSTRATION',
  'UNKNOWN',
];

export const GPS_PROVENANCE_CATEGORIES = [
  'USER_SUPPLIED_UNVERIFIED',
  'VERIFIED_FIELD_LOG',
  'SYNTHETIC',
  'UNKNOWN',
];

export function inspectionProvenanceTone(prov) {
  switch (prov) {
    case 'USER_INSPECTION':
      return 'success';
    case 'RESEARCH_EXPLORATORY':
      return 'info';
    case 'TEST_FIXTURE':
    case 'SYNTHETIC_EXPERIMENT':
    case 'DEMONSTRATION':
      return 'warning';
    case 'UNKNOWN':
    default:
      return 'muted';
  }
}

export function gpsProvenanceTone(prov, isSynthetic = false) {
  if (isSynthetic || prov === 'SYNTHETIC') return 'warning';
  switch (prov) {
    case 'VERIFIED_FIELD_LOG':
      return 'success';
    case 'USER_SUPPLIED_UNVERIFIED':
      return 'info';
    case 'UNKNOWN':
    default:
      return 'muted';
  }
}

export const UUID_RE = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;

export function isValidUuid(value) {
  return typeof value === 'string' && UUID_RE.test(value.trim());
}

export function formatNullable(value, digits = null, suffix = '') {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return 'Unavailable';
  }
  if (typeof value === 'number' && digits !== null) {
    return `${value.toFixed(digits)}${suffix}`;
  }
  return `${value}${suffix}`;
}

export function formatRatePercent(rate) {
  if (rate === null || rate === undefined || Number.isNaN(rate)) {
    return 'Unavailable';
  }
  return `${(Number(rate) * 100).toFixed(1)}%`;
}

export function formatBytes(bytes) {
  if (bytes === null || bytes === undefined || Number.isNaN(bytes)) {
    return 'Unavailable';
  }
  const n = Number(bytes);
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KiB`;
  return `${(n / (1024 * 1024)).toFixed(2)} MiB`;
}

export function formatDateTime(iso) {
  if (!iso) return 'Unavailable';
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return 'Unavailable';
  return d.toLocaleString();
}

export function formatSeconds(sec) {
  if (sec === null || sec === undefined || Number.isNaN(sec)) {
    return 'Unavailable';
  }
  return `${Number(sec).toFixed(2)} s`;
}

export function statusTone(status) {
  switch (status) {
    case 'COMPLETED':
      return 'success';
    case 'RUNNING':
    case 'INITIALIZING':
      return 'info';
    case 'QUEUED':
    case 'CANCELLING':
      return 'warning';
    case 'FAILED':
      return 'danger';
    case 'CANCELLED':
    default:
      return 'neutral';
  }
}

export function measurementStateTone(state) {
  switch (state) {
    case 'FRESH_VALID':
      return 'success';
    case 'CACHED_VALID':
      return 'info';
    case 'FRESH_INVALID':
      return 'warning';
    case 'EXPIRED':
      return 'neutral';
    case 'NOT_EVALUATED':
    default:
      return 'muted';
  }
}

export function qualityCategoryTone(category) {
  switch (category) {
    case 'HIGH_QUALITY':
      return 'success';
    case 'MODERATE_QUALITY':
      return 'info';
    case 'LOW_QUALITY':
      return 'warning';
    case 'INVALID':
      return 'danger';
    case 'INSUFFICIENT_EVIDENCE':
    default:
      return 'muted';
  }
}

export function matchStatusTone(status) {
  switch (status) {
    case 'POTENTIAL_MATCH':
      return 'success';
    case 'REVIEW_REQUIRED':
      return 'warning';
    case 'UNMATCHED':
    default:
      return 'neutral';
  }
}

export function reviewPriorityTone(cat) {
  switch (cat) {
    case 'PRIORITY_REVIEW':
      return 'danger';
    case 'MANUAL_VERIFICATION_REQUIRED':
      return 'warning';
    case 'STANDARD_REVIEW':
      return 'info';
    case 'REVIEWED':
      return 'success';
    case 'INSUFFICIENT_EVIDENCE':
    default:
      return 'muted';
  }
}

export function reviewStatusTone(status) {
  switch (status) {
    case 'CONFIRMED_VISUALLY':
      return 'success';
    case 'REJECTED_FALSE_POSITIVE':
      return 'neutral';
    case 'NEEDS_FIELD_INSPECTION':
      return 'danger';
    case 'IN_REVIEW':
      return 'info';
    case 'PENDING_REVIEW':
    default:
      return 'warning';
  }
}

export function downloadBlobFile(blob, filename) {
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement('a');
  anchor.href = url;
  anchor.download = filename || 'artifact';
  document.body.appendChild(anchor);
  anchor.click();
  anchor.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
