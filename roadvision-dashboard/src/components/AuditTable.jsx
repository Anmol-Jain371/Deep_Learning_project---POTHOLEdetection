import React, { useState } from 'react';
import { Search, Download, FileText, Droplets } from 'lucide-react';

export default function AuditTable({ data = [] }) {
  const [search, setSearch] = useState('');
  const [sevFilter, setSevFilter] = useState('ALL');

  const safeData = data || [];

  const filtered = safeData.filter(p => {
    const matchSearch = (p.id || '').toLowerCase().includes(search.toLowerCase()) ||
                        (p.status || '').toLowerCase().includes(search.toLowerCase());
    const matchSev = sevFilter === 'ALL' || (p.severity || '').includes(sevFilter);
    return matchSearch && matchSev;
  });

  const hasNum = (v) => v !== null && v !== undefined && !Number.isNaN(Number(v)) && Number.isFinite(Number(v));

  const exportCSV = () => {
    const headers = 'Pothole_ID,Timestamp_Sec,Severity,Experimental_Depth_cm_Uncalibrated,Experimental_Width_cm_Uncalibrated,Experimental_Length_cm_Uncalibrated,Experimental_Volume_Liters_Uncalibrated,Experimental_Asphalt_kg_Uncalibrated,Water_Hazard,Confidence,Physical_Accuracy_Status,Municipal_Status\n';
    const rows = safeData.map(p => {
      return [
        p.id,
        hasNum(p.timestamp) ? p.timestamp : 'UNAVAILABLE',
        p.severity || 'UNAVAILABLE',
        hasNum(p.depth_cm) ? p.depth_cm : 'UNAVAILABLE',
        hasNum(p.width_cm) ? p.width_cm : 'UNAVAILABLE',
        hasNum(p.length_cm) ? p.length_cm : 'UNAVAILABLE',
        hasNum(p.volume_liters) ? p.volume_liters : 'UNAVAILABLE',
        hasNum(p.asphalt_kg) ? p.asphalt_kg : 'UNAVAILABLE',
        p.water ? 'YES' : 'NO',
        hasNum(p.confidence) ? p.confidence : 'UNAVAILABLE',
        '"Uncalibrated experimental estimate — physical accuracy not established"',
        `"${p.status || 'Unavailable'}"`,
      ].join(',');
    }).join('\n');
    const blob = new Blob([headers + rows], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `road_hazard_audit_${new Date().toISOString().slice(0, 10)}.csv`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  const sevClass = (sev = '') => {
    if (sev.includes('SEVERE')) return 'severe';
    if (sev.includes('MODERATE')) return 'moderate';
    return 'mild';
  };

  const sevDotColor = (sev = '') => {
    if (sev.includes('SEVERE')) return '#dc2626';
    if (sev.includes('MODERATE')) return '#d97706';
    return '#16a34a';
  };

  const filterOptions = [
    { id: 'ALL', label: 'All' },
    { id: 'SEVERE', label: 'Severe' },
    { id: 'MODERATE', label: 'Moderate' },
    { id: 'MILD', label: 'Mild' },
  ];

  return (
    <div className="audit-panel">
      <div className="audit-toolbar">
        <div className="audit-toolbar-left">
          <FileText size={16} style={{ color: '#e8651a' }} />
          <h3>Road Defect Audit Log</h3>
          <span className="audit-badge">{safeData.length} records</span>
        </div>

        <div className="audit-toolbar-right">
          <div className="map-filters" role="group" aria-label="Filter defects by severity">
            {filterOptions.map(f => (
              <button
                key={f.id}
                className={`filter-chip ${sevFilter === f.id ? 'active' : ''}`}
                onClick={() => setSevFilter(f.id)}
                aria-pressed={sevFilter === f.id}
              >
                {f.label}
              </button>
            ))}
          </div>

          <div className="search-box">
            <Search size={14} style={{ color: '#64748b' }} />
            <input
              placeholder="Search hazard ID or status..."
              value={search}
              onChange={e => setSearch(e.target.value)}
              aria-label="Search hazard ID or status"
            />
          </div>

          <button className="btn-export" onClick={exportCSV} aria-label="Export audit records as CSV">
            <Download size={14} />
            Export CSV
          </button>
        </div>
      </div>

      <div className="audit-table-wrap">
        <table className="audit-table">
          <thead>
            <tr>
              <th>Hazard ID</th>
              <th>Timestamp</th>
              <th>Severity</th>
              <th>Est. Depth</th>
              <th>Water Risk</th>
              <th>Confidence</th>
              <th>Status</th>
            </tr>
          </thead>
          <tbody>
            {filtered.length === 0 ? (
              <tr>
                <td colSpan="7" style={{ textAlign: 'center', padding: '40px', color: '#94a3b8' }}>
                  {safeData.length === 0
                    ? 'No hazards registered in this video survey. Pavement is clear.'
                    : 'No matching records found for current filters.'}
                </td>
              </tr>
            ) : (
              filtered.map(p => (
                <tr key={p.id}>
                  <td className="cell-id">
                    {p.id}
                  </td>
                  <td className="cell-mono">
                    {hasNum(p.timestamp) ? `${Number(p.timestamp).toFixed(1)}s` : 'Unavailable'}
                  </td>
                  <td>
                    <span className={`severity-badge ${sevClass(p.severity || '')}`}>
                      <span className="sev-dot" style={{ background: sevDotColor(p.severity || '') }} />
                      {p.severity || 'Unavailable'}
                    </span>
                  </td>
                  <td className="cell-depth">
                    <strong>{hasNum(p.depth_cm) ? `${p.depth_cm} cm (uncal.)` : 'Unavailable'}</strong>
                    <div style={{ fontSize: '10px', color: '#94a3b8', marginTop: '2px' }}>
                      {hasNum(p.width_cm) && hasNum(p.length_cm)
                        ? `${p.width_cm} × ${p.length_cm} cm · `
                        : ''}
                      Uncalibrated experimental estimate — physical accuracy not established.
                    </div>
                  </td>
                  <td>
                    {p.water ? (
                      <span className="water-pill" title="Water-filled crater detected">
                        <Droplets size={12} />
                        Aquaplane Risk
                      </span>
                    ) : (
                      <span className="dry-pill">Dry Cavity</span>
                    )}
                  </td>
                  <td>
                    {hasNum(p.confidence) ? (
                      <div className="confidence-bar-wrap">
                        <div className="confidence-bar">
                          <div className="confidence-fill" style={{ width: `${Math.min(100, Number(p.confidence) * 100)}%` }} />
                        </div>
                        <span style={{ fontFamily: "'Space Mono', monospace", fontSize: '12px' }}>
                          {Math.round(Number(p.confidence) * 100)}%
                        </span>
                      </div>
                    ) : (
                      <span style={{ color: '#94a3b8', fontSize: '12px' }}>Unavailable</span>
                    )}
                  </td>
                  <td><span className="status-tag">{p.status || 'Unavailable'}</span></td>
                </tr>
              ))
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
}
