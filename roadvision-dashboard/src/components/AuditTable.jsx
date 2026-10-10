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

  const exportCSV = () => {
    const headers = 'Pothole_ID,Timestamp_Sec,Severity,Estimated_Depth_cm,Water_Hazard,Confidence,Municipal_Status\n';
    const rows = safeData.map(p => {
      return `${p.id},${p.timestamp || 0},${p.severity},${p.depth_cm},${p.water ? 'YES' : 'NO'},${p.confidence},"${p.status}"`;
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
                    {typeof p.timestamp === 'number' ? p.timestamp.toFixed(1) : p.timestamp || '0.0'}s
                  </td>
                  <td>
                    <span className={`severity-badge ${sevClass(p.severity)}`}>
                      <span className="sev-dot" style={{ background: sevDotColor(p.severity) }} />
                      {p.severity}
                    </span>
                  </td>
                  <td className="cell-depth">
                    {p.depth_cm} cm
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
                    <div className="confidence-bar-wrap">
                      <div className="confidence-bar">
                        <div className="confidence-fill" style={{ width: `${Math.min(100, (p.confidence || 0) * 100)}%` }} />
                      </div>
                      <span style={{ fontFamily: "'Space Mono', monospace", fontSize: '12px' }}>
                        {Math.round((p.confidence || 0) * 100)}%
                      </span>
                    </div>
                  </td>
                  <td><span className="status-tag">{p.status}</span></td>
                </tr>
              ))
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
}
