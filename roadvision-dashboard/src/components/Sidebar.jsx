import React, { useState } from 'react';
import {
  PanelLeftClose,
  PanelLeftOpen,
  LayoutDashboard,
  PlusCircle,
  History,
  MonitorPlay,
  Crosshair,
  GitCommitHorizontal,
  Activity,
  FileBarChart2,
  Settings,
  Archive,
  Box,
  MapPin,
  ClipboardCheck,
} from 'lucide-react';

const NAV_GROUPS = [
  {
    title: 'Operations',
    items: [
      { id: 'overview', label: 'Overview', icon: LayoutDashboard },
      { id: 'new', label: 'New Inspection', icon: PlusCircle },
      { id: 'history', label: 'Inspection History', icon: History },
      { id: 'workspace', label: 'Inspection Workspace', icon: MonitorPlay },
    ],
  },
  {
    title: 'Intelligence',
    items: [
      { id: 'hazard', label: 'Hazard Explorer', icon: Crosshair },
      { id: 'track', label: 'Track Analysis', icon: GitCommitHorizontal },
      { id: 'relative3d', label: 'Relative 3D Analysis', icon: Box },
      { id: 'geospatial', label: 'Geospatial Intelligence', icon: MapPin },
      { id: 'maintenance', label: 'Maintenance Intelligence', icon: ClipboardCheck },
      { id: 'performance', label: 'Performance', icon: Activity },
      { id: 'reports', label: 'Reports', icon: FileBarChart2 },
    ],
  },
  {
    title: 'Platform',
    items: [
      { id: 'settings', label: 'Settings', icon: Settings },
      { id: 'legacy', label: 'Legacy V1 Suite', icon: Archive },
    ],
  },
];

export default function Sidebar({ active, onNavigate }) {
  const [collapsed, setCollapsed] = useState(false);

  return (
    <aside className={`rv-sidebar ${collapsed ? 'collapsed' : ''}`} aria-label="Primary Navigation">
      <div className="rv-sidebar-header">
        <button
          type="button"
          className="rv-brand-btn"
          onClick={() => onNavigate('overview')}
          title="RoadVision AI 2.0 Overview"
        >
          <div className="rv-brand-mark">RV</div>
          {!collapsed && (
            <div className="rv-brand-copy">
              <span className="rv-brand-title">
                Road<span>Vision</span>
              </span>
              <span className="rv-brand-sub">AI 2.0 GIS CONSOLE</span>
            </div>
          )}
        </button>
        <button
          type="button"
          className="rv-collapse-btn"
          onClick={() => setCollapsed((c) => !c)}
          aria-label={collapsed ? 'Expand sidebar' : 'Collapse sidebar'}
          title={collapsed ? 'Expand sidebar' : 'Collapse sidebar'}
        >
          {collapsed ? <PanelLeftOpen size={17} /> : <PanelLeftClose size={17} />}
        </button>
      </div>

      <nav className="rv-nav">
        {NAV_GROUPS.map((group) => (
          <div key={group.title} className="rv-nav-group">
            {!collapsed && <div className="rv-nav-group-label">{group.title}</div>}
            {group.items.map((item) => {
              const Icon = item.icon;
              const isActive = active === item.id;
              return (
                <button
                  key={item.id}
                  type="button"
                  className={`rv-nav-item ${isActive ? 'active' : ''}`}
                  onClick={() => onNavigate(item.id)}
                  title={collapsed ? item.label : undefined}
                  aria-current={isActive ? 'page' : undefined}
                >
                  <Icon size={17} className="rv-nav-icon" />
                  {!collapsed && <span className="rv-nav-label">{item.label}</span>}
                </button>
              );
            })}
          </div>
        ))}
      </nav>
    </aside>
  );
}
