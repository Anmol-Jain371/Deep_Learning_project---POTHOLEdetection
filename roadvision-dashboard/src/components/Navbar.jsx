import React from 'react';
import { Eye, BarChart3, Layers } from 'lucide-react';

export default function Navbar({ active, onNavigate, variant = 'dark' }) {
  const links = [
    { id: 'home', label: 'Home', icon: Eye },
    { id: 'dashboard', label: 'Dashboard', icon: BarChart3 },
    { id: 'architecture', label: 'Architecture', icon: Layers },
  ];

  const isEditorial = variant === 'editorial';

  return (
    <nav className={`navbar ${isEditorial ? 'navbar-editorial' : ''}`}>
      <div className="navbar-brand" onClick={() => onNavigate('home')} style={{ cursor: 'pointer' }} role="button" tabIndex={0} onKeyDown={(e) => { if (e.key === 'Enter') onNavigate('home'); }}>
        <div className="brand-mark">RV</div>
        <div className="brand-text">
          Road<span>Vision</span> AI
        </div>
      </div>

      <div className="navbar-links">
        {links.map(link => {
          return (
            <button
              key={link.id}
              className={`nav-link ${active === link.id ? 'active' : ''}`}
              onClick={() => onNavigate(link.id)}
            >
              {link.label}
            </button>
          );
        })}
      </div>

      {isEditorial && active !== 'dashboard' && (
        <div className="navbar-action">
          <button
            className="navbar-cta-pill"
            onClick={() => onNavigate('dashboard')}
            title="Open RoadVision AI Command Center"
          >
            Launch Dashboard
          </button>
        </div>
      )}
    </nav>
  );
}
