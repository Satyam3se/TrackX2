import React, { useState } from 'react';
import Dashboard from './components/Dashboard';
import LiveVideoProcessor from './components/LiveVideoProcessor';

export default function App() {
  const [currentView, setCurrentView] = useState('dashboard');

  return (
    <div className="app">
      <nav style={{ flexShrink: 0, padding: '16px 24px', background: 'var(--bg-color-alt)', display: 'flex', gap: '16px', borderBottom: '1px solid var(--glass-border)' }}>
        <button 
          onClick={() => setCurrentView('dashboard')}
          className={`glass-btn ${currentView === 'dashboard' ? 'active' : ''}`}
        >
          Dashboard
        </button>
        <button 
          onClick={() => setCurrentView('live')}
          className={`glass-btn ${currentView === 'live' ? 'active' : ''}`}
        >
          Live Video Upload
        </button>
      </nav>
      {currentView === 'dashboard' ? (
        <div style={{ flex: 1, overflow: 'hidden' }}>
          <Dashboard />
        </div>
      ) : (
        <div id="live-video-scroller" style={{ flex: 1, overflowY: 'auto', padding: '24px' }}>
          <LiveVideoProcessor />
        </div>
      )}
    </div>
  );
}
