import React, { useState } from 'react';
import Dashboard from './components/Dashboard';
import LiveVideoProcessor from './components/LiveVideoProcessor';

export default function App() {
  const [currentView, setCurrentView] = useState('dashboard');

  return (
    <div style={{ height: '100vh', display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
      <nav style={{ flexShrink: 0, padding: '1rem', background: '#1a2235', display: 'flex', gap: '1rem', borderBottom: '1px solid #2a3553' }}>
        <button 
          onClick={() => setCurrentView('dashboard')}
          style={{ padding: '0.5rem 1rem', background: currentView === 'dashboard' ? '#00ffcc' : 'transparent', color: currentView === 'dashboard' ? '#0b0f19' : '#00ffcc', border: '1px solid #00ffcc', borderRadius: '4px', cursor: 'pointer', fontWeight: 'bold' }}
        >
          Dashboard
        </button>
        <button 
          onClick={() => setCurrentView('live')}
          style={{ padding: '0.5rem 1rem', background: currentView === 'live' ? '#00ffcc' : 'transparent', color: currentView === 'live' ? '#0b0f19' : '#00ffcc', border: '1px solid #00ffcc', borderRadius: '4px', cursor: 'pointer', fontWeight: 'bold' }}
        >
          Live Video Upload
        </button>
      </nav>
      {currentView === 'dashboard' ? (
        <div style={{ flex: 1, overflow: 'hidden' }}>
          <Dashboard />
        </div>
      ) : (
        <div id="live-video-scroller" style={{ flex: 1, overflowY: 'auto' }}>
          <LiveVideoProcessor />
        </div>
      )}
    </div>
  );
}
