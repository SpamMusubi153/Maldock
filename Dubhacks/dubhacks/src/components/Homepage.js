import React from 'react';
import './Homepage.css';
import wallpaper from '../assets/wallpaper.png'; 
function Homepage({ onStartRecording }) {
  return (
    <div className="homepage" style={{ backgroundImage: `url(${wallpaper})` }}>
      <button className="record-button" onClick={onStartRecording}>
        Begin Recording
      </button>
    </div>
  );
}

export default Homepage;
