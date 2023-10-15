import React, { useState } from 'react';
import './MovingWallpaper.css';
import backgroundVideo from '../assets/Background.mp4';

function MovingWallpaper({ onGoBack }) {
  const [showPasswordPrompt, setShowPasswordPrompt] = useState(false);
  const [password, setPassword] = useState('');
  const correctPassword = '123'; 

  const handleBackClick = () => {
    setShowPasswordPrompt(true);
  };

  const handlePasswordSubmit = () => {
    if (password === correctPassword) {
      setShowPasswordPrompt(false);
      onGoBack();
    } else {
      alert('Incorrect password!');
    }
  };

  return (
    <div className="moving-wallpaper">
      <div className="video-overlay"></div>
      <video className="video-background" autoPlay loop muted>
        <source src={backgroundVideo} type="video/mp4" />
      </video>
      <p>This Computer is Being Monitored by Dubhacks</p>

      {showPasswordPrompt && (
        <div className="modal">
          <div className="modal-content">
            <input
              type="password"
              className="password-input"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              placeholder="Enter password"
            />
            <button onClick={handlePasswordSubmit} className="password-submit">Submit</button>
          </div>
        </div>
      )}

      <button onClick={handleBackClick} className="back-button">Enter Password to Exit</button>
    </div>
  );
}

export default MovingWallpaper;
