// src/App.js
import React, { useState } from 'react';
import './App.css'; // You may need to adjust this import based on your project structure
import Homepage from './components/Homepage'; // Adjust the import path
import MovingWallpaper from './components/MovingWallpaper'; // Adjust the import path
import './assets/fonts/lemon_milk/LEMONMILK-Medium.otf';

function App() {
  const [isRecording, setIsRecording] = useState(false);

  const handleStartRecording = () => {
    setIsRecording(true);
  };

  const handleStopRecording = () => {
    setIsRecording(false);
  };
  

  return (
    <div className="App">
      
      {isRecording ? (
        <MovingWallpaper onGoBack={handleStopRecording} />
      ) : (
        <Homepage onStartRecording={handleStartRecording} />
      )}
    </div>
  );
}

export default App;
