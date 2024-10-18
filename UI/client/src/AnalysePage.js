import React, { useState, useEffect } from 'react';
import { MapContainer, TileLayer, useMap } from 'react-leaflet';
import './AnalysePage.css';

// Custom component to update map's center based on user's location
const SetUserLocation = ({ setPosition }) => {
  const map = useMap();

  useEffect(() => {
    if (navigator.geolocation) {
      navigator.geolocation.getCurrentPosition(
        (position) => {
          const { latitude, longitude } = position.coords;
          const userLocation = [latitude, longitude];
          setPosition(userLocation); // Update the parent component's state
          map.setView(userLocation, 13); // Set the map's view to the user's location
        },
        () => {
          alert('Unable to retrieve your location');
        }
      );
    } else {
      alert('Geolocation is not supported by your browser');
    }
  }, [map, setPosition]);

  return null; // This component does not render anything itself
};

const AnalysePage = () => {
  const [file, setFile] = useState(null); // File state
  const [description, setDescription] = useState(
    'Reminder: Make sure your recording is between 2 and 15 seconds, has good sound quality, and is from the European region.'
  );
  const [position, setPosition] = useState([51.505, -0.09]); // Default to London's coordinates

  const handleFileUpload = (e) => {
    const uploadedFile = e.target.files[0];
    if (uploadedFile && uploadedFile.type.startsWith('audio')) {
      setFile(uploadedFile);
      setDescription('File uploaded successfully. Waiting for analysis...');
    } else {
      alert('Please upload a valid audio file.');
    }
  };

  const handleFileChange = () => {
    setFile(null);
    setDescription(
      'Reminder: Make sure your recording is between 2 and 15 seconds, has good sound quality, and is from the European region.'
    );
  };

  return (
    <div className="analyse-page">
      <div className="analyse-box">
        <h3>Analysis Description</h3>
        <p>{description}</p>
      </div>

      <div className="file-upload-box">
        <button
          className="upload-button"
          onClick={() => document.getElementById('file-input').click()}
        >
          {file ? 'Upload Another File' : 'Upload a Recording'}
        </button>
        <input
          id="file-input"
          type="file"
          accept="audio/*"
          onChange={handleFileUpload}
          style={{ display: 'none' }}
        />
      </div>

      <div className="file-display-box">
        {file ? (
          <div>
            <p>Uploaded File: {file.name}</p>
            <button onClick={handleFileChange}>Choose Another File</button>
          </div>
        ) : (
          <p>No file uploaded yet.</p>
        )}
      </div>

      <div className="map-box">
        <MapContainer
          center={position} // Set the map center to the user's position
          zoom={13}
          scrollWheelZoom={true}
          style={{ height: '100%', width: '100%' }} // Ensure the map fills its container
        >
          <TileLayer
            url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
            attribution="&copy; <a href='https://www.openstreetmap.org/copyright'>OpenStreetMap</a> contributors"
          />
          <SetUserLocation setPosition={setPosition} />
        </MapContainer>
      </div>
    </div>
  );
};

export default AnalysePage;
