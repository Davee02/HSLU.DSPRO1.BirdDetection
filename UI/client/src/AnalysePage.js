import React, { useState, useEffect } from 'react';
import { MapContainer, TileLayer, Marker } from 'react-leaflet';
import L from 'leaflet';
import 'leaflet/dist/leaflet.css';
import './AnalysePage.css';

delete L.Icon.Default.prototype._getIconUrl;
L.Icon.Default.mergeOptions({
  iconRetinaUrl: require('leaflet/dist/images/marker-icon-2x.png'),
  iconUrl: require('leaflet/dist/images/marker-icon.png'),
  shadowUrl: require('leaflet/dist/images/marker-shadow.png'),
});

const AnalysePage = () => {
  const [file, setFile] = useState(null);
  const [description, setDescription] = useState(
    'Reminder: Make sure your recording is between 2 and 15 seconds, has good sound quality, and is from the European region.'
  );
  const [position, setPosition] = useState(null);

  useEffect(() => {
    if (navigator.geolocation) {
      navigator.geolocation.getCurrentPosition(
        (pos) => {
          const { latitude, longitude } = pos.coords;
          setPosition([latitude, longitude]);
        },
        () => alert('Unable to retrieve your location')
      );
    } else {
      alert('Geolocation is not supported by your browser');
    }
  }, []);

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
        {position ? (
          <MapContainer
            center={position}
            zoom={13}
            style={{ height: '100%', width: '100%' }}
          >
            <TileLayer
              url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
              attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
            />
            <Marker position={position}></Marker>
          </MapContainer>
        ) : (
          <p>Loading map...</p>
        )}
      </div>
    </div>
  );
};

export default AnalysePage;
