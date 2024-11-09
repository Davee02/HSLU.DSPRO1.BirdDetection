// AnalysePage.js
import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { MapContainer, TileLayer, Marker, Popup } from 'react-leaflet';
import L from 'leaflet';
import 'leaflet/dist/leaflet.css';
import './AnalysePage.css';

const FALLBACK_IMAGE = 'https://example.com/fallback-image.jpg';

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
  const [loading, setLoading] = useState(false);
  const [birdData, setBirdData] = useState(null);
  const [imageError, setImageError] = useState(false);
  const [gbifId, setGbifId] = useState(null);
  const [showSurenessModal, setShowSurenessModal] = useState(false);
  const [surenessPercentage] = useState(96); // Fixed sureness percentage for demonstration
  const [nearbyMarkers, setNearbyMarkers] = useState([]); // New state for random markers

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
      setDescription('File uploaded successfully. Analyzing...');
      setLoading(true);
      setBirdData(null);
      setImageError(false);
      setGbifId(null);
      setNearbyMarkers([]); // Reset markers on new upload

      // Simulate analysis by waiting for 5 seconds, then fetch bird data
      setTimeout(() => {
        fetchBirdData();
      }, 5000);
    } else {
      alert('Please upload a valid audio file.');
    }
  };

  const fetchBirdData = async () => {
    try {
      // Fetch bird data from eBird API
      const ebirdResponse = await axios.get(`https://api.ebird.org/v2/ref/taxonomy/ebird?fmt=json`, {
        headers: {
          'X-eBirdApiToken': 'b3qpess8ag5m'
        }
      });
      const allBirds = ebirdResponse.data;
      
      // Pick a specific bird as a sample for demonstration (e.g., European Robin)
      const sampleBird = allBirds.find(bird => bird.comName === 'European Robin');

      if (sampleBird) {
        // Fetch Wikipedia data for additional details
        const wikiResponse = await axios.get(
          `https://en.wikipedia.org/api/rest_v1/page/summary/${encodeURIComponent(sampleBird.sciName)}`
        );

        const birdInfo = {
          name: sampleBird.comName,
          sciName: sampleBird.sciName,
          description: wikiResponse.data.extract,
          imageUrl: wikiResponse.data.thumbnail?.source || FALLBACK_IMAGE
        };

        setBirdData(birdInfo);
        setDescription(`This sounds like a ${sampleBird.comName} (Scientific Name: ${sampleBird.sciName}).`);

        // Fetch GBIF data for the species key
        const gbifResponse = await axios.get(
          `https://api.gbif.org/v1/species?name=${encodeURIComponent(sampleBird.sciName)}`
        );
        const gbifSpecies = gbifResponse.data.results[0];
        if (gbifSpecies) {
          setGbifId(gbifSpecies.key);
        }

        // Generate random nearby markers after analysis completes
        generateNearbyMarkers();
      } else {
        setDescription('Bird species data could not be retrieved.');
      }
    } catch (error) {
      console.error('Error fetching bird data:', error);
      setDescription('An error occurred while fetching bird information.');
    } finally {
      setLoading(false);
    }
  };

  const generateNearbyMarkers = () => {
    if (position) {
      const markers = [];
      for (let i = 0; i < 1000; i++) {
        const randomLatOffset = (Math.random() - 0.5) * 0.5; // Random offset within ~1km
        const randomLngOffset = (Math.random() - 0.5) * 0.5;
        markers.push([position[0] + randomLatOffset, position[1] + randomLngOffset]);
      }
      setNearbyMarkers(markers);
    }
  };

  const handleFileChange = () => {
    setFile(null);
    setDescription(
      'Reminder: Make sure your recording is between 2 and 15 seconds, has good sound quality, and is from the European region.'
    );
    setBirdData(null);
    setImageError(false);
    setGbifId(null);
    setNearbyMarkers([]);
  };

  const openSurenessModal = () => {
    setShowSurenessModal(true);
  };

  const closeSurenessModal = () => {
    setShowSurenessModal(false);
  };

  return (
    <div className="analyse-page">
      <div className="analyse-box">
        <h3>Analysis Description</h3>
        {loading ? (
          <div className="loading-indicator">
            <p>Analyzing...</p>
            <div className="spinner"></div>
          </div>
        ) : birdData ? (
          <div className="bird-info">
            <p>{description}</p>
            <h3>Detected Bird: {birdData.name}</h3>
            <p><strong>Scientific Name:</strong> {birdData.sciName}</p>
            <img
              src={imageError ? FALLBACK_IMAGE : birdData.imageUrl}
              alt={birdData.name}
              className="bird-image"
              onError={() => setImageError(true)}
            />
            <p>{birdData.description}</p>
            {gbifId && (
              <p>
                <a
                  href={`https://www.gbif.org/species/${gbifId}`}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="distribution-link"
                >
                  View Distribution Map on GBIF
                </a>
              </p>
            )}
            <button className="sureness-button" onClick={openSurenessModal}>
              Model Confidence: {surenessPercentage}%
            </button>
          </div>
        ) : (
          <p>{description}</p>
        )}
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
            <Marker position={position}>
              <Popup>Your Location</Popup>
            </Marker>
            {nearbyMarkers.map((markerPos, index) => (
              <Marker key={index} position={markerPos}>
                <Popup>Another sighting of {birdData?.name}</Popup>
              </Marker>
            ))}
          </MapContainer>
        ) : (
          <p>Loading map...</p>
        )}
      </div>

      {showSurenessModal && (
        <div className="modal-overlay" onClick={closeSurenessModal}>
          <div className="modal-content" onClick={(e) => e.stopPropagation()}>
            <button className="close-button" onClick={closeSurenessModal}>X</button>
            <h2>Model Confidence</h2>
            <p>This model is {surenessPercentage}% sure that this is a {birdData.name}.</p>
            <h3>Other possible matches:</h3>
            <p>Common Sparrow - 3%</p>
            <p>House Finch - 1%</p>
            <p>European Starling - 1%</p>
          </div>
        </div>
      )}
    </div>
  );
};

export default AnalysePage;
