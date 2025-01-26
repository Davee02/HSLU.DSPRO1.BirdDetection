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
  const [nearbyMarkers, setNearbyMarkers] = useState([]); // New state for random markers
  const [confidence, setConfidence] = useState(null);
  const [predictions, setPredictions] = useState(null);

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

  const handleFileUpload = async (e) => {
    const uploadedFile = e.target.files[0];
    if (uploadedFile && uploadedFile.type.startsWith('audio')) {
      setFile(uploadedFile);
      setDescription('File uploaded successfully. Analyzing...');
      setLoading(true);
      setBirdData(null);
      setImageError(false);
      setGbifId(null);
      setNearbyMarkers([]);
      setConfidence(null);
      setPredictions(null);
  
      try {
        const formData = new FormData();
        formData.append('recording', uploadedFile);
  
        // Call the backend /predict endpoint
        const response = await axios.post('http://127.0.0.1:5000/predict', formData, {
          headers: {
            'Content-Type': 'multipart/form-data',
          },
        });
  
        const predictions = response.data.predictions;
        setPredictions(predictions);
        const topPrediction = Object.entries(predictions)[0]; // Get the most probable bird prediction
  
        if (topPrediction) {
          const [birdName, birdConfidence] = topPrediction;
          setConfidence(Math.round(birdConfidence * 100));
  
          setDescription(`This sounds like a ${birdName}. Fetching additional details...`);
          
          // Fetch scientific name and additional bird details
          await fetchBirdData(birdName);
        } else {
          setDescription('No bird species detected.');
        }
      } catch (error) {
        console.error('Error during analysis:', error);
        setDescription('An error occurred while analyzing the file.');
      } finally {
        setLoading(false);
      }
    } else {
      alert('Please upload a valid audio file.');
    }
  };

  const fetchBirdData = async (birdName) => {
    try {
      // Fetch bird data from eBird API
      const ebirdResponse = await axios.get(`https://api.ebird.org/v2/ref/taxonomy/ebird?fmt=json`, {
        headers: {
          'X-eBirdApiToken': 'b3qpess8ag5m'
        }
      });
      const allBirds = ebirdResponse.data;

      const birdDetail = allBirds.find((bird) => bird.comName.toLowerCase() === birdName.toLowerCase());
      
      if (!birdDetail) {
        setDescription('Bird information could not be retrieved.');
        return;
      }

      const wikiResponse = await axios.get(
        `https://en.wikipedia.org/api/rest_v1/page/summary/${encodeURIComponent(birdDetail.sciName)}`
      );

      const birdInfo = {
        name: birdDetail.comName,
        sciName: birdDetail.sciName,
        description: wikiResponse.data.extract,
        imageUrl: wikiResponse.data.thumbnail?.source || FALLBACK_IMAGE
      };

      setBirdData(birdInfo);
      setDescription(`This sounds like a ${birdDetail.comName} (Scientific Name: ${birdDetail.sciName}).`);

      // Fetch GBIF data for the species key
      const gbifResponse = await axios.get(
        `https://api.gbif.org/v1/species?name=${encodeURIComponent(birdDetail.sciName)}`
      );
      const gbifSpecies = gbifResponse.data.results[0];
      if (gbifSpecies) {
        setGbifId(gbifSpecies.key);
      }

      // Generate random nearby markers after analysis completes
      //generateNearbyMarkers();

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
    //setNearbyMarkers([]);
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
              Model Confidence: {confidence || 'N/A'}%
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
            <p>
              This model is {confidence || 'N/A'}% sure that this is a {birdData?.name || 'Unknown Bird'}.
            </p>
            <h3>Other possible matches:</h3>
            {predictions && Object.entries(predictions).slice(1, 4).map(([bird, conf]) => (
              <p key={bird}>
                {bird} - {Math.round(conf * 100)}%
              </p>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};

export default AnalysePage;