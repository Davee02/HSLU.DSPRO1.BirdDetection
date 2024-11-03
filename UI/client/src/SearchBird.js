// SearchBird.js
import React, { useState, useEffect } from 'react';
import axios from 'axios';
import './SearchBird.css';

const FALLBACK_IMAGE = 'https://example.com/fallback-image.jpg';

const SearchBird = () => {
  const [searchTerm, setSearchTerm] = useState('');
  const [birdResults, setBirdResults] = useState([]);
  const [allBirds, setAllBirds] = useState([]);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);
  const [selectedBird, setSelectedBird] = useState(null);
  const [gbifId, setGbifId] = useState(null);

  useEffect(() => {
    const fetchBirds = async () => {
      try {
        const response = await axios.get(`https://api.ebird.org/v2/ref/taxonomy/ebird?fmt=json`, {
          headers: {
            'X-eBirdApiToken': 'b3qpess8ag5m'
          }
        });
        
        setAllBirds(response.data);
      } catch (error) {
        setError('Error fetching bird data. Please try again.');
      }
    };

    fetchBirds();
  }, []);

  useEffect(() => {
    if (!searchTerm) {
      setBirdResults([]);
      return;
    }

    const fetchBirdsWithWikipediaInfo = async () => {
      setLoading(true);
      setBirdResults([]);

      const filteredBirds = allBirds
        .filter((bird) =>
          bird.comName.toLowerCase().includes(searchTerm.toLowerCase())
        )
        .slice(0, 20);

      const birdsWithWikiInfo = [];

      for (const bird of filteredBirds) {
        const simplifiedName = bird.sciName;

        try {
          const wikiResponse = await axios.get(
            `https://en.wikipedia.org/api/rest_v1/page/summary/${encodeURIComponent(simplifiedName)}`
          );

          if (wikiResponse.data && wikiResponse.data.extract) {
            birdsWithWikiInfo.push({
              name: bird.comName,
              sciName: bird.sciName,
              description: wikiResponse.data.extract,
              imageUrl: wikiResponse.data.thumbnail?.source || FALLBACK_IMAGE
            });

            if (birdsWithWikiInfo.length >= 10) break;
          }
        } catch (error) {
          console.warn(`Wikipedia data unavailable for ${bird.sciName}`);
        }
      }

      setBirdResults(birdsWithWikiInfo);
      setLoading(false);
    };

    fetchBirdsWithWikipediaInfo();
  }, [searchTerm, allBirds]);

  const handleBirdClick = async (bird) => {
    setSelectedBird(bird);

    try {
      const gbifResponse = await axios.get(
        `https://api.gbif.org/v1/species?name=${encodeURIComponent(bird.sciName)}`
      );
      const gbifSpecies = gbifResponse.data.results[0];
      if (gbifSpecies) {
        setGbifId(gbifSpecies.key);
      } else {
        setGbifId(null);
      }
    } catch (error) {
      console.error('Error fetching GBIF ID:', error);
      setGbifId(null);
    }
  };

  const closePopup = () => {
    setSelectedBird(null);
    setGbifId(null);
  };

  return (
    <div className="search-bird">
      <h3>Search for Birds</h3>
      <input
        type="text"
        placeholder="Enter bird name"
        value={searchTerm}
        onChange={(e) => setSearchTerm(e.target.value)}
        className="search-input"
      />

      {error && <p className="error-message">{error}</p>}
      {loading && <p className="loading-message">Loading results...</p>}

      <div className="bird-results">
        {birdResults.map((bird) => (
          <div
            key={bird.sciName}
            className="bird-result"
            onClick={() => handleBirdClick(bird)}
          >
            <h4>{bird.name}</h4>
            <p><strong>Scientific Name:</strong> {bird.sciName}</p>
          </div>
        ))}
      </div>

      {selectedBird && (
        <div className="modal-overlay" onClick={closePopup}>
          <div className="modal-content" onClick={(e) => e.stopPropagation()}>
            <button className="close-button" onClick={closePopup}>X</button>
            <h2>{selectedBird.name}</h2>
            <p><strong>Scientific Name:</strong> {selectedBird.sciName}</p>
            {selectedBird.imageUrl && (
              <img src={selectedBird.imageUrl} alt={selectedBird.name} className="bird-image" />
            )}
            <p>{selectedBird.description}</p>

            <h4>Distribution Map:</h4>
            {gbifId ? (
              <a
                href={`https://www.gbif.org/species/${gbifId}`}
                target="_blank"
                rel="noopener noreferrer"
                className="distribution-link"
              >
                View Distribution Map on GBIF
              </a>
            ) : (
              <p>Distribution map not available.</p>
            )}
          </div>
        </div>
      )}
    </div>
  );
};

export default SearchBird;
