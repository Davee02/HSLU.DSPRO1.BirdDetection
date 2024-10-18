import React, { useState } from 'react';
import './App.css';
import logo from './img/logo_dark.png'; // Import the logo

// Import bird images statically
import sparrow from './img/sparrow.jpg';
import robin from './img/robin.jpg';
import bluetit from './img/bluetit.jpg';
import blackbird from './img/blackbird.jpg';
import swallow from './img/swallow.jpg';
import greattit from './img/greattit.jpg';
import pigeon from './img/pigeon.jpg';
import magpie from './img/magpie.jpg';
import starling from './img/starling.jpg';
import finch from './img/finch.jpg';

// Sample bird data with static image imports
const birds = [
  { name: 'Common Sparrow', img: sparrow, description: 'A small, plump, brown-grey bird common in urban areas.' },
  { name: 'European Robin', img: robin, description: 'Known for its bright red breast, often seen in gardens.' },
  { name: 'Eurasian Blue Tit', img: bluetit, description: 'A small bird with blue and yellow feathers, common in woodlands.' },
  { name: 'Common Blackbird', img: blackbird, description: 'A medium-sized bird with a black body and orange beak.' },
  { name: 'Barn Swallow', img: swallow, description: 'A migratory bird with long tail feathers, often seen swooping over fields.' },
  { name: 'Great Tit', img: greattit, description: 'A robust bird with a striking black head and white cheeks.' },
  { name: 'Wood Pigeon', img: pigeon, description: 'A large, grey bird with white neck patches, common in urban and rural areas.' },
  { name: 'Magpie', img: magpie, description: 'A large bird with distinctive black and white plumage and long tail.' },
  { name: 'Common Starling', img: starling, description: 'A medium-sized bird with glossy, speckled feathers and a sharp beak.' },
  { name: 'House Finch', img: finch, description: 'A small bird with a red-orange head and chest, commonly found in gardens.' }
];

function App() {
  const [activePage, setActivePage] = useState('home');
  const [selectedBird, setSelectedBird] = useState(null); // To handle the bird description overlay

  const handlePageChange = (page) => {
    setActivePage(page);
  };

  const handleBirdClick = (bird) => {
    setSelectedBird(bird);
  };

  const handleOverlayClose = () => {
    setSelectedBird(null); // Close the overlay when clicked
  };

  return (
    <div className="App">
      <header className="App-header">
        <div className="App-logo">
          <img src={logo} alt="Logo" />
        </div>
        <nav className="App-nav">
          <ul>
            <li>
              <a
                href="#home"
                onClick={() => handlePageChange('home')}
                className={activePage === 'home' ? 'active' : ''}
              >
                Home
              </a>
            </li>
            <li>
              <a
                href="#analyse"
                onClick={() => handlePageChange('analyse')}
                className={activePage === 'analyse' ? 'active' : ''}
              >
                Analyse
              </a>
            </li>
          </ul>
        </nav>
      </header>

      <main>
        {activePage === 'home' && (
          <section className="home-content">
            <h1>ChirpTrack: Bird Sound Classification Made Simple</h1>
            <p>
              Welcome to ChirpTrack, an innovative tool designed to classify bird species based on their unique sounds.
              Our mission is to harness the power of machine learning to make identifying birds easier than ever before.
              By simply uploading a short audio recording, users can receive an instant prediction of the bird species present in the recording.
            </p>
            <p>
              ChirpTrack goes beyond just audio classification—it's a step towards understanding and preserving biodiversity.
              Currently, our tool supports multi-class classification for various bird species, providing valuable insights for bird enthusiasts,
              researchers, and conservationists alike.
            </p>

            <h2>Future Vision: Real-Time Acoustic Monitoring</h2>
            <p>
              Looking ahead, ChirpTrack aims to evolve into a real-time acoustic monitoring system.
              This system would use remote microphones in natural habitats, such as forests, to automatically detect and classify bird species.
              By monitoring these environments, we can track biodiversity, observe shifts in species populations, and assess overall ecosystem health.
            </p>
            <p>
              Whether you're curious about the birds in your backyard or conducting large-scale ecological studies, ChirpTrack is your go-to tool for bird sound classification.
              Join us in exploring the world of bird sounds and making strides toward conserving the natural world.
            </p>

            {/* Bird slider section */}
            <div className="bird-slider">
              {birds.map((bird, index) => (
                <div key={index} className="bird-card" onClick={() => handleBirdClick(bird)}>
                  <img src={bird.img} alt={bird.name} />
                  <div className="bird-name">{bird.name}</div>
                </div>
              ))}
            </div>

            {/* Overlay for bird description */}
            {selectedBird && (
              <div className="overlay" onClick={handleOverlayClose}>
                <div className="overlay-content">
                  <h3>{selectedBird.name}</h3>
                  <p>{selectedBird.description}</p>
                </div>
              </div>
            )}
          </section>
        )}

        {activePage === 'analyse' && (
          <div>
            <h1>Analyse Page</h1>
            <p>Analyse bird sounds here.</p>
          </div>
        )}
      </main>
    </div>
  );
}

export default App;
