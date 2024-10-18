import React, { useState } from 'react';
import './App.css';
import logo from './img/logo_dark.png';

function App() {
  const [activePage, setActivePage] = useState('home');

  const handlePageChange = (page) => {
    setActivePage(page);
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
