import React, { useState } from 'react';
import './App.css';
import logo from './img/logo_dark.png';
import AnalysePage from './AnalysePage';

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

const birds = [
  {
    name: 'Common Sparrow',
    img: sparrow,
    description: 'A small, plump, brown-grey bird common in urban areas.',
    info: 'Sparrows are incredibly adaptive and can live in a variety of environments, from urban areas to the countryside.',
    funFact: 'Sparrows are social birds that often live in colonies, and they take dust baths instead of water baths!'
  },
  {
    name: 'European Robin',
    img: robin,
    description: 'Known for its bright red breast, often seen in gardens.',
    info: 'Robins are fiercely territorial birds and are known to defend their patch aggressively, even against much larger birds.',
    funFact: 'In the UK, the Robin is often associated with Christmas and appears on many holiday cards!'
  },
  {
    name: 'Eurasian Blue Tit',
    img: bluetit,
    description: 'A small bird with blue and yellow feathers, common in woodlands.',
    info: 'Blue Tits are intelligent birds known for their acrobatic skills, often seen hanging upside down to feed on insects.',
    funFact: 'Blue Tits were once famous for learning to peck through milk bottle tops to steal cream!'
  },
  {
    name: 'Common Blackbird',
    img: blackbird,
    description: 'A medium-sized bird with a black body and orange beak.',
    info: 'Male blackbirds are entirely black, while females are brown and streaked. They are excellent singers, with melodious songs.',
    funFact: 'The song of the male Blackbird is so popular that it is often heard in television programs and films.'
  },
  {
    name: 'Barn Swallow',
    img: swallow,
    description: 'A migratory bird with long tail feathers, often seen swooping over fields.',
    info: 'Swallows are fast and agile fliers, often seen darting through the air as they catch insects.',
    funFact: 'Swallows are famous for their long migrations and can fly up to 200 miles a day during their travels!'
  },
  {
    name: 'Great Tit',
    img: greattit,
    description: 'A robust bird with a striking black head and white cheeks.',
    info: 'Great Tits are the largest of the UK tit species, known for their bold behavior and strong voice.',
    funFact: 'Great Tits are known for their ability to adapt their songs in noisy urban environments!'
  },
  {
    name: 'Wood Pigeon',
    img: pigeon,
    description: 'A large, grey bird with white neck patches, common in urban and rural areas.',
    info: 'Wood Pigeons are the largest and most widespread pigeon species in Europe, often found in parks and gardens.',
    funFact: 'Wood Pigeons can hold more water in their throats than most birds, allowing them to drink without raising their heads.'
  },
  {
    name: 'Magpie',
    img: magpie,
    description: 'A large bird with distinctive black and white plumage and long tail.',
    info: 'Magpies are known for their intelligence and curiosity, often collecting shiny objects.',
    funFact: 'Magpies are one of the few animal species that can recognize themselves in a mirror!'
  },
  {
    name: 'Common Starling',
    img: starling,
    description: 'A medium-sized bird with glossy, speckled feathers and a sharp beak.',
    info: 'Starlings are known for their ability to mimic sounds, including human speech and other birds.',
    funFact: 'Starlings create stunning aerial displays called "murmurations," where thousands of birds fly in synchronized patterns.'
  },
  {
    name: 'House Finch',
    img: finch,
    description: 'A small bird with a red-orange head and chest, commonly found in gardens.',
    info: 'House Finches are social birds and are often seen feeding in flocks.',
    funFact: 'The bright red color of the male House Finch comes from pigments in the food they eat!'
  }
];

function App() {
  const [activePage, setActivePage] = useState('home');
  const [currentIndex, setCurrentIndex] = useState(0); // Slider index
  const [flippedCards, setFlippedCards] = useState([]); // Track flipped cards

  const handlePageChange = (page) => {
    setActivePage(page);
  };

  const handleNext = () => {
    if (currentIndex < birds.length - 2) {
      setCurrentIndex(currentIndex + 1);
    }
  };

  const handlePrevious = () => {
    if (currentIndex > 0) {
      setCurrentIndex(currentIndex - 1);
    }
  };

  const handleFlip = (index) => {
    if (flippedCards.includes(index)) {
      setFlippedCards(flippedCards.filter((i) => i !== index));
    } else {
      setFlippedCards([...flippedCards, index]);
    }
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

            {/* Slider Title and Description */}
            <h2>Most Common Birds that Could Live in Your Backyard</h2>
            <p>These are some of the most common birds found in European backyards. Click on the cards to learn more about them!</p>

            {/* Bird slider section */}
            <div className="slider-container">
              <button className="slider-arrow left-arrow" onClick={handlePrevious} disabled={currentIndex === 0}>
                &#9664;
              </button>

              <div className="bird-slider">
                {birds.slice(currentIndex, currentIndex + 2).map((bird, index) => (
                  <div
                    key={index + currentIndex}
                    className={`bird-card ${flippedCards.includes(index + currentIndex) ? 'flipped' : ''}`}
                    onClick={() => handleFlip(index + currentIndex)}
                  >
                    <div className="card-front">
                      <img src={bird.img} alt={bird.name} />
                      <div className="bird-name">{bird.name}</div>
                    </div>
                    <div className="card-back">
                      <h3>{bird.name}</h3>
                      <p>{bird.description}</p>
                      <p><strong>More info:</strong> {bird.info}</p>
                      <p><strong>Fun fact:</strong> {bird.funFact}</p>
                    </div>
                  </div>
                ))}
              </div>

              <button
                className="slider-arrow right-arrow"
                onClick={handleNext}
                disabled={currentIndex >= birds.length - 2}
              >
                &#9654;
              </button>
            </div>
          </section>
        )}
        {activePage === 'analyse' && (
          <AnalysePage />
        )}
      </main>
    </div>
  );
}

export default App;
