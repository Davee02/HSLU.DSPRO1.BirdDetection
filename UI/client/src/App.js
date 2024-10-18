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
          <div>
            <h1>Welcome to the Home Page</h1>
            <p>This is the Home Page content.</p>
          </div>
        )}
        {activePage === 'analyse' && (
          <div>
            <h1>Welcome to the Analyse Page</h1>
            <p>This is the Analyse Page content.</p>
          </div>
        )}
      </main>
    </div>
  );
}

export default App;
