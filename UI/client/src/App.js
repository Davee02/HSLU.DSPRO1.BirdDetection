import React from 'react';
import './App.css';
import logo from './img/logo_dark.png';  // Adjusted path to your image

function App() {
  return (
    <div className="App">
      <header className="App-header">
        <div className="App-logo">
          <img src={logo} alt="Logo" />  {/* Use the imported logo */}
        </div>
        <nav className="App-nav">
          <ul>
            <li><a href="#home">Home</a></li>
            <li><a href="#analyse">Analyse</a></li>
          </ul>
        </nav>
      </header>
      <main>
        <h1>Welcome to the Home Page</h1>
        <p>This is a basic homepage structure.</p>
      </main>
    </div>
  );
}

export default App;
