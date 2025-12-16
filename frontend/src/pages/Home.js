import React from "react";
import Navbar from "../components/Navbar";
import "../App.css";

export default function Home() {
  return (
    <div className="home-page">
      <Navbar />

      <section className="hero-section">
        <h1>
          Discover Music Through <br />
          <span>Your Emotions</span>
        </h1>

        <p>
          Upload lyrics and audio to unlock emotion-based genre classification
          and personalized song recommendations powered by multimodal AI.
        </p>

        <div className="feature-cards">
          <div className="feature-card">
            🎵
            <h3>Multi-input Analysis</h3>
            <p>Lyrics + Audio</p>
          </div>

          <div className="feature-card">
            🧠
            <h3>Emotion Detection</h3>
            <p>AI-Powered Insights</p>
          </div>

          <div className="feature-card">
            ✨
            <h3>Smart Recommendations</h3>
            <p>Personalized Songs</p>
          </div>
        </div>
      </section>
    </div>
  );
}
