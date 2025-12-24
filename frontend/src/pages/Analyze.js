import React, { useState } from "react";
import "../App.css";

export default function Analyze() {
  const [lyrics, setLyrics] = useState("");
  const [file, setFile] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleAnalyze = async () => {
    if (!file && !lyrics) {
      alert("Please upload audio or enter lyrics");
      return;
    }

    const formData = new FormData();
    if (file) formData.append("file", file);
    formData.append("lyrics", lyrics);

    setLoading(true);

    try {
      const res = await fetch("http://127.0.0.1:5000/predict", {
        method: "POST",
        body: formData,
      });

      const data = await res.json();
      setResult(data);
    } catch (err) {
      console.error("Prediction error:", err);
      alert("Error analyzing song");
    }

    setLoading(false);
  };

  return (
    <div className="analyze-page">
      {/* ⬆️ Upload Card */}
      <div className="glass-upload-card">
        <h2>Upload Your Music</h2>

        <div className="upload-grid">
          {/* 🎤 Lyrics */}
          <div className="lyrics-box">
            <label>Lyrics Input</label>
            <textarea
              placeholder="Paste your song lyrics here..."
              value={lyrics}
              onChange={(e) => setLyrics(e.target.value)}
            />
          </div>

          {/* 🎵 Audio */}
          <div className="audio-box">
            <label>Audio Upload</label>
            <div className="drop-zone">
              <input
                type="file"
                accept="audio/*"
                onChange={(e) => setFile(e.target.files[0])}
              />
              <p>
                {file ? `🎵 ${file.name}` : "Drop audio file here or click to browse"}
              </p>
            </div>
          </div>
        </div>

        <button className="analyze-btn" onClick={handleAnalyze}>
          {loading ? "Analyzing..." : "Analyze Emotion & Genre"}
        </button>
      </div>

      {/* ⬇️ RESULTS */}
      {result && (
        <div className="results-section">
          {/* 🎶 GENRE */}
          <div className="result-card">
            <h3>🎵 Detected Genre</h3>
            <p className="genre-pill">
              {result.predicted_genre || "Unknown"}
            </p>
          </div>

          {/* 😊 EMOTION */}
          <div className="result-card">
            <h3>😊 Emotion Analysis</h3>
            <div className="emotion-values">
              <p>
                <strong>Valence:</strong>{" "}
                {result.valence !== undefined
                  ? result.valence.toFixed(2)
                  : "N/A"}
              </p>
              <p>
                <strong>Arousal:</strong>{" "}
                {result.arousal !== undefined
                  ? result.arousal.toFixed(2)
                  : "N/A"}
              </p>
            </div>
          </div>

          {/* 🎧 RECOMMENDATIONS (future-ready) */}
          {result.recommendations && (
            <div className="result-card">
              <h3>🎧 Recommended Songs</h3>
              <ul className="recommendation-list">
                {result.recommendations.map((song, idx) => (
                  <li key={idx}>{song}</li>
                ))}
              </ul>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
