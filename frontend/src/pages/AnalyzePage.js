import React, { useState } from "react";
import { predictSong } from "../services/api";
import "./AnalyzePage.css";

function AnalyzePage() {
  const [lyrics, setLyrics] = useState("");
  const [file, setFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);

  const handleAnalyze = async () => {
    if (!file && !lyrics) {
      alert("Please upload audio or paste lyrics.");
      return;
    }

    const formData = new FormData();
    if (file) formData.append("file", file);
    formData.append("lyrics", lyrics);

    try {
      setLoading(true);
      const response = await predictSong(formData);
      setResult(response.data);
      setLoading(false);
    } catch (err) {
      console.error(err);
      setLoading(false);
    }
  };

  return (
    <div className="analyze-wrapper">
      <div className="analyze-header">
        <h1>Analyze Music</h1>
        <p>Upload lyrics or audio to classify genre & emotion</p>
      </div>

      <div className="analyze-grid">
        {/* Lyrics Card */}
        <div className="analyze-card">
          <h3>📄 Paste Lyrics</h3>
          <textarea
            placeholder="Paste song lyrics here..."
            value={lyrics}
            onChange={(e) => setLyrics(e.target.value)}
          />
        </div>

        {/* Upload Card */}
        <div className="analyze-card">
          <h3>🎵 Upload Audio</h3>

          <label className="upload-box">
            <input
              type="file"
              accept=".mp3,.wav,.flac"
              onChange={(e) => setFile(e.target.files[0])}
              hidden
            />
            <div>
              {file ? file.name : "Click to upload MP3, WAV, or FLAC"}
            </div>
          </label>
        </div>
      </div>

      <div className="analyze-button-wrapper">
        <button onClick={handleAnalyze}>
          {loading ? "Analyzing..." : "Analyze Music"}
        </button>
      </div>

      {result && (
        <div className="result-section">
          <h2>Prediction Result</h2>
          <p><strong>Genre:</strong> {result.predicted_genre}</p>
          <p><strong>Valence:</strong> {result.valence}</p>
          <p><strong>Arousal:</strong> {result.arousal}</p>
        </div>
      )}
    </div>
  );
}

export default AnalyzePage;