import React, { useState } from "react";
import "../App.css";

export default function UploadBox({ setResult }) {
  const [file, setFile] = useState(null);
  const [loading, setLoading] = useState(false);

  const uploadSong = async () => {
    if (!file) {
      alert("Please select a song file");
      return;
    }

    const formData = new FormData();
    formData.append("file", file);

    try {
      setLoading(true);
      const res = await fetch("http://127.0.0.1:5000/predict", {
        method: "POST",
        body: formData,
      });

      const data = await res.json();
      setResult(data);
    } catch (err) {
      console.error(err);
      alert("Error analyzing song");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="upload-box">
      <h2>Upload Song</h2>

      <input
        type="file"
        accept=".mp3,.wav"
        onChange={(e) => setFile(e.target.files[0])}
        className="file-input"
      />

      <button onClick={uploadSong} disabled={loading}>
        {loading ? "Analyzing..." : "Analyze Song"}
      </button>
    </div>
  );
}
