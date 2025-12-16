import React, { useState } from "react";

export default function UploadBox({ setResult }) {
  const [file, setFile] = useState(null);

  const uploadSong = async () => {
    if (!file) return alert("Please select a file!");

    const formData = new FormData();
    formData.append("file", file);

    const res = await fetch("http://127.0.0.1:5000/predict", {
      method: "POST",
      body: formData,
    });

    const data = await res.json();
    setResult(data);
  };

  return (
    <div className="upload-box">
      <h2>Upload Song</h2>

      <input
        type="file"
        accept="audio/mp3"
        onChange={(e) => setFile(e.target.files[0])}
      />

      <button onClick={uploadSong}>Analyze Song</button>
    </div>
  );
}
