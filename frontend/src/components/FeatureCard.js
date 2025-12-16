import React from "react";

function UploadPanel({ setData, setAudioURL }) {
  const handleUpload = async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    setAudioURL(URL.createObjectURL(file));

    const formData = new FormData();
    formData.append("file", file);

    const res = await fetch("http://127.0.0.1:8000/analyze", {
      method: "POST",
      body: formData,
    });

    const json = await res.json();
    setData(json);
  };

  return (
    <div className="box">
      <h1>Song Analyzer</h1>
      <input type="file" accept="audio/*" onChange={handleUpload} />
    </div>
  );
}

export default UploadPanel;
