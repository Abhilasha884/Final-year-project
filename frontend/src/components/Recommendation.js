import React from "react";

export default function Recommendations({ songs }) {
  return (
    <div className="result-card">
      <h3>Recommended Songs</h3>
      <ul style={{ marginTop: "15px" }}>
        {songs.map((song, i) => (
          <li key={i}>{song}</li>
        ))}
      </ul>
    </div>
  );
}
