import React from "react";

export default function Recommendations({ songs }) {
  if (!songs || songs.length === 0) {
    return <p style={{ color: "#aaa" }}>No songs found for this genre</p>;
  }

  return (
    <div className="recommendations">
      {songs.map((song, index) => (
        <div key={index} className="song-card">
          <h4>{song.song_id}</h4>
          <p>Genre: {song.genre}</p>
          <p>Valence: {song.valence.toFixed(2)}</p>
          <p>Arousal: {song.arousal.toFixed(2)}</p>
        </div>
      ))}
    </div>
  );
}
