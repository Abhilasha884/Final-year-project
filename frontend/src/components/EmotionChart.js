import React from "react";

export default function EmotionChart({ emotions }) {
  return (
    <div className="result-card">
      <h3>Detected Emotions</h3>
      <ul style={{ marginTop: "15px" }}>
        {Object.entries(emotions).map(([emotion, value]) => (
          <li key={emotion}>
            {emotion}: {(value * 100).toFixed(1)}%
          </li>
        ))}
      </ul>
    </div>
  );
}
