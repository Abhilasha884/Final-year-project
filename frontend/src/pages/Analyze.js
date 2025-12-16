import React, { useState } from "react";
import UploadBox from "../components/UploadBox";
import EmotionChart from "../components/EmotionChart";
import Recommendations from "../components/Recommendation";
import "../App.css";

export default function Analyze() {
  const [result, setResult] = useState(null);

  return (
    <div className="analyze-page">
      <div className="glass-card">
        <UploadBox setResult={setResult} />
      </div>

      {result && (
        <div className="results-section">
          <EmotionChart emotions={result.emotions} />
          <Recommendations songs={result.recommendations} />
        </div>
      )}
    </div>
  );
}
