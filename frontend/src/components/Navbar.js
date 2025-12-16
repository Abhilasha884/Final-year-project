import React from "react";
import { Link } from "react-router-dom";

export default function Navbar() {
  return (
    <nav className="navbar">
      <div className="nav-left">
        🎵 <span>MoodTune AI</span>
      </div>

      <div className="nav-right">
        <Link to="/analyze" className="btn-primary">Analyze</Link>
        <button className="btn-secondary">Dashboard</button>
      </div>
    </nav>
  );
}
