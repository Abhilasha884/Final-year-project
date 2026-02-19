import "./sidebar.css";

export default function Sidebar() {
  return (
    <div className="sidebar">
      <h2 className="logo">🎵 MoodTune</h2>

      <ul className="menu">
        <li className="active">Dashboard</li>
        <li>Analyze</li>
        <li>Visualizations</li>
      </ul>
    </div>
  );
}
