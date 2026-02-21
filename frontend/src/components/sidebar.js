import { Link, useLocation } from "react-router-dom";
import { LayoutDashboard, Search, BarChart2 } from "lucide-react";
import "./sidebar.css";

export default function Sidebar() {
  const location = useLocation();

  return (
    <div className="sidebar">
      <h2 className="logo">🎵 MoodTune</h2>

      <ul className="menu">
        <li className={location.pathname === "/" ? "active" : ""}>
          <LayoutDashboard size={18} />
          <Link to="/">Dashboard</Link>
        </li>

        <li className={location.pathname === "/analyze" ? "active" : ""}>
          <Search size={18} />
          <Link to="/analyze">Analyze</Link>
        </li>

        <li>
          <BarChart2 size={18} />
          <Link to="/visualizations">Visualizations</Link>
        </li>
      </ul>
    </div>
  );
}
