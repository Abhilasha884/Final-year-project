import { useEffect, useState } from "react";
import axios from "axios";
import Sidebar from "../components/sidebar";
import StatCard from "../components/statcard";
import RecentTable from "../components/recentTable";
import "./DashboardPage.css";

export default function DashboardPage() {
  const [stats, setStats] = useState({});
  const [recent, setRecent] = useState([]);

  useEffect(() => {
    axios.get("http://localhost:8000/dashboard")
      .then(res => {
        setStats(res.data.stats);
        setRecent(res.data.recent);
      });
  }, []);

  return (
    <div className="dashboard-layout">
      <Sidebar />

      <div className="dashboard-content">
        <h1>Dashboard</h1>

        <div className="stats-row">
          <StatCard title="Songs Analyzed" value={stats.songs} />
          <StatCard title="Genres Detected" value={stats.genres} />
          <StatCard title="Emotions Found" value={stats.emotions} />
          <StatCard title="Accuracy Rate" value={stats.accuracy + "%"} />
        </div>

        <RecentTable data={recent} />
      </div>
    </div>
  );
}
