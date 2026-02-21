import { useEffect, useState } from "react";
import { Music, BarChart3, Smile, TrendingUp } from "lucide-react";
import axios from "axios";
import StatCard from "../components/statcard";
import RecentTable from "../components/recentTable";
import "./DashboardPage.css";

export default function DashboardPage() {
  const [stats, setStats] = useState({});
  const [recent, setRecent] = useState([]);

  useEffect(() => {
    axios
      .get("http://localhost:5000/dashboard")
      .then((res) => {
        setStats(res.data.stats || {});
        setRecent(res.data.recent || []);
      })
      .catch((err) => console.error(err));
  }, []);

  return (
    <div className="dashboard-content">
      <h1>Dashboard</h1>
      <p className="subtitle">
        Overview of your music analysis activity
      </p>

      <div className="stats-row">
        <StatCard
          icon={<Music size={22} color="#059669" />}
          title="Songs Analyzed"
          value={stats.songs || 0}
          color="#d1fae5"
        />

        <StatCard
          icon={<BarChart3 size={22} color="#dc2626" />}
          title="Genres Detected"
          value={stats.genres || 0}
          color="#fde2e4"
        />

        <StatCard
          icon={<Smile size={22} color="#7c3aed" />}
          title="Emotions Found"
          value={stats.emotions || 0}
          color="#ede9fe"
        />

        <StatCard
          icon={<TrendingUp size={22} color="#d97706" />}
          title="Accuracy Rate"
          value={`${stats.accuracy || 0}%`}
          color="#fef3c7"
        />
      </div>

      <RecentTable data={recent} />
    </div>
  );
}
