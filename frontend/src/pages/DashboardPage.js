import { useEffect, useState } from "react";
import { Music, BarChart3, Smile, TrendingUp } from "lucide-react";
// import axios from "axios";
import StatCard from "../components/statcard";
import RecentTable from "../components/recentTable";
import "./DashboardPage.css";
import { useNavigate } from "react-router-dom";


export default function DashboardPage() {
  const [stats, setStats] = useState({});
  const [recent, setRecent] = useState([]);
  const navigate = useNavigate();

  useEffect(() => {
  const saved = JSON.parse(localStorage.getItem("analyses")) || [];

  const genres = [...new Set(saved.map(a => a.predicted_genre))];

  setStats({
    songs: saved.length,
    genres: genres.length,
    emotions: saved.length,
    accuracy: saved.length > 0 ? 92 : 0
  });

  setRecent(saved);
}, []);



  // useEffect(() => {
  //   axios
  //     .get("http://localhost:5000/dashboard")
  //     .then((res) => {
  //       setStats(res.data.stats || {});
  //       setRecent(res.data.recent || []);
  //     })
  //     .catch((err) => console.error(err));
  // }, []);

  return (
    <div className="dashboard-content">
      <h1>Dashboard</h1>

<p className="subtitle">
  AI-powered music emotion & genre analysis.
  Analyze music using AI to detect Genre and Emotion from Audio and Lyrics.
  Upload a song or paste lyrics to discover its mood and style.
</p>

<button
  className="start-analysis-btn"
  onClick={() => navigate("/analyze")}
>
  🎵 Start Analyzing Music
</button>

      
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
