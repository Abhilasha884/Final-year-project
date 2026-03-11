import { useEffect, useState } from "react";
import {
  PieChart,
  Pie,
  Cell,
  Tooltip,
  ResponsiveContainer,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
} from "recharts";
import "./VisualizationPage.css";

const COLORS = [
  "#2a9d8f",
  "#e76f51",
  "#6a4c93",
  "#f4a261",
  "#4ecdc4",
  "#f28482",
  "#a29bfe",
];

export default function VisualizationPage() {

  const [genreData, setGenreData] = useState([]);
  const [emotionData, setEmotionData] = useState([]);

  useEffect(() => {

  fetch("http://localhost:5000/visualization-data")
    .then((res) => res.json())
    .then((data) => {

      setGenreData(data.genres);
      setEmotionData(data.emotions);

    })
    .catch((err) => console.error(err));

  }, []);


  

  return (
    <div className="visualization-content">

      <h1>Visualizations</h1>
      <p className="subtitle">
        Explore patterns in your analyzed music
      </p>

      <div className="chart-grid">

        {/* Genre Donut Chart */}

        <div className="chart-card">

          <h3>Genre Distribution</h3>

          <ResponsiveContainer width="100%" height={300}>

            <PieChart>

              <Pie
                data={genreData}
                dataKey="value"
                nameKey="name"
                innerRadius={70}
                outerRadius={100}
                paddingAngle={3}
              >

                {genreData.map((entry, index) => (
                  <Cell key={index} fill={COLORS[index % COLORS.length]} />
                ))}

              </Pie>

              <Tooltip />

            </PieChart>

          </ResponsiveContainer>

        </div>

        {/* Emotion Bar Chart */}

        <div className="chart-card">

          <h3>Emotion Frequency</h3>

          <ResponsiveContainer width="100%" height={300}>

            <BarChart data={emotionData}>

              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="name" />
              <YAxis />
              <Tooltip />

              <Bar dataKey="value">

                {emotionData.map((entry, index) => (
                  <Cell key={index} fill={COLORS[index % COLORS.length]} />
                ))}

              </Bar>

            </BarChart>

          </ResponsiveContainer>

        </div>

      </div>

    </div>
  );
}