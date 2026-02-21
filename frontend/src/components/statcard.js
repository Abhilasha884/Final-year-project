import "./statcard.css";

export default function StatCard({ icon, title, value, color }) {
  return (
    <div className="stat-card">
      <div className="stat-left" style={{ background: color }}>
        {icon}
      </div>

      <div>
        <h2>{value}</h2>
        <p>{title}</p>
      </div>
    </div>
  );
}
