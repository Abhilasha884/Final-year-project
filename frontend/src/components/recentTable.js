import "./recentTable.css";

export default function RecentTable({ data }) {
  return (
    <div className="recent-container">
      <h3>Recent Analyses</h3>

      <table>
        <thead>
          <tr>
            <th>Song</th>
            <th>Artist</th>
            <th>Genre</th>
            <th>Emotion</th>
            <th>Confidence</th>
          </tr>
        </thead>

        <tbody>
          {data.map((item, index) => (
            <tr key={index}>
              <td>{item.song}</td>
              <td>{item.artist}</td>
              <td>
                <span className="tag genre">{item.genre}</span>
              </td>
              <td>
                <span className="tag emotion">{item.emotion}</span>
              </td>
              <td>
                <div className="progress">
                  <div
                    className="progress-bar"
                    style={{ width: item.confidence + "%" }}
                  ></div>
                </div>
                {item.confidence}%
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
