import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import DashboardPage from "./pages/DashboardPage";
// import AnalyzePage from "./pages/AnalyzePage";
// import VisualizationsPage from "./pages/VisualizationsPage";

function App() {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<DashboardPage />} />
        {/* <Route path="/analyze" element={<AnalyzePage />} /> */}
        {/* <Route path="/visualizations" element={<VisualizationsPage />} /> */}
      </Routes>
    </Router>
  );
}

export default App;

