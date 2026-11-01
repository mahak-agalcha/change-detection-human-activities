import { useEffect, useState } from "react";
import {
  createAnalysis,
  runAnalysis,
  getAnalysis,
  listAnalyses,
  getStats,
} from "./api";
import Results from "./Results";
import History from "./History";

export default function App() {
  const [oldFile, setOldFile] = useState(null);
  const [newFile, setNewFile] = useState(null);
  const [location, setLocation] = useState("");
  const [oldDate, setOldDate] = useState("");
  const [newDate, setNewDate] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [result, setResult] = useState(null);
  const [history, setHistory] = useState([]);
  const [stats, setStats] = useState(null);

  async function refresh() {
    try {
      setHistory(await listAnalyses());
      setStats(await getStats());
    } catch (e) {
      console.error(e);
    }
  }

  useEffect(() => {
    refresh();
  }, []);

  function scrollToResults() {
    setTimeout(
      () =>
        document
          .getElementById("results")
          ?.scrollIntoView({ behavior: "smooth" }),
      100
    );
  }

  async function handleSubmit() {
    setError("");
    setResult(null);
    if (!oldFile || !newFile) {
      setError("Please select both images.");
      return;
    }
    if (oldDate && newDate && oldDate > newDate) {
      setError("Old image date must not be after new image date.");
      return;
    }
    setLoading(true);
    try {
      const { id } = await createAnalysis(oldFile, newFile, {
        location_name: location.trim(),
        old_date: oldDate,
        new_date: newDate,
      });
      await runAnalysis(id);
      setResult(await getAnalysis(id));
      await refresh();
      scrollToResults();
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }

  async function openAnalysis(id) {
    setError("");
    try {
      setResult(await getAnalysis(id));
      scrollToResults();
    } catch (e) {
      setError(e.message);
    }
  }

  const preview = (f) => (f ? URL.createObjectURL(f) : null);

  return (
    <div style={{ maxWidth: 900, margin: "0 auto", padding: 24 }}>
      <h1 style={{ fontSize: 28, lineHeight: 1.3 }}>
        Change Detection Using Satellite Imagery
      </h1>

      <div style={{ display: "flex", gap: 24 }}>
        {[
          ["Old image", oldFile, setOldFile, oldDate, setOldDate],
          ["New image", newFile, setNewFile, newDate, setNewDate],
        ].map(([label, file, setFile, date, setDate]) => (
          <div key={label} style={{ flex: 1 }}>
            <label>{label} </label>
            <input
              type="file"
              accept=".png,.jpg,.jpeg,.tif,.tiff"
              onChange={(e) => setFile(e.target.files[0] || null)}
            />
            <div style={{ marginTop: 8 }}>
              <label>Capture date </label>
              <input
                type="date"
                value={date}
                onChange={(e) => setDate(e.target.value)}
              />
            </div>
            {file && (
              <img
                src={preview(file)}
                alt={label}
                style={{ width: "100%", marginTop: 8 }}
              />
            )}
          </div>
        ))}
      </div>

      <div style={{ marginTop: 16 }}>
        <label>Location </label>
        <input
          type="text"
          value={location}
          maxLength={120}
          placeholder="e.g. Indore, Madhya Pradesh"
          onChange={(e) => setLocation(e.target.value)}
          style={{ width: 300 }}
        />
      </div>

      <button onClick={handleSubmit} disabled={loading} style={{ marginTop: 16 }}>
        {loading ? "Analyzing..." : "Analyze"}
      </button>

      {error && <p style={{ color: "red" }}>{error}</p>}

      {result && <Results result={result} />}

      <History history={history} stats={stats} onOpen={openAnalysis} />
    </div>
  );
}