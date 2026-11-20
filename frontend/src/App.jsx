import { useEffect, useMemo, useState } from "react";
import {
  createAnalysis,
  runAnalysis,
  getAnalysis,
  listAnalyses,
  getStats,
} from "./api";
import Results from "./Results";
import History from "./History";

const APP_NAME = "GeoShift";

function UploadCard({ label, file, onFile, date, onDate }) {
  const [over, setOver] = useState(false);
  const url = useMemo(() => (file ? URL.createObjectURL(file) : null), [file]);

  function onDrop(e) {
    e.preventDefault();
    setOver(false);
    if (e.dataTransfer.files[0]) onFile(e.dataTransfer.files[0]);
  }

  return (
    <div>
      <label
        className={`drop ${over ? "over" : ""}`}
        onDragOver={(e) => {
          e.preventDefault();
          setOver(true);
        }}
        onDragLeave={() => setOver(false)}
        onDrop={onDrop}
      >
        {url ? (
          <img src={url} alt={label} />
        ) : (
          <>
            <span className="big">🛰️</span>
            <b>{label}</b>
            <span>Drag and drop or click to choose</span>
            <span style={{ fontSize: 12 }}>PNG, JPG or TIFF, up to 20 MB</span>
          </>
        )}
        <input
          type="file"
          accept=".png,.jpg,.jpeg,.tif,.tiff"
          onChange={(e) => onFile(e.target.files[0] || null)}
        />
      </label>
      <div className="field">
        Capture date
        <input type="date" value={date} onChange={(e) => onDate(e.target.value)} />
      </div>
    </div>
  );
}

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

  const step = result ? 3 : loading ? 2 : 1;
  const stepClass = (n) => (step === n ? "active" : step > n ? "done" : "");

  return (
    <div className="container">
      <div className="header">
        <div className="brand">
          <div className="logo">🌍</div>
          {APP_NAME}
        </div>
        <span className="pill">Satellite change detection</span>
      </div>
      <p className="tagline">
        See what changed between two satellite images, and be honest about why.
      </p>

      <div className="steps">
        <div className={`step ${stepClass(1)}`}>1. Upload</div>
        <div className={`step ${stepClass(2)}`}>2. Analyze</div>
        <div className={`step ${stepClass(3)}`}>3. Review</div>
      </div>

      <div className="panel">
        <div className="grid2">
          <UploadCard
            label="Old image"
            file={oldFile}
            onFile={setOldFile}
            date={oldDate}
            onDate={setOldDate}
          />
          <UploadCard
            label="New image"
            file={newFile}
            onFile={setNewFile}
            date={newDate}
            onDate={setNewDate}
          />
        </div>

        <div className="row">
          <div className="field">
            Location
            <input
              type="text"
              value={location}
              maxLength={120}
              placeholder="e.g. Indore, Madhya Pradesh"
              onChange={(e) => setLocation(e.target.value)}
            />
          </div>
          <button className="btn" onClick={handleSubmit} disabled={loading}>
            {loading && <span className="spinner" />}
            {loading ? "Analyzing..." : "Analyze changes"}
          </button>
        </div>

        {error && <div className="error">{error}</div>}
      </div>

      {result && <Results result={result} />}

      <History history={history} stats={stats} onOpen={openAnalysis} />
    </div>
  );
}