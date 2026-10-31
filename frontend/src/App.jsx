import { useState } from "react";
import { createAnalysis, runAnalysis, getAnalysis } from "./api";
import Results from "./Results";

export default function App() {
  const [oldFile, setOldFile] = useState(null);
  const [newFile, setNewFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [result, setResult] = useState(null);

  async function handleSubmit() {
    setError("");
    setResult(null);
    if (!oldFile || !newFile) {
      setError("Please select both images.");
      return;
    }
    setLoading(true);
    try {
      const { id } = await createAnalysis(oldFile, newFile);
      await runAnalysis(id);
      setResult(await getAnalysis(id));
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
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
          ["Old image", oldFile, setOldFile],
          ["New image", newFile, setNewFile],
        ].map(([label, file, setFile]) => (
          <div key={label} style={{ flex: 1 }}>
            <label>{label} </label>
            <input
              type="file"
              accept=".png,.jpg,.jpeg,.tif,.tiff"
              onChange={(e) => setFile(e.target.files[0] || null)}
            />
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

      <button onClick={handleSubmit} disabled={loading} style={{ marginTop: 16 }}>
        {loading ? "Analyzing..." : "Analyze"}
      </button>

      {error && <p style={{ color: "red" }}>{error}</p>}

      {result && <Results result={result} />}
    </div>
  );
}