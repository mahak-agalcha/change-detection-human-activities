import { API } from "./api";
import ChangeMap from "./ChangeMap";

const CAUSE_TEXT = {
  human: "Likely human-related",
  natural: "Likely natural / disaster-related",
  unknown: "Unknown (insufficient evidence)",
};

function countBy(items, key) {
  const out = {};
  items.forEach((i) => (out[i[key]] = (out[i[key]] || 0) + 1));
  return out;
}

const card = {
  flex: 1, padding: 12, border: "1px solid #ccc",
  borderRadius: 8, textAlign: "center",
};

export default function Results({ result }) {
  const regions = result.regions;
  const byType = countBy(regions, "object_type");
  const byCause = countBy(regions, "cause_label");

  return (
    <div style={{ marginTop: 24 }}>
      <h2>Results (status: {result.status})</h2>

      <div style={{ display: "flex", gap: 12, marginBottom: 16 }}>
        <div style={card}><b>{regions.length}</b><br />Total changes</div>
        {Object.entries(byType).map(([k, v]) => (
          <div key={k} style={card}><b>{v}</b><br />{k}</div>
        ))}
      </div>

      <div style={{ display: "flex", gap: 12 }}>
        <div style={{ flex: 1 }}>
          <p>Old image</p>
          <img src={`${API}${result.old_image}`} alt="old" style={{ width: "100%" }} />
        </div>
        <div style={{ flex: 1 }}>
          <p>New image</p>
          <img src={`${API}${result.new_image}`} alt="new" style={{ width: "100%" }} />
        </div>
      </div>
      <h3>Map</h3>
      <ChangeMap regions={regions} />

      <h3>Detected changes</h3>
      {regions.length === 0 ? (
        <p>No changes detected.</p>
      ) : (
        <table border="1" cellPadding="6" style={{ borderCollapse: "collapse", width: "100%" }}>
          <thead>
            <tr><th>#</th><th>Object</th><th>Cause analysis</th><th>Confidence</th></tr>
          </thead>
          <tbody>
            {regions.map((r, i) => (
              <tr key={r.id}>
                <td>{i + 1}</td>
                <td>{r.object_type}</td>
                <td>{CAUSE_TEXT[r.cause_label] || r.cause_label}</td>
                <td>{Math.round(r.confidence * 100)}%</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}

      <p style={{ fontSize: 12, color: "#666" }}>
        Images show what changed, not why. "Unknown" means there is not enough
        contextual evidence to decide the cause.
        {Object.keys(byCause).length > 0 && " Causes: " + JSON.stringify(byCause)}
      </p>
    </div>
  );
}