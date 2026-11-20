import { API } from "./api";
import ChangeMap from "./ChangeMap";
import CompareSlider from "./CompareSlider";

const PLACEHOLDER_MODEL = true; // set to false once the real model is connected

const CAUSE_TEXT = {
  human: "Likely human-related",
  natural: "Likely natural / disaster-related",
  unknown: "Unknown (insufficient evidence)",
};
const TYPE_ICON = { building: "🏢", road: "🛣️", vehicle: "🚗" };

function countBy(items, key) {
  const out = {};
  items.forEach((i) => (out[i[key]] = (out[i[key]] || 0) + 1));
  return out;
}

export default function Results({ result }) {
  const regions = result.regions;
  const byType = countBy(regions, "object_type");

  return (
    <div id="results" className="panel">
      <h2>Results</h2>
      <div className="chips">
        <span className="chip">📍 {result.location_name || "Location not provided"}</span>
        <span className="chip">Old: {result.old_date || "n/a"}</span>
        <span className="chip">New: {result.new_date || "n/a"}</span>
        <span className="chip">Status: {result.status}</span>
      </div>

      {PLACEHOLDER_MODEL && (
        <div className="banner">
          Prototype output: the detection below comes from a placeholder, not
          the trained model. Do not treat it as a real finding.
        </div>
      )}

      <a className="btn-ghost" href={`${API}/analyses/${result.id}/report`}>
        Download PDF report
      </a>

      <div className="stats">
        <div className="stat">
          <b>{regions.length}</b>
          <span>Total changes</span>
        </div>
        {Object.entries(byType).map(([k, v]) => (
          <div className="stat" key={k}>
            <b>{v}</b>
            <span>
              {TYPE_ICON[k] || "•"} {k}
            </span>
          </div>
        ))}
      </div>

      <h3>Compare images</h3>
      <p className="note" style={{ marginTop: 0 }}>
        Drag across the image to compare before and after.
      </p>
      <CompareSlider
        before={`${API}${result.old_image}`}
        after={`${API}${result.new_image}`}
      />

      <h3 style={{ marginTop: 20 }}>Map</h3>
      <ChangeMap regions={regions} />

      <h3 style={{ marginTop: 20 }}>Detected changes</h3>
      {regions.length === 0 ? (
        <p className="empty">No changes detected.</p>
      ) : (
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th>#</th>
                <th>Object</th>
                <th>Cause analysis</th>
                <th>Confidence</th>
              </tr>
            </thead>
            <tbody>
              {regions.map((r, i) => {
                const pct = Math.round(r.confidence * 100);
                return (
                  <tr key={r.id}>
                    <td>{i + 1}</td>
                    <td>
                      {TYPE_ICON[r.object_type] || "•"} {r.object_type}
                    </td>
                    <td>
                      <span className={`badge b-${r.cause_label}`}>
                        {CAUSE_TEXT[r.cause_label] || r.cause_label}
                      </span>
                    </td>
                    <td>
                      <span className="bar">
                        <span style={{ width: `${pct}%` }} />
                      </span>
                      {pct}%
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}

      <p className="note">
        Images show what changed, not why. "Unknown" means there is not enough
        contextual evidence to decide the cause.
      </p>
    </div>
  );
}