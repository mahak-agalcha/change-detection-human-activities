import { API } from "./api";

export default function History({ history, stats, onOpen }) {
  return (
    <div className="panel">
      <h2>Analysis history</h2>

      {stats && (
        <div className="chips">
          <span className="chip">Analyses: {stats.total_analyses}</span>
          <span className="chip">Changes found: {stats.total_regions}</span>
          {Object.entries(stats.by_cause || {}).map(([k, v]) => (
            <span key={k} className={`badge b-${k}`}>
              {k}: {v}
            </span>
          ))}
        </div>
      )}

      {history.length === 0 ? (
        <p className="empty">No analyses yet. Upload two images to begin.</p>
      ) : (
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th>ID</th>
                <th>Location</th>
                <th>Image dates</th>
                <th>Status</th>
                <th>Changes</th>
                <th></th>
              </tr>
            </thead>
            <tbody>
              {history.map((a) => (
                <tr key={a.id}>
                  <td>{a.id}</td>
                  <td>{a.location_name || "-"}</td>
                  <td>
                    {a.old_date || "?"} to {a.new_date || "?"}
                  </td>
                  <td>{a.status}</td>
                  <td>{a.regions_count}</td>
                  <td>
                    <button className="btn-ghost" onClick={() => onOpen(a.id)}>
                      View
                    </button>{" "}
                    <a className="btn-ghost" href={`${API}/analyses/${a.id}/report`}>
                      PDF
                    </a>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}