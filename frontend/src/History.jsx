import { API } from "./api";

export default function History({ history, stats, onOpen }) {
  return (
    <div style={{ marginTop: 40 }}>
      <h2>Analysis history</h2>

      {stats && (
        <p>
          Total analyses: <b>{stats.total_analyses}</b> | Changes found:{" "}
          <b>{stats.total_regions}</b> | By object:{" "}
          {JSON.stringify(stats.by_object_type)} | By cause:{" "}
          {JSON.stringify(stats.by_cause)}
        </p>
      )}

      {history.length === 0 ? (
        <p>No analyses yet.</p>
      ) : (
        <table
          border="1"
          cellPadding="6"
          style={{ borderCollapse: "collapse", width: "100%" }}
        >
          <thead>
            <tr>
              <th>ID</th>
              <th>Date</th>
              <th>Status</th>
              <th>Changes</th>
              <th>Actions</th>
            </tr>
          </thead>
          <tbody>
            {history.map((a) => (
              <tr key={a.id}>
                <td>{a.id}</td>
                <td>{new Date(a.created_at).toLocaleString()}</td>
                <td>{a.status}</td>
                <td>{a.regions_count}</td>
                <td>
                  <button onClick={() => onOpen(a.id)}>View</button>{" "}
                  <a href={`${API}/analyses/${a.id}/report`}>PDF</a>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  );
}