const API = import.meta.env.VITE_API_URL || "http://127.0.0.1:8000";

async function handle(res) {
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || `Request failed (${res.status})`);
  }
  return res.json();
}

export async function createAnalysis(oldFile, newFile) {
  const form = new FormData();
  form.append("old_image", oldFile);
  form.append("new_image", newFile);
  return handle(await fetch(`${API}/analyses`, { method: "POST", body: form }));
}

export async function runAnalysis(id) {
  return handle(await fetch(`${API}/analyses/${id}/run`, { method: "POST" }));
}

export async function getAnalysis(id) {
  return handle(await fetch(`${API}/analyses/${id}`));
}