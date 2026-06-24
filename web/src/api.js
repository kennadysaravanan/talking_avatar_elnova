// Thin client for the orchestrator REST + WS API.
export const API_BASE = import.meta.env.VITE_API_BASE || "http://localhost:8080";

export async function createSession(file) {
  const fd = new FormData();
  fd.append("photo", file);
  const res = await fetch(`${API_BASE}/session`, { method: "POST", body: fd });
  if (!res.ok) {
    const detail = await res.json().catch(() => ({}));
    throw new Error(detail.detail || `session create failed (${res.status})`);
  }
  return res.json(); // { session_id, room, livekit_url, viewer_token }
}

export async function destroySession() {
  await fetch(`${API_BASE}/session`, { method: "DELETE" }).catch(() => {});
}

// Control WebSocket: send text in, receive state/turn events.
export function openControlSocket(sessionId, onEvent) {
  const wsBase = API_BASE.replace(/^http/, "ws");
  const ws = new WebSocket(`${wsBase}/ws/${sessionId}`);
  ws.onmessage = (e) => {
    try { onEvent(JSON.parse(e.data)); } catch { /* ignore */ }
  };
  return ws;
}
