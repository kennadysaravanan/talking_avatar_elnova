import React, { useCallback, useEffect, useRef, useState } from "react";
import { Room, RoomEvent, Track } from "livekit-client";
import { createSession, destroySession, openControlSocket } from "./api.js";

// Phases: idle-upload -> warming (loading overlay) -> live
export default function App() {
  const [phase, setPhase] = useState("upload");
  const [error, setError] = useState("");
  const [sessionId, setSessionId] = useState(null);
  const [speaking, setSpeaking] = useState(false);
  const [text, setText] = useState("");
  const videoRef = useRef(null);
  const roomRef = useRef(null);
  const wsRef = useRef(null);

  const attachTrack = useCallback((track) => {
    if (track.kind === Track.Kind.Video && videoRef.current) {
      track.attach(videoRef.current);
    }
  }, []);

  const start = useCallback(async (file) => {
    setError("");
    setPhase("warming"); // shows "Preparing your avatar…" overlay
    try {
      const info = await createSession(file);
      setSessionId(info.session_id);

      // Join the LiveKit room (subscribe-only) and render the single avatar track.
      const room = new Room({ adaptiveStream: true, dynacast: true });
      roomRef.current = room;
      room.on(RoomEvent.TrackSubscribed, (track) => attachTrack(track));
      await room.connect(info.livekit_url, info.viewer_token);

      // Control socket for text in / state out.
      const ws = openControlSocket(info.session_id, (evt) => {
        if (evt.type === "turn_started") setSpeaking(true);
        if (evt.type === "state" && evt.state === "idle") setSpeaking(false);
      });
      wsRef.current = ws;

      setPhase("live");
    } catch (e) {
      setError(String(e.message || e));
      setPhase("upload");
    }
  }, [attachTrack]);

  const send = useCallback(() => {
    const t = text.trim();
    if (!t || !wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) return;
    // Sending while speaking == interruption (handled server-side, text mode).
    wsRef.current.send(JSON.stringify({ type: "text", text: t }));
    setSpeaking(true);
    setText("");
  }, [text]);

  useEffect(() => () => {
    wsRef.current?.close();
    roomRef.current?.disconnect();
    destroySession();
  }, []);

  return (
    <div style={{ maxWidth: 760, margin: "0 auto", padding: 24 }}>
      <h2>Real-Time Talking Avatar</h2>

      {phase === "upload" && (
        <div>
          <p>Upload a realistic 2D photo to begin.</p>
          <input type="file" accept="image/*"
                 onChange={(e) => e.target.files?.[0] && start(e.target.files[0])} />
          {error && <p style={{ color: "#ff6b6b" }}>{error}</p>}
        </div>
      )}

      {(phase === "warming" || phase === "live") && (
        <div style={{ position: "relative", aspectRatio: "720 / 400", background: "#000",
                      borderRadius: 12, overflow: "hidden" }}>
          <video ref={videoRef} autoPlay playsInline muted={false}
                 style={{ width: "100%", height: "100%", objectFit: "cover" }} />
          {phase === "warming" && (
            <div style={{ position: "absolute", inset: 0, display: "flex",
                          alignItems: "center", justifyContent: "center",
                          background: "rgba(0,0,0,0.6)", fontSize: 18 }}>
              Preparing your avatar…
            </div>
          )}
        </div>
      )}

      {phase === "live" && (
        <div style={{ marginTop: 16, display: "flex", gap: 8 }}>
          <input
            style={{ flex: 1, padding: 10, borderRadius: 8, border: "1px solid #333",
                     background: "#15181c", color: "#e8eaed" }}
            placeholder={speaking ? "Type to interrupt…" : "Ask something…"}
            value={text}
            onChange={(e) => setText(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && send()}
          />
          <button onClick={send} style={{ padding: "10px 18px", borderRadius: 8,
                   border: "none", background: "#3b82f6", color: "#fff", cursor: "pointer" }}>
            Send
          </button>
        </div>
      )}
      {phase === "live" && (
        <p style={{ color: "#8b95a1", fontSize: 13, marginTop: 8 }}>
          {speaking ? "Avatar is speaking — sending again interrupts it." : "Idle."}
        </p>
      )}
    </div>
  );
}
