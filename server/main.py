"""
main.py  —  FastAPI orchestrator.

Endpoints:
  POST   /session            multipart photo upload -> create + warm session ->
                             {session_id, room, livekit_url, viewer_token}
  WS     /ws/{session_id}    control channel: {"type":"text","text":...} drives a turn;
                             {"type":"state",...} events flow back to the client.
  DELETE /session            tear down the active session
  GET    /healthz            liveness

The browser joins the LiveKit room with viewer_token (subscribe-only) and renders the single
continuous avatar video track. Text in / state out go over the WS; video comes over WebRTC.

SINGLE SESSION PER POD (constraint #6): POST /session 409s if one is already active.
"""

from __future__ import annotations

import logging
import os
import tempfile

from fastapi import FastAPI, File, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from livekit import api

from .session import SessionManager

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("main")

app = FastAPI(title="LiveAvatar Orchestrator")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

manager = SessionManager()
UPLOAD_DIR = os.environ.get("AVATAR_UPLOAD_DIR", os.path.join(tempfile.gettempdir(), "avatar_uploads"))
os.makedirs(UPLOAD_DIR, exist_ok=True)


def _viewer_token(room: str) -> str:
    return (
        api.AccessToken(os.environ["LIVEKIT_API_KEY"], os.environ["LIVEKIT_API_SECRET"])
        .with_identity("viewer")
        .with_grants(api.VideoGrants(room_join=True, room=room, can_subscribe=True,
                                     can_publish=False))
        .to_jwt()
    )


@app.get("/healthz")
async def healthz():
    s = manager.current
    return {"ok": True, "active_session": s.session_id if s else None}


@app.post("/session")
async def create_session(photo: UploadFile = File(...)):
    if manager.current is not None:
        raise HTTPException(status_code=409, detail="A conversation is already active (one per pod).")
    # store the uploaded reference photo
    ext = os.path.splitext(photo.filename or "")[1] or ".jpg"
    path = os.path.join(UPLOAD_DIR, f"ref_{os.urandom(4).hex()}{ext}")
    with open(path, "wb") as fh:
        fh.write(await photo.read())
    logger.info("stored reference photo -> %s", path)

    session = await manager.create(path)
    ok = await session.warm_up()
    if not ok:
        await manager.destroy()
        raise HTTPException(status_code=504, detail="Avatar warm-up timed out (no first frame).")

    return {
        "session_id": session.session_id,
        "room": session.room_name,
        "livekit_url": os.environ["LIVEKIT_URL"],
        "viewer_token": _viewer_token(session.room_name),
        "ref_image_note": (
            "Prototype warms the resident worker at launch with REF_IMAGE. Hot per-session ref "
            "swap without a worker restart is future work (see README)."
        ),
    }


@app.delete("/session")
async def delete_session():
    await manager.destroy()
    return {"ok": True}


@app.websocket("/ws/{session_id}")
async def ws_control(ws: WebSocket, session_id: str):
    await ws.accept()
    session = manager.current
    if session is None or session.session_id != session_id:
        await ws.send_json({"type": "error", "error": "no such session"})
        await ws.close()
        return
    await ws.send_json({"type": "state", "state": session.state})
    try:
        while True:
            msg = await ws.receive_json()
            if msg.get("type") == "text":
                text = (msg.get("text") or "").strip()
                if not text:
                    continue
                # Interruption is handled inside on_user_text (cancels any in-flight turn).
                interrupted = session.turns.speaking
                await session.turns.on_user_text(text)
                await ws.send_json({"type": "turn_started", "interrupted": interrupted})
            elif msg.get("type") == "ping":
                await ws.send_json({"type": "pong"})
    except WebSocketDisconnect:
        logger.info("ws disconnected for session %s", session_id)
