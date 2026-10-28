import os
import sys
import json
import time
import asyncio
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

# Ensure workspace is in sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from copilot_engine import ConversationalCopilotEngine

app = FastAPI(title="Voice Intelligence & Knowledge Copilot")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize AI Engine Singleton
print("Initializing Voice & RAG Intelligence Engine...")
engine = ConversationalCopilotEngine()

INDEX_HTML_PATH = os.path.join(BASE_DIR, "templates", "index.html")

@app.get("/", response_class=HTMLResponse)
async def serve_dashboard():
    """Serves the Glassmorphism interactive dashboard."""
    if os.path.exists(INDEX_HTML_PATH):
        with open(INDEX_HTML_PATH, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse("<h1>Error: templates/index.html not found.</h1>", status_code=404)

@app.post("/api/upload")
async def upload_document(file: UploadFile = File(...)):
    """Receives PDF or TXT files, chunks and embeds into the local vector store."""
    try:
        content = await file.read()
        res = engine.ingest_document(file.filename, content)
        return JSONResponse(res)
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)

@app.get("/api/documents")
async def get_documents():
    """Returns list of currently indexed documents."""
    return JSONResponse({
        "documents": engine.documents,
        "total_chunks": len(engine.chunks)
    })

@app.delete("/api/documents/{filename:path}")
async def delete_document(filename: str):
    """Removes a specific document from the knowledge base."""
    res = engine.remove_document(filename)
    if res.get("status") == "error":
        return JSONResponse(res, status_code=404)
    return JSONResponse(res)

@app.post("/api/documents/delete")
async def delete_document_post(payload: dict):
    """POST alternative to delete a specific document."""
    filename = payload.get("filename")
    if not filename:
        return JSONResponse({"status": "error", "message": "Missing filename"}, status_code=400)
    res = engine.remove_document(filename)
    return JSONResponse(res)

@app.post("/api/clear")
async def clear_documents():
    """Clears all indexed documents."""
    return JSONResponse(engine.clear_documents())

@app.post("/api/reset-session")
async def reset_session():
    """Resets conversational turns for a new call."""
    engine.turns = []
    engine.pending_text = ""
    engine.call_sentiment_ema = 0.0
    engine.action_items = []
    return JSONResponse({"status": "ok", "message": "Conversation session reset."})

@app.get("/api/summary")
async def get_summary():
    """Returns the final executive conversation report."""
    return JSONResponse({"summary": engine.generate_call_summary()})

# ==================== WEBSOCKET LIVE AUDIO STREAM ====================

@app.websocket("/ws/live")
async def websocket_endpoint(websocket: WebSocket):
    """
    Bi-directional streaming WebSocket:
      - Receives 16kHz PCM audio bytes from browser microphone
      - Transcribes with faster-whisper int8
      - Processes sentiment, prosodic tone, and RAG knowledge cards
      - Pushes real-time JSON events to UI
    """
    await websocket.accept()
    print("[WebSocket] Client connected.")

    audio_buffer = bytearray()
    silence_frames = 0
    SAMPLE_RATE = 16000
    BYTES_PER_SAMPLE = 2  # 16-bit PCM

    # Dynamic speech detection thresholds (calibrated for natural conversational cadence)
    MIN_PHRASE_DURATION_S = 0.60   # At least 0.6s of audio before evaluating (catches short inquiries)
    MAX_PHRASE_DURATION_S = 3.80   # Max 3.8s continuous speech window before segment slicing
    SILENCE_THRESHOLD_S = 0.55     # 550ms human conversational pause before finalizing turn
    SILENCE_ENERGY_RMS = 0.018     # Real room noise is < 0.018; human vocal speech is >= 0.025

    try:
        while True:
            data = await websocket.receive()

            # Handle JSON control messages (e.g. End Call)
            if "text" in data:
                msg = json.loads(data["text"])
                if msg.get("action") == "end_call":
                    summary = engine.generate_call_summary()
                    await websocket.send_json({"type": "summary", "data": summary})
                    continue

            # Handle binary PCM audio frames
            if "bytes" in data:
                chunk = data["bytes"]
                audio_buffer.extend(chunk)

                # Check energy of this incoming audio chunk
                chunk_np = np.frombuffer(chunk, dtype=np.int16).astype(np.float32) / 32768.0
                rms = float(np.sqrt(np.mean(chunk_np ** 2))) if len(chunk_np) > 0 else 0.0

                if rms < SILENCE_ENERGY_RMS:
                    silence_frames += len(chunk_np)
                else:
                    silence_frames = 0

                buffer_duration_s = len(audio_buffer) / (SAMPLE_RATE * BYTES_PER_SAMPLE)
                silence_duration_s = silence_frames / SAMPLE_RATE

                # Trigger condition: natural conversational pause (>= 0.55s) or continuous buffer reached (>= 3.8s)
                should_transcribe = (
                    (buffer_duration_s >= MIN_PHRASE_DURATION_S and silence_duration_s >= SILENCE_THRESHOLD_S)
                    or buffer_duration_s >= MAX_PHRASE_DURATION_S
                )

                if should_transcribe and len(audio_buffer) >= (SAMPLE_RATE * BYTES_PER_SAMPLE * 0.4):
                    is_silence_pause = (silence_duration_s >= SILENCE_THRESHOLD_S)

                    # Convert buffer to float32 NumPy array
                    audio_np = np.frombuffer(bytes(audio_buffer), dtype=np.int16).astype(np.float32) / 32768.0

                    peak = float(np.max(np.abs(audio_np))) if len(audio_np) > 0 else 0.0
                    buffer_rms = float(np.sqrt(np.mean(audio_np ** 2))) if len(audio_np) > 0 else 0.0

                    # Hard Silence Gate: Ignore ambient room noise / fan hum without human speech
                    if peak < 0.038 and buffer_rms < 0.016:
                        if is_silence_pause:
                            audio_buffer.clear()
                            silence_frames = 0
                        continue

                    # Automatic Gain Control (AGC) peak normalization
                    if peak >= 0.038:
                        audio_np = (audio_np / peak) * 0.90

                    # Transcribe asynchronously in background thread
                    segments = await asyncio.to_thread(engine.transcribe, audio_np, return_segments=True)

                    if segments:
                        if is_silence_pause:
                            # Natural conversational pause: finalize all segments
                            for seg in segments:
                                seg_text = seg.text.strip()
                                if seg_text:
                                    turn_events = await asyncio.to_thread(
                                        engine.process_speech_stream,
                                        seg_text,
                                        raw_audio=audio_np,
                                        is_silence_pause=True,
                                        is_complete_segment=True
                                    )
                                    for turn_res in turn_events:
                                        await websocket.send_json({"type": "turn", "data": turn_res})

                            # Clear buffer and reset live interim line
                            audio_buffer.clear()
                            silence_frames = 0
                            await websocket.send_json({"type": "partial", "text": ""})

                        else:
                            # Continuous non-stop speech (buffer >= 3.8s):
                            # Slices audio at exact segment timestamp boundary to eliminate word clipping
                            if len(segments) > 1:
                                # Finalize all except last incomplete segment
                                for seg in segments[:-1]:
                                    seg_text = seg.text.strip()
                                    if seg_text:
                                        turn_events = await asyncio.to_thread(
                                            engine.process_speech_stream,
                                            seg_text,
                                            raw_audio=audio_np,
                                            is_silence_pause=False,
                                            is_complete_segment=True
                                        )
                                        for turn_res in turn_events:
                                            await websocket.send_json({"type": "turn", "data": turn_res})

                                # Advance buffer to start after the last finalized segment
                                cut_time = segments[-2].end
                                cut_bytes = int(cut_time * SAMPLE_RATE * BYTES_PER_SAMPLE)
                                if cut_bytes < len(audio_buffer):
                                    audio_buffer = bytearray(audio_buffer[cut_bytes:])
                                else:
                                    audio_buffer.clear()
                                silence_frames = 0

                                # Emit interim partial for trailing speech
                                last_text = segments[-1].text.strip()
                                if last_text:
                                    await websocket.send_json({"type": "partial", "text": last_text})

                            else:
                                # Single segment in window: ALWAYS finalize turn and advance buffer!
                                seg = segments[0]
                                seg_text = seg.text.strip()
                                if seg_text:
                                    turn_events = await asyncio.to_thread(
                                        engine.process_speech_stream,
                                        seg_text,
                                        raw_audio=audio_np,
                                        is_silence_pause=False,
                                        is_complete_segment=True
                                    )
                                    for turn_res in turn_events:
                                        await websocket.send_json({"type": "turn", "data": turn_res})

                                cut_time = seg.end
                                cut_bytes = int(cut_time * SAMPLE_RATE * BYTES_PER_SAMPLE)
                                if cut_bytes < len(audio_buffer):
                                    audio_buffer = bytearray(audio_buffer[cut_bytes:])
                                else:
                                    audio_buffer.clear()
                                silence_frames = 0
                                await websocket.send_json({"type": "partial", "text": ""})
                    else:
                        # No valid speech detected in buffer: reset to prevent infinite growth
                        audio_buffer.clear()
                        silence_frames = 0
                        await websocket.send_json({"type": "partial", "text": ""})

                # Absolute safety cap: If audio buffer ever exceeds 8.0s, trim to latest 3.0s
                if len(audio_buffer) > (SAMPLE_RATE * BYTES_PER_SAMPLE * 8.0):
                    audio_buffer = bytearray(audio_buffer[-int(SAMPLE_RATE * BYTES_PER_SAMPLE * 3.0):])
                    silence_frames = 0

    except WebSocketDisconnect:
        print("[WebSocket] Client disconnected.")
    except Exception as e:
        print(f"[WebSocket Error]: {e}")

if __name__ == "__main__":
    import uvicorn
    print("\n=======================================================")
    print("  VOICE INTELLIGENCE & REAL-TIME KNOWLEDGE COPILOT")
    print("  Local Server running at: http://localhost:8000")
    print("=======================================================\n")
    uvicorn.run(app, host="0.0.0.0", port=8000)
