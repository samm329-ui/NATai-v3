import base64
from pathlib import Path
from typing import Optional, Dict, Iterator, Any, Union
import time
import json
import asyncio
import os
import re

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from starlette.requests import Request as StarletteRequest
from starlette.middleware.base import BaseHTTPMiddleware
import uvicorn
from contextlib import asynccontextmanager
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError

from app.models import ChatRequest, ChatResponse, TTSRequest

from app.utils.retry import with_retry
from app.utils.time_info import get_time_information

RATE_LIMIT_MESSAGE = (
    "You've reached your daily API limit for this assistant. "
    "Your credits will reset in a few hours, or you can upgrade your plan for more."
    "\nPlease try again later."
)

def is_rate_limit_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return "429" in msg or "rate limit" in msg or "tokens per day" in msg

from app.services.vector_store import VectorStoreService
from app.services.groq_service import GroqService, AllowableApiFailedError
from app.services.realtime_service import RealtimeGroqService
from app.services.brain_service import BrainService
from app.services.vision_service import VisionService
from app.services.task_executor import TaskExecutor
from app.services.task_manager import TaskManager
from app.services.chat_service import ChatService

from app.config import (
    VECTOR_STORE_DIR, GROQ_API_KEYS, GROQ_MODEL, TAVILY_API_KEY,
    EMBEDDING_MODEL, CHUNK_SIZE, CHUNK_OVERLAP, MAX_CHAT_HISTORY_TURNS,
    ASSISTANT_NAME, TTS_VOICE, TTS_RATE
)

import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)-20s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)

logger = logging.getLogger("N.A.T.A.S.H.A")

vector_store_service: VectorStoreService = None
groq_service: GroqService = None
realtime_service: RealtimeGroqService = None
brain_service: BrainService = None
task_executor: TaskExecutor = None
task_manager: TaskManager = None
vision_service: VisionService = None
chat_service: ChatService = None

def print_title():
    title = """
    ========================================================
       _   _  _     _  _      ___  __   ___  ___ _____ 
      | \ | || |   (_)| |    / _ \ \ \ / / ||_ _/ ___| 
      |  \| || |    | || |   | |_) | \ V / | | | \___ \ 
      | |\  || |___ | || |___|  _ <   | |  | | | |___) |
      |_| \_||_____||_||_____|_| \_\  |_|  |_|___|____/ 
                                                       
     Neural AI Technology & Advanced System for Human Assistance
    ========================================================
    """
    print(title)
    
@asynccontextmanager
async def lifespan(app: FastAPI):
    global vector_store_service, groq_service, realtime_service, brain_service
    global task_executor, task_manager, vision_service, chat_service
    
    print_title()
    
    logger.info("*" * 60)
    logger.info("N.A.T.A.S.H.A - Starting up...")
    logger.info("*" * 60)
    
    logger.info("[CONFIG] Assistant Name: %s", ASSISTANT_NAME)
    logger.info("[CONFIG] Groq Model: %s", GROQ_MODEL)
    logger.info("[CONFIG] Groq API keys loaded: %d", len(GROQ_API_KEYS))
    logger.info("[CONFIG] Tavily API key: %s", "Configured" if TAVILY_API_KEY else "NOT SET")
    logger.info("[CONFIG] Vision Model: %s", "Configured" if GROQ_API_KEYS else "NOT SET")
    logger.info("[CONFIG] Image generation: Pollinations.ai (free, no API key)")
    logger.info("[CONFIG] Embedding model: %s", EMBEDDING_MODEL)
    logger.info("[CONFIG] Chunk size: %d | Overlap: %d | Max History turns: %d",
                CHUNK_SIZE, CHUNK_OVERLAP, MAX_CHAT_HISTORY_TURNS)
                
    try:
         t0 = time.perf_counter()
         
         logger.info("Initializing vector store service...")
         vector_store_service = VectorStoreService()
         vector_store_service.get_retriever()
         
         logger.info("[TIMING] vector_store_service: %.3fs", time.perf_counter() - t0)
         
         t0 = time.perf_counter()
         groq_service = GroqService(vector_store_service)
         logger.info("Groq service initialized successfully")
         
         realtime_service = RealtimeGroqService(vector_store_service)
         logger.info("Realtime Groq service (with Tavily search...) initialized successfully")
         
         brain_service = BrainService(groq_service)
         logger.info("Brain service (Groq query classification...) initialized successfully")
         
         task_executor = TaskExecutor(groq_service)
         logger.info("Task Executor initialized successfully")
         
         task_manager = TaskManager(task_executor)
         logger.info("Background Task Manager initialized successfully")
         
         vision_service = VisionService()
         logger.info("Vision Service initialized successfully")
         
         chat_service = ChatService()
         chat_service.groq_service = groq_service
         chat_service.realtime_service = realtime_service
         chat_service.brain_service = brain_service
         chat_service.task_executor = task_executor
         chat_service.task_manager = task_manager
         chat_service.vision_service = vision_service
         chat_service.vector_store = vector_store_service
         logger.info("Chat service initialized successfully")
         
         logger.info("Service Status:")
         logger.info("  - Vector Store: Ready")
         logger.info("  - Groq AI (General): Ready")
         logger.info("  - Groq AI (Realtime): Ready")
         logger.info("  - Brain (Unified Decision): Ready")
         logger.info("  - Task Executor: Ready")
         logger.info("  - Background Task Manager: Ready")
         logger.info("  - Vision (Groq): Ready")
         logger.info("  - Chat Service: Ready")
         
         logger.info("*" * 60)
         logger.info("N.A.T.A.S.H.A is online and ready!")
         logger.info("API: http://localhost:8000")
         logger.info("Frontend: http://localhost:8000/app (open in browser)")
         logger.info("*" * 60)
         
         yield
         
    finally:
          logger.info("Shutting down N.A.T.A.S.H.A...")
         if task_manager:
             task_manager.shutdown()
             
         if chat_service:
             for session_id in list(chat_service.chat_sessions.keys()):
                 chat_service.save_chat_session(session_id)
                 
         logger.info("All sessions saved. Goodbye!")
         
app = FastAPI(
    title="N.A.T.A.S.H.A API",
    description="Neural AI Technology & Advanced System for Human Assistance",
    version="1.0.0",
    lifespan=lifespan,
    docs_url=None,
    redoc_url=None,
    openapi_url=None
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class TimingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        t0 = time.perf_counter()
        response = await call_next(request)
        elapsed = time.perf_counter() - t0
        path = request.url.path
        if not path.startswith("/assets") and path != "/favicon.ico":
             logger.info("[REQUEST] %s %s - %d (%.3fs)", request.method, path, response.status_code, elapsed)
        return response
        
app.add_middleware(TimingMiddleware)

@app.get("/api")
async def api_info():
    return {
        "message": "N.A.T.A.S.H.A API",
        "endpoints": [
            "/chat"          # General chat (non-streaming)
            "/chat/stream"   # General chat (streaming chunks)
            "/realtime"      # Realtime chat (non-streaming)
            "/realtime/stream" # Realtime chat (streaming chunks)
            "/natasha/stream" # Natasha unified route (two-stage brain: classify -> route -> execute/stream)
            "/tasks/{id}"    # Get background task status and result
            "/health"        # System health check
            "/tts"           # Text-to-speech (POST text, returns streamed MP3)
        ]
    }

@app.get("/health")
async def health():
    try:
        status = {
            "status": "healthy",
            "vector_store": vector_store_service is not None,
            "groq_service": groq_service is not None,
            "realtime_service": realtime_service is not None,
            "brain_service": brain_service is not None,
            "task_executor": task_executor is not None,
            "task_manager": task_manager is not None,
            "vision_service": vision_service is not None,
            "chat_service": chat_service is not None
        }
        return status
    except Exception as e:
        logger.error("Health check failed: %s", e)
        return {"status": "degraded", "error": str(e)}
        
@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    if not chat_service:
        raise HTTPException(status_code=503, detail="Chat service not initialized")
        
    logger.info("[API] /chat incoming | session_id: %s, message_len=%d", request.session_id, len(request.message))
    
    try:
         session_id = chat_service.get_or_create_session(request.session_id)
         
         response_text = chat_service.process_message_sync(session_id, request.message)
         
         logger.info("[API] /chat response | session_id: %s, response_len=%d", session_id, len(response_text))
         
         return ChatResponse(response=response_text, session_id=session_id)
         
    except AllowableApiFailedError as e:
         return ChatResponse(response=str(e), session_id=request.session_id or "unknown")
         
    except Exception as e:
         if is_rate_limit_error(e):
              logger.warning("[API] Chat rate limit hit: %s", e)
              raise HTTPException(status_code=429, detail=RATE_LIMIT_MESSAGE)
         logger.error("[API] /chat error: %s", e, exc_info=True)
         raise HTTPException(status_code=500, detail=f"Error processing chat: {str(e)}")

SPLIT_RE = re.compile(r"([.!?\n])\s+")
MIN_WORDS_FIRST = 1
MIN_WORDS = 3
MERGE_IF_WORDS = 2
TTS_BUFFER_TIMEOUT = 2.0
ABBREV_HOLD_RE = re.compile(r"(?i)\b(?:mr|ms|mrs|dr|prof|st|jr|sr|inc|ltd|vs|etc)\.$")

def _should_hold_sentence_for_continuation(sent: str) -> bool:
    s = sent.strip()
    if not s.endswith("."):
        return False
        
    words = s.split()
    if len(words) < 1:
        return False
        
    return bool(ABBREV_HOLD_RE.search(words[-1]))
    
def _split_sentences(buf: str) -> list[str]:
    parts = SPLIT_RE.split(buf)
    sentences = []
    pending = ""
    
    for i in range(0, len(parts), 2):
        raw = parts[i].strip()
        p = parts[i+1].strip() if i+1 < len(parts) else ""
        
        if not raw and not p:
            continue
            
        s = raw + p
        if pending:
             s = pending + " " + s
             pending = ""
             
        min_req = MIN_WORDS_FIRST if not sentences else MIN_WORDS
        
        if len(s.split()) < min_req:
             pending = s
        else:
             sentences.append(s)
             
    if pending:
        if sentences:
            sentences[-1] += " " + pending
        else:
            sentences.append(pending)
            
    return sentences
    
def _merge_short(sentences):
    if not sentences:
        return []
        
    merged = []
    i = 0
    while i < len(sentences):
        cur = sentences[i]
        
        while i + 1 < len(sentences) and len(sentences[i+1].split()) <= MERGE_IF_WORDS:
            i += 1
            cur += " " + sentences[i]
            
        merged.append(cur)
        i += 1
        
    return merged

def generate_tts_sync(text: str, voice: str, rate: str) -> bytes:
    try:
        import edge_tts
        communicate = edge_tts.Communicate(text=text, voice=voice, rate=rate)
        
        chunks = []
        for chunk in communicate.stream_sync():
            if chunk["type"] == "audio":
                chunks.append(chunk["data"])
                
        return b"".join(chunks)
    except ImportError:
        return b""
        
tts_pool = ThreadPoolExecutor(max_workers=4)

def _stream_generator(session_id: str, chunk_iter, is_realtime: bool, tts_enabled: bool = False):
    yield f"data: {json.dumps({'session_id': session_id, 'done': False})}\n\n"
    
    audio_queue = []
    events = []
    
    is_first = True
    last_submit_time = time.perf_counter()
    
    def _submit(text):
        nonlocal last_submit_time
        if not text or not text.strip():
            return
            
        audio_queue.append(tts_pool.submit(generate_tts_sync, text, TTS_VOICE, TTS_RATE))
        last_submit_time = time.perf_counter()
        
    def _drain_ready():
        while audio_queue and audio_queue[0].done():
            fut = audio_queue.pop(0)
            try:
                audio = fut.result()
                if audio:
                    b64 = base64.b64encode(audio).decode('ascii')
                    events.append(f"data: {json.dumps({'audio': b64, 'sentence': sent})}\n\n")
            except Exception as e:
                logger.warning("[TTS-INLINE] Failed for '%s': %s", sent[:40], e)
                
        return events
        
    def _yield_completed_audio():
        if not tts_enabled:
            return
            
        for ev in _drain_ready():
            yield ev
            
    try:
        for chunk in chunk_iter:
            if isinstance(chunk, dict) and "activity" in chunk:
                 yield f"data: {json.dumps({'chunk': chunk, 'activity': chunk['activity']})}\n\n"
                 continue
                 
            if isinstance(chunk, dict) and "search_results" in chunk:
                 yield f"data: {json.dumps({'chunk': chunk['search_results']})}\n\n"
                 yield from _yield_completed_audio()
                 continue
                 
            if isinstance(chunk, dict) and "actions" in chunk:
                 yield f"data: {json.dumps({'chunk': chunk['actions']})}\n\n"
                 yield from _yield_completed_audio()
                 continue
                 
            if isinstance(chunk, dict) and "background_tasks" in chunk:
                 yield f"data: {json.dumps({'chunk': chunk['background_tasks']})}\n\n"
                 yield from _yield_completed_audio()
                 continue
                 
            if chunk:
                 yield f"data: {json.dumps({'chunk': chunk, 'done': False})}\n\n"
                 
            if not tts_enabled:
                 continue
                 
            yield from _yield_completed_audio()
            
            buffer += chunk
            sentences = _split_sentences(buffer)
            sentences = _merge_short(sentences)
            
            if not sentences:
                continue
                
            held = None
            
            for i, sent in enumerate(sentences):
                min_w = MIN_WORDS_FIRST if is_first and not sentences else MIN_WORDS
                
                is_last = (i == len(sentences) - 1)
                
                if len(sent.split()) < min_w:
                    held = sent
                    break
                    
                if is_last and _should_hold_sentence_for_continuation(sent):
                    held = sent
                    break
                    
                else:
                    _submit(sent)
                    is_first = False
                    
            if held:
                if buffer and len(buffer.split()) >= TTS_BUFFER_TIMEOUT:
                    if time.perf_counter() - last_submit_time > TTS_BUFFER_TIMEOUT:
                        _submit(held.strip())
                        held = None
                        is_first = False
                        
            buffer = held if held else ""
            
    except Exception as e:
        logger.error("[STREAM] Error in generator: %s", e)
        yield f"data: {json.dumps({'chunk': '', 'done': True, 'error': str(e)})}\n\n"
        return
        
    if tts_enabled:
        remaining = buffer.strip()
        if remaining:
             if len(remaining.split()) >= MERGE_IF_WORDS:
                  _submit(remaining)
                  
        for fut in audio_queue:
             try:
                 audio = fut.result(timeout=15)
                 if audio:
                      b64 = base64.b64encode(audio).decode('ascii')
                      yield f"data: {json.dumps({'audio': b64, 'sentence': sent})}\n\n"
             except FuturesTimeoutError:
                 logger.warning("[TTS-INLINE] Timeout for '%s'", sent[:40])
             except Exception as e:
                 logger.warning("[TTS-INLINE] Failed for '%s': %s", sent[:40], e)
                 
    yield f"data: {json.dumps({'chunk': '', 'done': True, 'session_id': session_id})}\n\n"
    
@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    if not chat_service:
        raise HTTPException(status_code=503, detail="Chat service not initialized")
        
    logger.info("[API] /chat/stream incoming | session_id: %s, message_len=%d", request.session_id, len(request.message))
    
    try:
         session_id = chat_service.get_or_create_session(request.session_id)
         
         chunk_iter = chat_service.process_message_stream(session_id, request.message)
         
         return StreamingResponse(
             _stream_generator(session_id, chunk_iter, is_realtime=False, tts_enabled=request.tts),
             media_type="text/event-stream",
             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
         )
         
    except AllowableApiFailedError as e:
         raise HTTPException(status_code=503, detail=str(e))
         
    except Exception as e:
         if is_rate_limit_error(e):
              logger.warning("[API] Chat stream rate limit hit: %s", e)
              raise HTTPException(status_code=429, detail=RATE_LIMIT_MESSAGE)
         logger.error("[API] /chat/stream error: %s", e, exc_info=True)
         raise HTTPException(status_code=500, detail=f"Error processing chat: {str(e)}")
         
@app.post("/realtime", response_model=ChatResponse)
async def chat_realtime(request: ChatRequest):
    if not realtime_service:
        raise HTTPException(status_code=503, detail="Realtime service not initialized")
        
    logger.info("[API] /realtime incoming | session_id: %s, message_len=%d", request.session_id, len(request.message))
    
    try:
         session_id = chat_service.get_or_create_session(request.session_id)
         
         response_text = chat_service.process_message_sync(session_id, request.message)
         
         logger.info("[API] /realtime response | session_id: %s, response_len=%d", session_id, len(response_text))
         
         return ChatResponse(response=response_text, session_id=session_id)
         
    except AllowableApiFailedError as e:
         return ChatResponse(response=str(e), session_id=request.session_id or "unknown")
         
    except Exception as e:
         if is_rate_limit_error(e):
              logger.warning("[API] Realtime chat rate limit hit: %s", e)
              raise HTTPException(status_code=429, detail=RATE_LIMIT_MESSAGE)
         logger.error("[API] /realtime error: %s", e, exc_info=True)
         raise HTTPException(status_code=500, detail=f"Error processing chat: {str(e)}")

@app.post("/realtime/stream")
async def chat_realtime_stream(request: ChatRequest):
    if not chat_service or not realtime_service:
        raise HTTPException(status_code=503, detail="Realtime service not initialized")
        
    logger.info("[API] /realtime/stream incoming | session_id: %s, message_len=%d", request.session_id, len(request.message))
    
    try:
         session_id = chat_service.get_or_create_session(request.session_id)
         
         chunk_iter = chat_service.process_realtime_stream(session_id, request.message)
         
         return StreamingResponse(
             _stream_generator(session_id, chunk_iter, is_realtime=True, tts_enabled=request.tts),
             media_type="text/event-stream",
             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
         )
         
    except AllowableApiFailedError as e:
         raise HTTPException(status_code=503, detail=str(e))
         
    except Exception as e:
         if is_rate_limit_error(e):
              logger.warning("[API] Realtime stream rate limit hit: %s", e)
              raise HTTPException(status_code=429, detail=RATE_LIMIT_MESSAGE)
         logger.error("[API] /realtime/stream error: %s", e, exc_info=True)
         raise HTTPException(status_code=500, detail=f"Error processing chat: {str(e)}")

@app.post("/natasha/stream")
async def natasha_stream(request: ChatRequest):
    if not chat_service:
        raise HTTPException(status_code=503, detail="Service not initialized")
        
    logger.info("[API] /natasha/stream incoming | session_id: %s, message_len=%d, img=%s", request.session_id, len(request.message), "yes" if request.img_base64 else "no")
    
    try:
         session_id = chat_service.get_or_create_session(request.session_id)
         
         chunk_iter = chat_service.process_message_stream(
             session_id=session_id, request_message=request.message, img_base64=request.img_base64
         )
         
         return StreamingResponse(
             _stream_generator(session_id, chunk_iter, is_realtime=False, tts_enabled=request.tts),
             media_type="text/event-stream",
             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
         )
         
    except AllowableApiFailedError as e:
         raise HTTPException(status_code=503, detail=str(e))
         
    except Exception as e:
         if is_rate_limit_error(e):
              logger.warning("[API] Natasha stream rate limit hit: %s", e)
              raise HTTPException(status_code=429, detail=RATE_LIMIT_MESSAGE)
         logger.error("[API] /natasha/stream error: %s", e, exc_info=True)
         raise HTTPException(status_code=500, detail=f"Error processing chat: {str(e)}")
         
@app.get("/tasks/{task_id}")
async def get_task_status(task_id: str):
    if not task_manager:
        raise HTTPException(status_code=503, detail="Task manager not initialized")
        
    data = task_manager.get_serializable(task_id)
    if not data:
        raise HTTPException(status_code=404, detail="Task not found")
        
    return data

@app.get("/tasks/{task_id}/image")
async def get_task_image(task_id: str):
    if not task_manager:
        raise HTTPException(status_code=503, detail="Task manager not initialized")
        
    entry = task_manager.get(task_id)
    if not entry:
        raise HTTPException(status_code=404, detail="Task not found")
        
    if entry.status != "completed" or not entry.image_bytes:
        raise HTTPException(status_code=400, detail="Image not ready")
        
    from fastapi.responses import Response
    return Response(content=entry.image_bytes, media_type="image/png")

@app.get("/chat/history/{session_id}")
async def get_chat_history(session_id: str):
    if not chat_service:
        raise HTTPException(status_code=503, detail="Chat service not initialized")
        
    if not chat_service.validate_session_id(session_id):
        raise HTTPException(status_code=400, detail="Invalid session ID format")
        
    try:
         messages = chat_service.get_chat_history(session_id)
         
         formatted = []
         for msg in messages:
              formatted.append({"role": msg["role"], "content": msg["content"]})
              
         return {"session_id": session_id, "messages": formatted}
         
    except Exception as e:
         logger.error("Error retrieving history: %s", e, exc_info=True)
         raise HTTPException(status_code=500, detail=f"Error retrieving history: {str(e)}")

@app.post("/tts")
async def test_tts_speech(request: TTSRequest):
    if not request.text:
         raise HTTPException(status_code=400, detail="Text is required")
         
    async def generate():
         try:
              import edge_tts
              communicate = edge_tts.Communicate(text=request.text, voice=TTS_VOICE, rate=TTS_RATE)
              async for chunk in communicate.stream():
                   if chunk["type"] == "audio":
                        yield chunk["data"]
         except Exception as e:
              logger.error("[TTS] Error generating speech: %s", e)
              
    return StreamingResponse(
        generate(),
        media_type="audio/mpeg",
        headers={"Cache-Control": "no-cache"},
    )
    
_frontend_dir = Path(__file__).resolve().parent.parent / "frontend"

if _frontend_dir.exists():
    app.mount("/app", StaticFiles(directory=str(_frontend_dir), html=True), name="frontend")
    
    @app.get("/")
    async def root_redirect():
        from fastapi.responses import RedirectResponse
        return RedirectResponse(url="/app", status_code=302)

def run():
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
    
if __name__ == "__main__":
    run()