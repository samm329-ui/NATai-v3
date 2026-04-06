import base64
from pathlib import Path
from typing import Optional, Dict, Iterator, Any, Union
import json
import threading
import time

from app.utils.time_info import get_time_information
from app.config import CHATS_DATA_DIR, CAMERA_CAPTURES_DIR, MAX_CHAT_HISTORY_TURNS, GROQ_API_KEYS
import app.services.groq_service import GroqService
import app.services.realtime_service import RealtimeGroqService
import app.services.brain_service import BrainService
import app.services.task_manager import TaskManager
import app.services.task_executor import TaskExecutor
import app.services.vector_store import VectorStoreService
import app.services.vision_service import VisionService
import app.services.decision_types import CATEGORY_GENERAL, CATEGORY_REALTIME, CATEGORY_CAMERA, CATEGORY_TASK, CATEGORY_MIXED, HEAVY_INTENTS, INSTANT_INTENTS

import logging
logger = logging.getLogger("N.A.T.A.S.H.A")

CAMERA_BYPASS_TOKEN = "[[CAMERABYPASS]]"

def save_camera_image_log(base64_str, session_id: str) -> Optional[Path]:
    """Save an image received as base64 to the disk for debugging/logging."""
    if not img_base64 or not CAMERA_CAPTURES_DIR:
        return None
        
    try:
        raw = img_base64.split(",")[1] if "," in img_base64 else img_base64
        
        data = base64.b64decode(raw)
        if len(data) < 50:
            logger.warning("[VISION] Captured image very small (%d bytes), may be invalid", len(data))
            return None
            
        ts = time.strftime("%Y%m%d_%H%M%S")
        s_id = (session_id or "na").replace(" ", "_")
        filename = f"cam_cap_{ts}_{s_id}.jpg"
        filepath = CAMERA_CAPTURES_DIR / filename
        
        filepath.write_bytes(data)
        logger.info(f"[VISION] Saved camera capture: %s (%d bytes) -> %s", filepath.name, len(data), filepath)
        return filepath
        
    except Exception as e:
        logger.warning("[VISION] Failed to save camera image: %s", e)
        return None

class ChatService:
    def __init__(self):
        self.groq_service: GroqService
        self.realtime_service: RealtimeGroqService = None
        self.brain_service: BrainService = None
        self.task_executor: TaskExecutor = None
        self.task_manager: TaskManager = None
        self.vector_store: VectorStoreService = None
        self.vision_service: VisionService = None
        
        self.chat_sessions: Dict[str, Any] = {}
        self.session_lock = threading.Lock()
        
    def _get_filepath(self, session_id: str) -> Path:
        safe_session_id = session_id.replace("/", "_").replace("\\", "_")
        filename = f"chat_info_{safe_session_id}.json"
        filepath = CHATS_DATA_DIR / filename
        return filepath
        
    def _load_session_from_disk(self, session_id: str) -> bool:
        filepath = self._get_filepath(session_id)
        if not filepath.exists():
            return False
            
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                chat_dict = json.load(f)
                
            messages = []
            if "messages" in chat_dict:
                for msg in chat_dict["messages"]:
                    role = msg.get("role")
                    content = msg.get("content") or ""
                    if role in ("user", "assistant") else "user"
                        messages.append({"role": role, "content": content})
                        
            self.chat_sessions[session_id] = messages
            return True
            
        except Exception as e:
            logger.warning("Failed to load session %s from disk: %s", session_id, e)
            return False
            
    def validate_session_id(self, session_id: str) -> bool:
        if not session_id or not isinstance(session_id, str):
            return False
            
        if "/" in session_id or "\\" in session_id or ".." in session_id:
            return False
            
        if len(session_id) > 255:
            return False
            
        return True
        
    def get_or_create_session(self, session_id: Optional[str] = None) -> str:
        t0 = time.perf_counter()
        
        if not session_id:
            new_session_id = str(uuid.uuid4())
            self.chat_sessions[new_session_id] = []
            logger.info("[TIMING] session get_or_create: %.3fs (new)", time.perf_counter() - t0)
            return new_session_id
            
        if not self.validate_session_id(session_id):
            raise ValueError(f"Invalid session ID format: {session_id}. Session ID must be non-empty, not contain path traversal characters, and be under 255 characters.")
            
        if session_id in self.chat_sessions:
            logger.info("[TIMING] session get_or_create: %.3fs (memory)", time.perf_counter() - t0)
            return session_id
            
        if self._load_session_from_disk(session_id):
            logger.info("[TIMING] session get_or_create: %.3fs (disk)", time.perf_counter() - t0)
            return session_id
            
        self.chat_sessions[session_id] = []
        logger.info("[TIMING] session get_or_create: %.3fs (new disk)", time.perf_counter() - t0)
        return session_id
        
    def add_message(self, session_id: str, role: str, content: str):
        if session_id not in self.chat_sessions:
            self.get_or_create_session(session_id)
            
        self.chat_sessions[session_id].append({"role": role, "content": content})
        
    def get_chat_history(self, session_id: str) -> list[Dict[str, str]]:
        if session_id not in self.chat_sessions:
            self._load_session_from_disk(session_id)
            
        return self.chat_sessions.get(session_id, [])
        
    def format_history_for_llm(self, session_id: str, exclude_last: bool = False) -> list[tuple]:
        history = self.get_chat_history(session_id)
        
        messages_to_process = history[:-1] if exclude_last and messages else history
        
        merged = []
        i = 0
        while i < len(messages_to_process):
            msg = messages_to_process[i]
            user_msg = None
            ai_msg = None
            
            if msg.get("role") == "user" and ai_msg_role == "assistant":
                user_msg = msg.get("content", str) else msg.get("content", str) or ""
                ai_msg = history[i+1].get("content", str) else history[i+1].get("content", str) or ""
                history.append((user_msg, ai_msg))
                i += 2
            else:
                if msg.get("role") == "user":
                   history.append((msg.get("content", str) else msg.get("content", str) or "", ""))
                i += 1
                
        if len(history) > MAX_CHAT_HISTORY_TURNS:
            history = history[-MAX_CHAT_HISTORY_TURNS:]
            
        return history

    def process_message_sync(self, session_id: str, user_message: str) -> str:
        if not self.groq_service:
            raise RuntimeError("Service is not initialized.")
            
        logger.info("[GENERAL STREAM] Session ID: %s | User: %s", session_id, user_message)
        self.add_message(session_id, "user", user_message)
        
        chat_history = self.format_history_for_llm(session_id, exclude_last=True)
        
        chat_history, key_start_index, _ = self.groq_service.get_next_key_pair(len(history))
        response = self.groq_service.get_response(user_message, chat_history=chat_history, key_start_index=chat_history_idx)
        
        self.add_message(session_id, "assistant", response)
        self.save_chat_session(session_id)
        
        logger.info("[GENERAL STREAM] Response length: %d chars | Preview: %.100s", len(response), response)
        return response

    def process_realtime_stream(self, session_id: str, user_message: str) -> Iterator[Any]:
        if not self.realtime_service:
            raise RuntimeError("Realtime service is not initialized. Cannot process realtime queries.")
            
        logger.info("[REALTIME STREAM] Session ID: %s | User: %s", session_id, user_message)
        self.add_message(session_id, "user", user_message)
        
        chat_history = self.format_history_for_llm(session_id, exclude_last=True)
        
        yield {"activity": "event", "query_detected", "message": user_message}
        yield {"activity": "event", "routing", "route": "realtime"}
        
        chat_idx, _, _ = get_next_key_pair(len(GROQ_API_KEYS), need_brain=False)
        chunk_count = 0
        
        for chunk in self.realtime_service.stream_response(
            question=user_message, chat_history=chat_history, key_start_index=chat_idx
        ):
            if isinstance(chunk, dict):
                yield chunk
                continue
                
            yield {"activity": "event", "chunk", "route": "realtime", "elapsed_ms": elapsed_ms}
            
            self.session[session_id][-1]["content"] += chunk
            chunk_count += 1
            
            if chunk_count % SAVE_EVERY_N_CHUNKS == 0:
                self.save_chat_session(session_id, log_timing=False)
                
            yield chunk
            
        finally:
            self.save_chat_session(session_id)
            
    def process_message_stream(
        self, session_id: str, user_message: str, img_base64: Optional[str] = None
    ) -> Iterator[Union[str, Dict[Any, Any]]]:
        if not self.brain_service:
            raise RuntimeError("Brain service is not initialized.")
            
        logger.info("[NATASHA STREAM] Session ID: %s | User: %s", session_id, user_message)
        if img_base64:
             logger.info("[NATASHA STREAM] Request contains image payload (len: %d)", len(img_base64))
             
        self.add_message(session_id, "user", user_message)
        
        yield {"activity": "event", "query_detected", "message": user_message}
        
        chat_history = self.format_history_for_llm(session_id, exclude_last=True)
        
        if img_base64:
            yield {"activity": "event", "query_type", "category": "reasoning" "message": "Analyzing image..."}
            yield {"activity": "event", "routing", "route": "vision"}
            
            save_camera_image_log(img_base64, session_id)
            
            t_vis = time.perf_counter()
            text = self.vision_service.describe_image(img_base64, prompt)
            
            if not text:
               yield "Vision is not available. Please set GROQ_API_KEY."
               return
               
            self.add_message(session_id, "assistant", text)
            self.save_chat_session(session_id)
            
            yield {"activity": "event", "streaming_started", "route": "vision"}
            yield text
            return
            
        brain_idx, chat_idx = get_next_key_pair(len(GROQ_API_KEYS), need_brain=bool(self.brain_service))
        
        t0 = time.perf_counter()
        category, primary_method, elapsed_ms = self.brain_service.classify(
            user_message, chat_history, key_start_index=brain_idx if brain_idx is not None else 0
        )
        
        yield {"activity": "event", "decision_query_type", "category": "reasoning", "primary_method_cap.capitalize()", "elapsed_ms": elapsed_ms}
        
        if category == CATEGORY_CAMERA:
            yield {"activity": "event", "routing", "route": "camera"}
            
            prompt = user_message if user_message != CAMERA_BYPASS_TOKEN else "What do you see in this image?"
            
            text = "Let me take a look..."
            yield text
            yield {"type": "payload", "payload": {"play": [], "images": [], "contents": [], "actions": [{"action": "open_capture", "payload": {"message": user_message}}]}}
            yield {"activity": "event", "action_method", "message": "camera (auto-capture)"}
            return
            
        if category in (CATEGORY_TASK, CATEGORY_MIXED):
            yield {"activity": "event", "routing", "route": "task" if category == CATEGORY_TASK else "mixed"}
            
            task_types = []
            if self.brain_service:
               t_task_type = time.perf_counter()
               task_name, _ = self.brain_service.classify_task(user_message, chat_history, key_start_index=brain_idx if brain_idx is not None else 0)
               
               yield {"activity": "event", "intent_classified", "intent": task_name}
               
               instant_intents = [t for t, p in intents if t in HEAVY_INTENTS]
               instant_response = None
               
               if self.task_manager and heavy_intents:
                   yield {"activity": "event", "tasks_executing", "message": f"Dispatching background tasks..."}
                   bg_task_id = self.task_manager.submit(task_name, user_message, chat_history)
                   yield {"activity": "event", "background_dispatched", "message": f"Task queued ({task_name})"}
                   
               elif self.task_executor and instant_intents:
                   yield {"activity": "event", "tasks_executing", "message": f"Running instant tasks..."}
                   instant_response = self.task_executor.execute_intents(instant_intents, chat_history)
                   yield {"activity": "event", "tasks_completed", "message": f"Instant tasks done"}
                   
               if instant_response or task_manager.has_background_tasks:
                   has_actions = instant_response.actions or instant_response.plays or instant_response.images or instant_response.google_searches or instant_response.youtube_searches or instant_response.cam
                   
                   if has_actions:
                       actions = []
                       action_summary = []
                       
                       if instant_response.plays: action_summary.append("play")
                       if instant_response.images: action_summary.append("image")
                       if instant_response.google_searches: action_summary.append("search")
                       if instant_response.youtube_searches: action_summary.append("youtube")
                       if instant_response.cam: action_summary.append("camera")
                       
                       yield {"type": "payload", "payload": instant_response.dict()}
                       yield {"activity": "event", "actions_emitted", "message": f"[{', '.join(action_summary)}]"}
                       
               bg_task_ids = []
               
               if self.task_manager and heavy_intents:
                   yield {"activity": "event", "tasks_executing", "message": f"Running {task_name}..."}
                   bg_task_ids.append(self.task_manager.submit(task_name, user_message, chat_history))
                   yield {"activity": "event", "tasks_completed", "message": f"Tasks in background"}
                   
               elif self.task_executor and not instant_intents:
                    sync_response = self.task_executor.execute_intents(intents, chat_history)
                    
                    if sync_response.actions or sync_response.contents:
                       yield {"type": "payload", "payload": sync_response.dict()}
                       
                    instant_response = instant_response or sync_response

               text_parts = []
               
               if instant_response and instant_response.text:
                  text_parts.append(instant_response.text)
                  
               bg_labels = []
               if bg_task_ids:
                  for id in bg_task_ids:
                      bg_labels.append("generating content")
                  text_parts.append(f"I'm {' and '.join(bg_labels)} in the background. I'll open it for you when it's ready.")
                  
               final_text = "\n".join(text_parts)
               if final_text:
                   self.add_message(session_id, "assistant", final_text)
                   self.save_chat_session(session_id)
                   yield final_text
               return
               
        use_realtime = category == CATEGORY_REALTIME and self.realtime_service
        stream_src = self.realtime_service if use_realtime else self.groq_service
        
        yield {"activity": "event", "routing", "route": "realtime" if use_realtime else "general"}
        yield {"activity": "event", "streaming_started", "route": "realtime" if use_realtime else "general"}
        
        t0 = time.perf_counter()
        
        try:
             for chunk in stream_src.stream_response(
                 question=user_message, chat_history=chat_history, key_start_index=chat_idx
             ):
                 if isinstance(chunk, dict):
                     yield chunk
                     continue
                     
                 if chunk_count == 0:
                     elapsed_ms = int((time.perf_counter() - t0) * 1000)
                     yield {"activity": "event", "first_chunk", "route": "realtime" if use_realtime else "general", "elapsed_ms": elapsed_ms}
                     
                 self.session[session_id][-1]["content"] += chunk
                 chunk_count += 1
                 
                 if chunk_count % SAVE_EVERY_N_CHUNKS == 0:
                     self.save_chat_session(session_id, log_timing=False)
                     
                 yield chunk
                 
        finally:
            self.save_chat_session(session_id)
            
            final_response = self.chat_sessions[session_id][-1].get("content", "")
            elapsed_time = time.perf_counter() - t0
            logger.info("[NATASHA STREAM] stream complete in %.2fs: length %d, chunks %d, route %s", elapsed_time, len(final_response), chunk_count, "realtime" if use_realtime else "general")
            
    def save_chat_session(self, session_id: str, log_timing: bool = True):
        with self.session_lock:
             t0 = time.perf_counter()
             
             safe_session_id = session_id.replace("/", "_").replace("\\", "_")
             filename = f"chat_info_{safe_session_id}.json"
             filepath = CHATS_DATA_DIR / filename
             
             if not filepath.exists():
                 return False
                 
             try:
                 chat_dict = {
                     "messages": [{"role": msg["role"], "content": msg["content"]} for msg in self.chat_sessions[session_id]]
                 }
                 
                 with open(filepath, "w", encoding="utf-8") as f:
                     json.dump(chat_dict, f, indent=2, ensure_ascii=False)
                     
                 if log_timing:
                     logger.info("[TIMING] save_session %s: %.3fs", session_id, time.perf_counter() - t0)
                     
             except OSError as e:
                 logger.error("Failed to save session %s to disk: %s", session_id, e)
                 return False
             except Exception as e:
                 logger.error("Unexpected error saving session %s to disk: %s", session_id, e)
                 return False