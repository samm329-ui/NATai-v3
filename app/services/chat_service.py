import base64
import uuid
from pathlib import Path
from typing import Optional, Dict, Iterator, Any, Union
import json
import threading
import time

from app.utils.time_info import get_time_information
from app.config import (
    CHATS_DATA_DIR,
    CAMERA_CAPTURES_DIR,
    MAX_CHAT_HISTORY_TURNS,
    GROQ_API_KEYS,
)
from app.utils.key_rotation import get_next_key_pair
from app.services.groq_service import GroqService
from app.services.realtime_service import RealtimeGroqService
from app.services.brain_service import BrainService
from app.services.task_manager import TaskManager
from app.services.task_executor import TaskExecutor
from app.services.vector_store import VectorStoreService
from app.services.vision_service import VisionService
from app.services.decision_types import (
    CATEGORY_GENERAL,
    CATEGORY_REALTIME,
    CATEGORY_CAMERA,
    CATEGORY_TASK,
    CATEGORY_MIXED,
    HEAVY_INTENTS,
    INSTANT_INTENTS,
)

import logging

logger = logging.getLogger("N.A.T.A.S.H.A")

SAVE_EVERY_N_CHUNKS = 5

CAMERA_BYPASS_TOKEN = "[[CAMERABYPASS]]"


def save_camera_image_log(base64_str, session_id: str) -> Optional[Path]:
    """Save an image received as base64 to the disk for debugging/logging."""
    if not base64_str or not CAMERA_CAPTURES_DIR:
        return None

    try:
        raw = base64_str.split(",")[1] if "," in base64_str else base64_str

        data = base64.b64decode(raw)
        if len(data) < 50:
            logger.warning(
                "[VISION] Captured image very small (%d bytes), may be invalid",
                len(data),
            )
            return None

        ts = time.strftime("%Y%m%d_%H%M%S")
        s_id = (session_id or "na").replace(" ", "_")
        filename = f"cam_cap_{ts}_{s_id}.jpg"
        filepath = CAMERA_CAPTURES_DIR / filename

        filepath.write_bytes(data)
        logger.info(
            f"[VISION] Saved camera capture: %s (%d bytes) -> %s",
            filepath.name,
            len(data),
            filepath,
        )
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
                    role = msg.get("role", "user")
                    content = msg.get("content") or ""
                    if role in ("user", "assistant"):
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
            filepath = self._get_filepath(new_session_id)
            try:
                filepath.parent.mkdir(parents=True, exist_ok=True)
                with open(filepath, "w", encoding="utf-8") as f:
                    json.dump({"messages": []}, f)
            except Exception as e:
                logger.warning(
                    "Failed to create session file for %s: %s", new_session_id, e
                )
            logger.info(
                "[TIMING] session get_or_create: %.3fs (new)", time.perf_counter() - t0
            )
            return new_session_id

        if not self.validate_session_id(session_id):
            raise ValueError(
                f"Invalid session ID format: {session_id}. Session ID must be non-empty, not contain path traversal characters, and be under 255 characters."
            )

        if session_id in self.chat_sessions:
            logger.info(
                "[TIMING] session get_or_create: %.3fs (memory)",
                time.perf_counter() - t0,
            )
            return session_id

        if self._load_session_from_disk(session_id):
            logger.info(
                "[TIMING] session get_or_create: %.3fs (disk)", time.perf_counter() - t0
            )
            return session_id

        self.chat_sessions[session_id] = []
        logger.info(
            "[TIMING] session get_or_create: %.3fs (new disk)", time.perf_counter() - t0
        )
        return session_id

    def add_message(self, session_id: str, role: str, content: str):
        if session_id not in self.chat_sessions:
            self.get_or_create_session(session_id)

        self.chat_sessions[session_id].append({"role": role, "content": content})

    def get_chat_history(self, session_id: str) -> list[Dict[str, str]]:
        if session_id not in self.chat_sessions:
            self._load_session_from_disk(session_id)

        return self.chat_sessions.get(session_id, [])

    def format_history_for_llm(
        self, session_id: str, exclude_last: bool = False
    ) -> list[tuple]:
        history = self.get_chat_history(session_id)

        messages_to_process = history[:-1] if exclude_last and history else history

        merged = []
        i = 0
        while i < len(messages_to_process):
            msg = messages_to_process[i]
            user_msg = None
            ai_msg = None

            if (
                msg.get("role") == "user"
                and i + 1 < len(messages_to_process)
                and messages_to_process[i + 1].get("role") == "assistant"
            ):
                user_msg = msg.get("content") or ""
                ai_msg = messages_to_process[i + 1].get("content") or ""
                merged.append((user_msg, ai_msg))
                i += 2
            else:
                if msg.get("role") == "user":
                    merged.append((msg.get("content") or "", ""))
                i += 1

        if len(merged) > MAX_CHAT_HISTORY_TURNS:
            merged = merged[-MAX_CHAT_HISTORY_TURNS:]

        return merged

    def process_message_sync(self, session_id: str, user_message: str) -> str:
        if not self.groq_service:
            raise RuntimeError("Service is not initialized.")

        logger.info(
            "[GENERAL STREAM] Session ID: %s | User: %s", session_id, user_message
        )
        self.add_message(session_id, "user", user_message)

        chat_history = self.format_history_for_llm(session_id, exclude_last=True)

        key_start_index, _ = get_next_key_pair(len(GROQ_API_KEYS), need_brain=False)
        response = self.groq_service.get_response(
            user_message,
            chat_history=chat_history,
            key_start_index=key_start_index or 0,
        )

        self.add_message(session_id, "assistant", response)
        self.save_chat_session(session_id)

        logger.info(
            "[GENERAL STREAM] Response length: %d chars | Preview: %.100s",
            len(response),
            response,
        )
        return response

    def process_thinking_stream(
        self,
        session_id: str,
        user_message: str,
        force_clarify: bool = False,
        clarification_choice: Optional[str] = None,
    ) -> Iterator[Union[str, Dict[Any, Any]]]:
        if not self.brain_service:
            raise RuntimeError("Brain service is not initialized.")

        logger.info(
            "[THINKING STREAM] Session ID: %s | User: %s", session_id, user_message
        )

        self.add_message(session_id, "user", user_message)

        yield {"activity": "event", "query_detected": True, "message": user_message}
        yield {"activity": "event", "routing": True, "route": "thinking"}

        chat_history = self.format_history_for_llm(session_id, exclude_last=True)

        if clarification_choice:
            key_idx, _ = get_next_key_pair(len(GROQ_API_KEYS), need_brain=False)
            category, _, method, elapsed_ms = self.brain_service.classify(
                f"{user_message}\n\nUser chose: {clarification_choice}",
                chat_history,
                key_index=key_idx or 0,
            )
            yield {
                "activity": "event",
                "clarification_confirmed": True,
                "choice": clarification_choice,
            }

            for chunk in self.process_message_stream(session_id, user_message):
                if isinstance(chunk, dict):
                    yield chunk
                    continue
                yield chunk
            return

        key_idx, chat_idx = get_next_key_pair(
            len(GROQ_API_KEYS), need_brain=bool(self.brain_service)
        )

        t0 = time.perf_counter()
        category, task_types, primary_method, elapsed_ms = self.brain_service.classify(
            user_message,
            chat_history,
            key_index=key_idx if key_idx is not None else 0,
        )

        yield {
            "activity": "event",
            "decision_query_type": True,
            "category": category,
            "primary_method": primary_method.capitalize(),
            "elapsed_ms": elapsed_ms,
        }

        from app.config import (
            THINKING_MODE_ENABLED,
            CLARIFICATION_THRESHOLD,
        )

        broad_queries = [
            "analysis",
            "market",
            "stock",
            "help",
            "something",
            "anything",
            "tell me about",
            "what about",
            "give me",
            "create",
            "make",
        ]
        is_broad = (
            any(word in user_message.lower() for word in broad_queries)
            and len(user_message.split()) < 10
        )

        needs_clarification = (
            force_clarify
            or (THINKING_MODE_ENABLED and category in ["general", "realtime"])
            or is_broad
        )

        if needs_clarification and not task_types:
            clarify_result = self.brain_service.assess_clarification_need(
                user_message, chat_history
            )

            if clarify_result and clarify_result.get("needs_clarification"):
                yield {
                    "activity": "event",
                    "clarification_requested": True,
                    "question": clarify_result.get("question"),
                    "options": clarify_result.get("options", []),
                }
                yield {
                    "type": "clarification",
                    "question": clarify_result.get("question"),
                    "options": clarify_result.get("options", []),
                }
                return

        for chunk in self.process_message_stream(session_id, user_message):
            if isinstance(chunk, dict):
                yield chunk
                continue
            yield chunk

    def process_realtime_stream(
        self, session_id: str, user_message: str
    ) -> Iterator[Any]:
        if not self.realtime_service:
            raise RuntimeError(
                "Realtime service is not initialized. Cannot process realtime queries."
            )

        logger.info(
            "[REALTIME STREAM] Session ID: %s | User: %s", session_id, user_message
        )
        self.add_message(session_id, "user", user_message)

        chat_history = self.format_history_for_llm(session_id, exclude_last=True)

        yield {"activity": "event", "query_detected": True, "message": user_message}
        yield {"activity": "event", "routing": True, "route": "realtime"}

        chat_idx, _ = get_next_key_pair(len(GROQ_API_KEYS), need_brain=False)
        chunk_count = 0

        try:
            for chunk in self.realtime_service.stream_response(
                question=user_message,
                chat_history=chat_history,
                key_start_index=chat_idx or 0,
            ):
                if isinstance(chunk, dict):
                    yield chunk
                    continue

                self.chat_sessions[session_id][-1]["content"] += chunk
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

        logger.info(
            "[NATASHA STREAM] Session ID: %s | User: %s", session_id, user_message
        )
        if img_base64:
            logger.info(
                "[NATASHA STREAM] Request contains image payload (len: %d)",
                len(img_base64),
            )

        self.add_message(session_id, "user", user_message)

        yield {"activity": "event", "query_detected": True, "message": user_message}

        chat_history = self.format_history_for_llm(session_id, exclude_last=True)

        if img_base64:
            yield {
                "activity": "event",
                "query_type": True,
                "category": "reasoning",
                "message": "Analyzing image...",
            }
            yield {"activity": "event", "routing": True, "route": "vision"}

            save_camera_image_log(img_base64, session_id)

            t_vis = time.perf_counter()
            text = self.vision_service.describe_image(img_base64, user_message)

            if not text:
                yield "Vision is not available. Please set GROQ_API_KEY."
                return

            self.add_message(session_id, "assistant", text)
            self.save_chat_session(session_id)

            yield {"activity": "event", "streaming_started": True, "route": "vision"}
            yield text
            return

        brain_idx, chat_idx = get_next_key_pair(
            len(GROQ_API_KEYS), need_brain=bool(self.brain_service)
        )

        t0 = time.perf_counter()
        category, task_types, primary_method, elapsed_ms = self.brain_service.classify(
            user_message,
            chat_history,
            key_index=brain_idx if brain_idx is not None else 0,
        )

        yield {
            "activity": "event",
            "decision_query_type": True,
            "category": category,
            "primary_method": primary_method.capitalize(),
            "elapsed_ms": elapsed_ms,
        }

        if category == CATEGORY_CAMERA:
            yield {"activity": "event", "routing": True, "route": "camera"}

            prompt = (
                user_message
                if user_message != CAMERA_BYPASS_TOKEN
                else "What do you see in this image?"
            )

            text = "Let me take a look..."
            yield text
            yield {
                "type": "payload",
                "payload": {
                    "play": [],
                    "images": [],
                    "contents": [],
                    "actions": [
                        {"action": "open_capture", "payload": {"message": user_message}}
                    ],
                },
            }
            yield {
                "activity": "event",
                "action_method": True,
                "message": "camera (auto-capture)",
            }
            return

        if category in (CATEGORY_TASK, CATEGORY_MIXED):
            yield {
                "activity": "event",
                "routing": True,
                "route": "task" if category == CATEGORY_TASK else "mixed",
            }

            task_types = []
            intents = []
            task_name = "unknown"
            if self.brain_service:
                t_task_type = time.perf_counter()
                task_types, method_task, _ = self.brain_service.classify_task(
                    user_message,
                    chat_history,
                    key_index=brain_idx if brain_idx is not None else 0,
                )
                task_name = task_types[0] if task_types else "unknown"
                intents = self.brain_service.extract_task_payloads(
                    user_message, task_types, chat_history
                )

            yield {
                "activity": "event",
                "intent_classified": True,
                "intent": task_name,
            }

            heavy_intents = [t for t, p in intents if t in HEAVY_INTENTS]
            instant_intents = [(t, p) for t, p in intents if t in INSTANT_INTENTS]
            instant_response = None

            if self.task_manager and heavy_intents:
                yield {
                    "activity": "event",
                    "tasks_executing": True,
                    "message": f"Dispatching background tasks...",
                }
                bg_task_id = self.task_manager.submit(
                    task_name, {"message": user_message}, chat_history
                )
                yield {
                    "activity": "event",
                    "background_dispatched": True,
                    "message": f"Task queued ({task_name})",
                }

            elif self.task_executor and instant_intents:
                yield {
                    "activity": "event",
                    "tasks_executing": True,
                    "message": f"Running instant tasks...",
                }
                instant_response = self.task_executor.execute_intents(
                    instant_intents, chat_history
                )
                yield {
                    "activity": "event",
                    "tasks_completed": True,
                    "message": f"Instant tasks done",
                }

            if instant_response and (
                instant_response.actions
                or instant_response.plays
                or instant_response.images
                or instant_response.google_searches
                or instant_response.youtube_searches
                or instant_response.cam
            ):
                action_summary = []

                if instant_response.plays:
                    action_summary.append("play")
                if instant_response.images:
                    action_summary.append("image")
                if instant_response.google_searches:
                    action_summary.append("search")
                if instant_response.youtube_searches:
                    action_summary.append("youtube")
                if instant_response.cam:
                    action_summary.append("camera")

                import dataclasses

                payload_dict = dataclasses.asdict(instant_response)
                yield {"type": "payload", "payload": payload_dict}
                yield {
                    "activity": "event",
                    "actions_emitted": True,
                    "message": f"[{', '.join(action_summary)}]",
                }

            bg_task_ids = []

            if self.task_manager and heavy_intents:
                yield {
                    "activity": "event",
                    "tasks_executing": True,
                    "message": f"Running {task_name}...",
                }
                bg_task_ids.append(
                    self.task_manager.submit(
                        task_name, {"message": user_message}, chat_history
                    )
                )
                yield {
                    "activity": "event",
                    "tasks_completed": True,
                    "message": f"Tasks in background",
                }

            elif self.task_executor and not instant_intents:
                sync_response = self.task_executor.execute_intents(
                    intents, chat_history
                )

                if sync_response.actions or sync_response.contents:
                    import dataclasses

                    sync_payload = dataclasses.asdict(sync_response)
                    yield {"type": "payload", "payload": sync_payload}

                instant_response = instant_response or sync_response

            text_parts = []

            if instant_response and instant_response.text:
                text_parts.append(instant_response.text)

            bg_labels = []
            if bg_task_ids:
                for id in bg_task_ids:
                    bg_labels.append("generating content")
                text_parts.append(
                    f"I'm {' and '.join(bg_labels)} in the background. I'll open it for you when it's ready."
                )

                final_text = "\n".join(text_parts)
                if final_text:
                    self.add_message(session_id, "assistant", final_text)
                    self.save_chat_session(session_id)
                    yield final_text
                return

        use_realtime = category == CATEGORY_REALTIME and self.realtime_service
        stream_src = self.realtime_service if use_realtime else self.groq_service

        yield {
            "activity": "event",
            "routing": True,
            "route": "realtime" if use_realtime else "general",
        }
        yield {
            "activity": "event",
            "streaming_started": True,
            "route": "realtime" if use_realtime else "general",
        }

        t0 = time.perf_counter()
        chunk_count = 0

        try:
            for chunk in stream_src.stream_response(
                question=user_message,
                chat_history=chat_history,
                key_start_index=chat_idx,
            ):
                if isinstance(chunk, dict):
                    yield chunk
                    continue

                if chunk_count == 0:
                    elapsed_ms = int((time.perf_counter() - t0) * 1000)
                    yield {
                        "activity": "event",
                        "first_chunk": True,
                        "route": "realtime" if use_realtime else "general",
                        "elapsed_ms": elapsed_ms,
                    }

                self.chat_sessions[session_id][-1]["content"] += chunk
                chunk_count += 1

                if chunk_count % SAVE_EVERY_N_CHUNKS == 0:
                    self.save_chat_session(session_id, log_timing=False)

                yield chunk

        finally:
            self.save_chat_session(session_id)

            final_response = self.chat_sessions[session_id][-1].get("content", "")
            elapsed_time = time.perf_counter() - t0
            logger.info(
                "[NATASHA STREAM] stream complete in %.2fs: length %d, chunks %d, route %s",
                elapsed_time,
                len(final_response),
                chunk_count,
                "realtime" if use_realtime else "general",
            )

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
                    "messages": [
                        {"role": msg["role"], "content": msg["content"]}
                        for msg in self.chat_sessions[session_id]
                    ]
                }

                with open(filepath, "w", encoding="utf-8") as f:
                    json.dump(chat_dict, f, indent=2, ensure_ascii=False)

                if (
                    self.vector_store
                    and len(self.chat_sessions.get(session_id, [])) >= 2
                ):
                    messages = self.chat_sessions[session_id]
                    user_msg = ""
                    ai_msg = ""
                    for msg in messages[-2:]:
                        if msg.get("role") == "user":
                            user_msg = msg.get("content", "")
                        elif msg.get("role") == "assistant":
                            ai_msg = msg.get("content", "")
                    if user_msg or ai_msg:
                        self.vector_store.add_chat_memory(session_id, user_msg, ai_msg)

                if log_timing:
                    logger.info(
                        "[TIMING] save_session %s: %.3fs",
                        session_id,
                        time.perf_counter() - t0,
                    )

            except OSError as e:
                logger.error("Failed to save session %s to disk: %s", session_id, e)
                return False
            except Exception as e:
                logger.error(
                    "Unexpected error saving session %s to disk: %s", session_id, e
                )
                return False
