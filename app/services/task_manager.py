import uuid
import time
import threading
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor

from app.services.decision_types import INTENT_GENERATE_IMAGE, INTENT_CONTENT

import logging
logger = logging.getLogger("J.A.R.V.I.S")

TASK_TTL = 3600

@dataclass
class TaskEntry:
    task_id: str
    status: str = "running"
    intent_type: str = ""
    prompt: str = ""
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    image_bytes: Optional[bytes] = None

class TaskManager:
    def __init__(self, task_executor):
        self.self_task_executor = task_executor
        self.tasks: Dict[str, TaskEntry] = {}
        self.lock = threading.Lock()
        self.pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="bg-task")
        logger.info("[TASK-MGR] Background task manager initialized (4 workers)")
        
    def submit(
        self,
        intent_type: str,
        payload: dict,
        chat_history: Optional[List[Tuple]] = None
    ) -> str:
        task_id = str(uuid.uuid4())[:8]
        prompt = payload.get("prompt", payload.get("message", ""))[:200]
        
        if intent_type == INTENT_GENERATE_IMAGE:
            label = "Generating image"
        elif intent_type == INTENT_CONTENT:
            label = "Writing content"
        else:
            label = "Processing task"
            
        entry = TaskEntry(
            task_id=task_id,
            status="running",
            intent_type=intent_type,
            prompt=prompt
        )
        
        with self.lock:
             self.tasks[task_id] = entry
             
        self.pool.submit(self._run, task_id, intent_type, payload, chat_history)
        logger.info("[TASK-MGR] Submitted bg task %s: %s, intent=%s", task_id, label, intent_type)
        return task_id
        
    def get(self, task_id: str) -> Optional[TaskEntry]:
        with self.lock:
             return self.tasks.get(task_id)
             
    def get_serializable(self, task_id: str) -> Optional[dict]:
        entry = self.get(task_id)
        if not entry:
            return None
            
        return {
            "task_id": entry.task_id,
            "status": entry.status,
            "intent_type": entry.intent_type,
            "prompt": entry.prompt,
            "result": entry.result,
            "error": entry.error
        }
        
    def _run(self, task_id: str, intent_type: str, payload: dict, chat_history):
        t0 = time.perf_counter()
        
        try:
            if intent_type == INTENT_GENERATE_IMAGE:
                img_result = self.self_task_executor._do_generate_image(payload)
                if img_result:
                    with self.lock:
                        entry = self.tasks[task_id]
                        entry.status = "completed"
                        entry.image_bytes = img_result
                        entry.result = {
                            "type": "image",
                            "url": f"/api/tasks/{task_id}/image",
                            "prompt": payload.get("prompt", payload.get("message", ""))
                        }
                else:
                    raise RuntimeError("Image generation returned no result. Check API key or content policy.")
                    
            elif intent_type == INTENT_CONTENT:
                text = self.self_task_executor._do_content(payload, chat_history)
                if text:
                     with self.lock:
                         entry = self.tasks[task_id]
                         entry.status = "completed"
                         entry.result = {
                             "type": "content",
                             "text": text,
                             "prompt": payload.get("prompt", payload.get("message", ""))
                         }
                else:
                     raise RuntimeError("Content generation returned no result.")
            else:
                 raise ValueError(f"Unsupported background task type: {intent_type}")
                 
            elapsed = time.perf_counter() - t0
            logger.info("[TASK-MGR] Task %s completed in %.2fs", task_id, elapsed)
            
        except Exception as e:
            with self.lock:
                entry = self.tasks[task_id]
                entry.status = "failed"
                entry.error = str(e)[:500]
            logger.warning("[TASK-MGR] Task %s failed: %s", task_id, e)
            
        self.cleanup_old()
        
    def cleanup_old(self):
        cutoff = time.time() - TASK_TTL
        with self.lock:
            to_remove = [tid for tid, e in self.tasks.items() if e.created_at < cutoff]
            for tid in to_remove:
                del self.tasks[tid]
        if to_remove:
            logger.info("[TASK-MGR] Cleaned up %d expired tasks", len(to_remove))
            
    def shutdown(self):
        logger.info("[TASK-MGR] Shutting down...")
        self.pool.shutdown(wait=False)
        logger.info("[TASK-MGR] Shutdown complete")