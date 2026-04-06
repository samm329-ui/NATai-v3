import logging
from typing import List, Optional, Any, Dict
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError as FuturesTimeoutError

from app.services.decision_types import (
    INTENT_OPEN_WEBCAM, INTENT_CLOSE_WEBCAM, INTENT_CAMERA,
    INTENT_PLAY, INTENT_GENERATE_IMAGE, INTENT_CONTENT,
    INTENT_GOOGLE_SEARCH, INTENT_YOUTUBE_SEARCH, INTENT_CHAT
)
from app.config import TASK_EXECUTION_TIMEOUT

logger = logging.getLogger("J.A.R.V.I.S")

@dataclass
class TaskResponse:
    text: str = ""
    actions: List[Dict[str, Any]] = field(default_factory=list)
    plays: List[str] = field(default_factory=list)
    images: List[str] = field(default_factory=list)
    contents: List[str] = field(default_factory=list)
    google_searches: List[str] = field(default_factory=list)
    youtube_searches: List[str] = field(default_factory=list)
    cam: Optional[Dict] = None
    
class TaskExecutor:
    def __init__(self, groq_service=None):
        self.groq_service = groq_service
        logger.info("[TASK] TaskExecutor initialized (Pollinations.ai for images)")
        
    def execute_intents(
        self,
        intents: List[tuple],
        chat_history: Optional[List[tuple]] = None
    ) -> TaskResponse:
        
        response = TaskResponse()
        tasks = []
        
        for intent_type, payload in intents:
            if intent_type == INTENT_OPEN_WEBCAM:
                 tasks.append(("webcam_open", lambda p=payload: self._do_open_payload(p)))
            elif intent_type == INTENT_PLAY:
                 tasks.append(("play", lambda p=payload: self._do_play_payload(p)))
            elif intent_type == INTENT_GENERATE_IMAGE:
                 tasks.append(("image", lambda p=payload: self._do_generate_image(p)))
            elif intent_type == INTENT_CONTENT:
                 tasks.append(("content", lambda p=payload: self._do_content(p, chat_history)))
            elif intent_type == INTENT_GOOGLE_SEARCH:
                 tasks.append(("google_search", lambda p=payload: self._do_google_search(p)))
            elif intent_type == INTENT_YOUTUBE_SEARCH:
                 tasks.append(("youtube_search", lambda p=payload: self._do_youtube_search(p)))
            elif intent_type == INTENT_CLOSE_WEBCAM:
                 response.text = "Webcam closed."
                 response.actions.append({"action": "close"})
            elif intent_type == INTENT_CAMERA:
                 response.text = "Opening the webcam for you. Once it's on, send your message again and I'll describe what I see."
                 response.cam = {"action": "open"}
            elif intent_type == INTENT_CHAT:
                 pass
                 
        if not tasks:
            if not response.text and not response.cam:
                 response.text = "I'm not sure what you'd like me to do. Could you clarify?"
            return response
            
        import time
        t0 = time.perf_counter()
        failed_tags = []
        
        try:
             with ThreadPoolExecutor(max_workers=min(len(tasks), 4)) as executor:
                  futures = {executor.submit(fn): tag for tag, fn in tasks}
                  
                  for future in as_completed(futures, timeout=TASK_EXECUTION_TIMEOUT):
                       tag = futures[future]
                       try:
                            result = future.result()
                            if not result:
                                failed_tags.append(tag)
                                continue
                                
                            if tag == "play":
                                response.plays.append(result)
                            elif tag == "image":
                                response.images.append(result)
                            elif tag == "content":
                                response.contents.append(result)
                            elif tag == "google_search":
                                response.google_searches.append(result)
                            elif tag == "youtube_search":
                                response.youtube_searches.append(result)
                                
                       except Exception as e:
                            failed_tags.append(tag)
                            err_msg = str(e)
                            logger.warning("[TASK] Task %s failed: %s", tag, err_msg)
                            
                            if "content policy" in err_msg.lower() or "safety" in err_msg.lower():
                                 response.text = "I couldn't generate that image. It may violate content guidelines."
                            else:
                                 response.text = f"Something went wrong with that task: {err_msg}"
                                 
        except FuturesTimeoutError:
             logger.warning("[TASK] Task execution timed out after %ds", TASK_EXECUTION_TIMEOUT)
             if not response.text:
                  response.text = "Some tasks took too long. Please try again."
                  
        elapsed = time.perf_counter() - t0
        logger.info("[TASK] Executed %d tasks in %.2fs. (failed: %d)", len(tasks), elapsed, len(failed_tags))
        
        if not response.text:
             response.text = self._build_conversational_response(response)
             
        return response
        
    def _url_to_display_name(self, url: str) -> str:
        mapping = {
            "facebook.com": "Facebook", "instagram.com": "Instagram", "youtube.com": "YouTube",
            "twitter.com": "Twitter", "x.com": "X", "linkedin.com": "LinkedIn",
            "reddit.com": "Reddit", "discord.com": "Discord",
            "github.com": "GitHub", "wikipedia.org": "Wikipedia", "stackoverflow.com": "Stack Overflow",
            "amazon.com": "Amazon", "google.com": "Google", "gmail.com": "Gmail",
            "whatsapp.com": "WhatsApp"
        }
        
        try:
             import urllib.parse
             parsed = urllib.parse.urlparse(url)
             domain = parsed.netloc.lower() or parsed.path.lower()
             domain = domain.replace("www.", "").split("/")[0]
             
             for key, name in mapping.items():
                 if key in domain:
                      return name
             return domain.split(".")[0].title() if "." in domain else "the link"
        except Exception:
             return "the link"
             
    def _build_conversational_response(self, response: TaskResponse) -> str:
        parts = []
        
        if response.plays:
            names = [self._url_to_display_name(u) for u in response.plays]
            if len(names) == 1:
                parts.append(f"I've started playing that for you.")
            else:
                last = names[-1]
                rest = ", ".join(names[:-1])
                parts.append(f"I've opened {rest} and {last} for you.")
                
        if response.images:
            count = len(response.images)
            parts.append(f"I've generated the image{'s' if count > 1 else ''} for you.")
            
        if response.contents:
            parts.append("I've written that for you.")
            
        if response.google_searches or response.youtube_searches:
             parts.append("I ran the search for you.")
             
        if not parts:
            return "Done."
            
        return " ".join(parts)
        
    def _validate_url(self, url: str) -> Optional[str]:
        if not url or not isinstance(url, str):
            return None
        url = url.strip()
        if not url.startswith("http"):
            return None
        try:
            import urllib.parse
            parsed = urllib.parse.urlparse(url)
            if parsed.scheme not in ("http", "https"):
                 logger.warning("[TASK] Rejected non-http URL: %s", url)
                 return None
            return url
        except Exception:
            return None

    def _do_open_payload(self, payload: dict) -> Optional[str]:
        return self._validate_url(payload.get("url", ""))

    def _do_play_payload(self, payload: dict) -> Optional[str]:
        return self._validate_url(payload.get("url", ""))
        
    def _do_generate_image(self, payload: dict) -> Optional[bytes]:
        prompt = payload.get("prompt", payload.get("message", "")).strip()
        if not prompt:
            return None
            
        if len(prompt) < 3:
            logger.warning("[TASK] Image prompt too short (< 3 chars)")
            return None
            
        return self._pollinate_image(prompt)
        
    def _pollinate_image(self, prompt: str) -> Optional[bytes]:
        import urllib.parse
        import httpx
        import time
        
        t0 = time.perf_counter()
        encoded_prompt = urllib.parse.quote(prompt)
        api_url = f"https://image.pollinations.ai/prompt/{encoded_prompt}?width=1024&height=1024&nologo=True&enhance=True&safe=False"
        
        logger.info("[TASK] Fetching Pollinations image: %s...", api_url[:100])
        
        try:
             with httpx.Client(timeout=30.0, follow_redirects=True) as client:
                  resp = client.get(api_url)
                  
                  if resp.status_code == 200 and len(resp.content) > 1000:
                       logger.info("[TASK] Pollinations image fetched (%d bytes) in %.2fs", len(resp.content), time.perf_counter() - t0)
                       return resp.content
                       
                  logger.warning("[TASK] Pollinations failed: HTTP %s, len=%d", resp.status_code, len(resp.content))
                  return None
                  
        except Exception as e:
             logger.warning("[TASK] Pollinations failed: %s", e)
             return None
             
    def _do_content(self, payload: dict, chat_history: Optional[List[tuple]] = None) -> Optional[str]:
        if not self.groq_service:
            return None
            
        prompt = payload.get("prompt", payload.get("message", "")).strip()
        if not prompt:
            return None
            
        content_question = f"Write the following. Be thorough and well-structured. Return only the requested content, no preamble.\n\n{prompt}"
        
        try:
             result = self.groq_service.get_response(
                 content_question,
                 chat_history=chat_history,
             )
             
             if not result or len(result.strip()) < 10:
                  logger.warning("[TASK] Content generation returned empty or very short result")
                  return None
                  
             return result
             
        except Exception as e:
             logger.warning("[TASK] Content generation error: %s", e)
             return None

    def _do_google_search(self, payload: dict) -> Optional[str]:
        query = payload.get("query", payload.get("message", "")).strip()
        if not query:
            return None
        import urllib.parse
        encoded = urllib.parse.quote(query)
        return f"https://www.google.com/search?q={encoded}"
        
    def _do_youtube_search(self, payload: dict) -> Optional[str]:
        query = payload.get("query", payload.get("message", "")).strip()
        if not query:
            return None
        import urllib.parse
        encoded = urllib.parse.quote(query)
        return f"https://www.youtube.com/results?search_query={encoded}"