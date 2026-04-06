import time
from typing import Iterator, Optional, Any, Tuple, List

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from app.services.groq_service import GroqService, escape_curly_braces, AllowableApiFailedError
from app.utils.retry import with_retry
from app.utils.time_info import get_time_information
from app.config import (
    REALTIME_WEB_SEARCH_ADDENDUM,
    TAVILY_API_KEY,
    GROQ_API_KEYS
)
from app.services.decision_types import INTENT_CLASSIFY_MODEL

import logging
logger = logging.getLogger("J.A.R.V.I.S")

QUERY_EXTRACTION_PROMPT = """
You are a search query optimizer. Convert the user's message into a clean, focused search query.
Only output the exact search query, no other text.

* Remove filler words (like "who is", "what is", "tell me about", "search for")
* Focus on the core entities and keywords
* Add specific details (today, 2024, exact names, full names)
* For weather queries, include location (e.g. "weather New York today")
* For sports queries, include specific teams or events (e.g. "Real Madrid score today")
* Resolve references (him, that, it) from conversation history
* Correct misspellings if they are obvious

Context: {history}
User Message: "{question}"
Search Query:
"""

class RealtimeGroqService(GroqService):
    def __init__(self, vector_store_service):
        super().__init__(vector_store_service)
        
        self.tavily_client = None
        if TAVILY_API_KEY:
            try:
                from tavily import TavilyClient
                self.tavily_client = TavilyClient(api_key=TAVILY_API_KEY)
                logger.info("Tavily search client initialized successfully")
            except Exception as e:
                logger.warning("Failed to init Tavily: %s", e)
        else:
            logger.warning("TAVILY_API_KEY not set. Realtime search will be unavailable.")
            
        if GROQ_API_KEYS:
             from langchain_groq import ChatGroq
             from app.config import (
                 GROQ_MODEL, MAX_TOKENS_GROQ, REQUEST_TIMEOUT_FAST
             )
             try:
                 self.fast_llm = ChatGroq(
                     groq_api_key=GROQ_API_KEYS[0],
                     model_name=INTENT_CLASSIFY_MODEL,
                     temperature=0.0,
                     max_tokens=100,
                     timeout=REQUEST_TIMEOUT_FAST,
                 )
             except Exception as e:
                 logger.error("Failed to init fast_llm: %s", e)
                 self.fast_llm = None
        else:
            self.fast_llm = None
            
    def _extract_search_query(
        self, question: str, chat_history: Optional[List[Tuple]] = None
    ) -> str:
        if not self.fast_llm:
             return question
             
        q_lower = question.strip().lower()
        
        has_filler = any(w in q_lower for w in [
            "what", "who", "where", "when", "why", "how",
            "is", "are", "do", "does", "did",
            "can", "could", "would", "will", "should",
            "tell", "know", "find", "search", "look"
        ])
        
        if len(q_lower) < 30 and not has_filler:
            return question
            
        try:
             history_context = ""
             if chat_history:
                 recent = chat_history[-3:]
                 parts = []
                 for h, a in recent:
                      parts.append(f"User: {h[:100]}")
                      parts.append(f"AI: {a[:100]}")
                 history_context = "\n".join(parts)
                 
             full_prompt = QUERY_EXTRACTION_PROMPT.format(
                 history=history_context,
                 question=question
             )
             
             t0 = time.perf_counter()
             response = self.fast_llm.invoke(full_prompt)
             extracted = response.content.strip()
             
             if extracted and len(extracted) < 100:
                  logger.info("[REALTIME] Query extraction: '%s' -> '%s' (%.3fs)", question, extracted, time.perf_counter() - t0)
                  return extracted
                  
             logger.warning("[REALTIME] Query extraction returned unusable result, using raw question")
             return question
             
        except Exception as e:
             logger.warning("[REALTIME] Query extraction failed: %s, using raw question", e)
             return question
             
    def prefetch_web_search(self, query: str, num_results: int = 5) -> str:
        if not self.tavily_client:
            logger.warning("Tavily client not initialized. TAVILY_API_KEY not set.")
            return ""
            
        if not query:
            return ""
            
        t0 = time.perf_counter()
        
        try:
             results = self.tavily_client.search(
                 query=query,
                 search_depth="basic",
                 include_answer=True,
                 include_raw_content=False,
                 max_results=num_results
             )
             
             formatted = ""
             if results and isinstance(results, dict):
                 ai_answer = results.get("answer", "")
                 if ai_answer:
                      formatted += f"**AI SYNTHESIZED ANSWER (use this as your primary source):**\n{ai_answer}\n\n"
                      
                 if "results" in results and results["results"]:
                     formatted += "**INDIVIDUAL SOURCES:**\n"
                     for r in results["results"]:
                          title = r.get("title", "No Title")
                          content = r.get("content", "")
                          url = r.get("url", "")
                          score = r.get("score", 0)
                          
                          if content:
                               formatted += f"- **{title}** (Relevance: {score:.2f})\n"
                               formatted += f"  {content}\n"
                               if url:
                                   formatted += f"  URL: {url}\n"
                               formatted += "\n"
                               
             if formatted:
                 logger.info("[REALTIME] Web search completed in %.2fs. Found %d results.", time.perf_counter() - t0, len(results.get("results", [])))
                 return formatted
                 
             logger.info("[REALTIME] Web search returned no useful results.")
             return ""
             
        except Exception as e:
             logger.error("Error during real-time web search: %s", e)
             return ""
             
    def stream_response(
        self,
        question: str,
        chat_history: Optional[List[Tuple]] = None,
        key_start_index: int = 0
    ) -> Iterator[Any]:
    
        yield {"activity": "event", "extracting_query", "message": "Extracting search query..."}
        
        search_query = self._extract_search_query(question, chat_history)
        
        yield {"activity": "event", "query_extracted", "message": f"Searching web for: '{search_query}'"}
        yield {"activity": "event", "searching_web", "message": f"Searching web for '{search_query}'"}
        
        formatted_results = self.prefetch_web_search(search_query)
        
        if formatted_results:
             yield {"activity": "event", "search_completed", "message": f"Search returned {len(formatted_results)} chars of content"}
        else:
             yield {"activity": "event", "search_failed", "message": f"Search returned no results"}
             
        prompt, messages = self._build_prompt_and_messages(
            question, chat_history, add_addendum=REALTIME_WEB_SEARCH_ADDENDUM
        )
        
        if formatted_results:
             extra_parts = f"\n\nReal-Time Web Search Results for '{search_query}':\n"
             extra_parts += escape_curly_braces(formatted_results)
             
             prompt.template += extra_parts
             
        try:
             yield from self._stream_llm(prompt, messages, question, key_start_index=key_start_index)
        except AllowableApiFailedError:
             raise
        except Exception as e:
             logger.error("Error streaming realtime response: %s", e)
             raise RuntimeError(f"Error streaming realtime response: {e}") from e