import json
from typing import List, Optional, Tuple, Iterator, Any
import time

from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import PromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser

from app.utils.time_info import get_time_information
from app.services.vector_store import VectorStoreService
from app.utils.retry import with_retry
from app.utils.key_rotation import get_next_key_pair
from app.config import (
    GENERAL_CHAT_ADDENDUM,
    NATASHA_SYSTEM_PROMPT,
    GROQ_API_KEYS, GROQ_API_KEY_VISION
)

import logging
logger = logging.getLogger("N.A.T.A.S.H.A")

ALL_APIS_FAILED_MESSAGE = \
    "I'm unable to process your request at the moment. All API services are " \
    "temporarily unavailable. Please try again in a few minutes."
    
class AllowableApiFailedError(Exception):
    pass

def escape_curly_braces(text: str) -> str:
    if not text:
        return ""
    return text.replace("{", "{{").replace("}", "}}")

REPEAT_WINDOW = 500
REPEAT_THRESHOLD = 5

def detect_repetition_loop(text: str) -> bool:
    if len(text) < REPEAT_WINDOW:
        return False
        
    tail = text[-REPEAT_WINDOW:]
    
    for i in range(1, 100):
        phrase = tail[-i:]
        if tail.count(phrase) > REPEAT_THRESHOLD:
            return True
            
    return False

def truncate_at_repetition(text: str) -> str:
    if not detect_repetition_loop(text):
        return text
        
    for i in range(1, 100):
        phrase = text[-i:]
        if text.count(phrase) > REPEAT_THRESHOLD:
            first = text.find(phrase)
            return text[:first] + "... [Repetition cut]"
            
    return text
    
def is_rate_limit_error(e: Exception) -> bool:
    msg = str(e).lower()
    return "429" in msg or "rate limit" in msg or "tokens per day" in msg

class GroqService:
    def __init__(self, vector_store_service: VectorStoreService):
        self.llms = []
        self._init_llms()
        
        self.vector_store = vector_store_service
        self.prompt_template = PromptTemplate.from_template(
            NATASHA_SYSTEM_PROMPT
        )
        self.chain = None
        
    def _init_llms(self):
        if not GROQ_API_KEYS:
            logger.error("No GROQ API keys configured. Set GROQ_API_KEY (and optionally GROQ_API_KEY_1, ...) in .env")
            return
            
        from langchain_groq import ChatGroq
        from app.config import GROQ_MODEL, MAX_TOKENS_GROQ, REQUEST_TIMEOUT_FAST
        
        for key in GROQ_API_KEYS:
            try:
                llm = ChatGroq(
                    groq_api_key=key,
                    model_name=GROQ_MODEL,
                    temperature=0.5,
                    max_tokens=1024,
                    timeout=REQUEST_TIMEOUT_FAST,
                    model_kwargs={"frequency_penalty": 0.3},
                )
                self.llms.append(llm)
            except Exception as e:
                logger.error("Failed to initialize ChatGroq with a key: %s", e)
                
        if self.llms:
            logger.info("Initialized GroqService with %d API key(s) (primary-first fallback)", len(self.llms))
            
    def _invoke_llm(
        self,
        prompt: PromptTemplate,
        messages: List,
        question: str,
        key_start_index: int = 0
    ) -> str:
        if not self.llms:
             raise ValueError("No LLMs initialized")
             
        last_exc = None
        n = len(self.llms)
        
        for i in range(n):
             idx = (key_start_index + i) % n
             llm = self.llms[idx]
             masked_key = f"***{GROQ_API_KEYS[idx][-4:]}"
             
             try:
                 chain = prompt | llm | StrOutputParser()
                 
                 logger.debug("Trying API key #%d (ending in %s)", idx + 1, masked_key)
                 
                 res = chain.invoke({"history": messages, "question": question})
                 
                 if res and detect_repetition_loop(res):
                     logger.warning("[GROQ] Repetition loop detected - truncating response (50 chars): %s", res[:50])
                     return truncate_at_repetition(res)
                     
                 if i > 0:
                     logger.info("Fallback successful! API key #%d (%s) succeeded.", idx + 1, masked_key)
                     
                 return res
                 
             except Exception as e:
                 last_exc = e
                 if is_rate_limit_error(e):
                     logger.warning("API key #%d (%s) rate limited.", idx + 1, masked_key)
                 else:
                     logger.warning("API key #%d (%s) failed: %s", idx + 1, masked_key, str(e)[:100])
                     
                 if i < n - 1:
                     logger.info("Falling back to next API key...")
                     time.sleep(0.5)
                     
        logger.error("All %d API key(s) failed during invoke().", n)
        raise AllowableApiFailedError(ALL_APIS_FAILED_MESSAGE)
        
    def _stream_llm(
        self,
        prompt: PromptTemplate,
        messages: List,
        question: str,
        key_start_index: int = 0
    ) -> Iterator[str]:
        if not self.llms:
             raise ValueError("No LLMs initialized")
             
        last_exc = None
        n = len(self.llms)
        
        for i in range(n):
             idx = (key_start_index + i) % n
             llm = self.llms[idx]
             masked_key = f"***{GROQ_API_KEYS[idx][-4:]}"
             
             try:
                 chain = prompt | llm | StrOutputParser()
                 
                 logger.debug("Streaming with API key #%d (%s)", idx + 1, masked_key)
                 
                 stream_start = time.perf_counter()
                 chunk_count = 0
                 accumulated = ""
                 repetition_stopped = False
                 
                 last_check_len = 0
                 REPEAT_CHECK_INTERVAL = 100
                 
                 for chunk in chain.stream({"history": messages, "question": question}):
                     yield chunk
                     chunk_count += 1
                     accumulated += chunk
                     
                     if len(accumulated) - last_check_len > REPEAT_CHECK_INTERVAL:
                         last_check_len = len(accumulated)
                         if detect_repetition_loop(accumulated):
                             logger.warning("[GROQ] Repetition loop detected after 50 chars - stopping: %s", accumulated[:50])
                             repetition_stopped = True
                             break
                             
                 total_stream_time = time.perf_counter() - stream_start
                 logger.info("Stream total: %.2fs, chunks: %d%s", total_stream_time, chunk_count, ", TRUNCATED: REPETITION" if repetition_stopped else "")
                 
                 if i > 0 and chunk_count > 0:
                     logger.info("Fallback successful! API key #%d (%s) succeeded.", idx + 1, masked_key)
                 return
                 
             except Exception as e:
                 last_exc = e
                 if is_rate_limit_error(e):
                     logger.warning("API key #%d (%s) rate limited.", idx + 1, masked_key)
                 else:
                     logger.warning("API key #%d (%s) failed: %s", idx + 1, masked_key, str(e)[:100])
                     
                 if i < n - 1:
                     logger.info("Falling back to next API key for stream...")
                     time.sleep(0.5)
                     
        logger.error("All %d API key(s) failed during stream().", n)
        raise AllowableApiFailedError(ALL_APIS_FAILED_MESSAGE)
        
    def _build_prompt_and_messages(
        self,
        question: str,
        chat_history: Optional[List[Tuple]] = None,
        add_addendum: str = "",
    ) -> Tuple[PromptTemplate, List]:
        
        context_docs = []
        context = ""
        context_sources = []
        
        try:
             retriever = self.vector_store.get_retriever()
             if retriever:
                 context_docs = retriever.invoke(question)
                 
                 context = "\n".join([doc.page_content for doc in context_docs])
                 context_sources = [doc.metadata.get("source", "unknown") for doc in context_docs]
                 logger.info("[CONTEXT] Retrieved relevant chunks for query")
                 
        except Exception as e:
            logger.warning("[VECTOR] Vector retrieval failed, using empty context: %s", e)
            
        time_info = get_time_information()
        
        system_message = NATASHA_SYSTEM_PROMPT
        if add_addendum:
             system_message += "\n\n" + add_addendum
             
        system_message = system_message.replace("{time_info}", time_info)
        
        if context:
            extra_parts = escape_curly_braces(context)
            system_message += f"\n\nContext Information:\n{extra_parts}"
            
        messages = []
        if chat_history:
             for human_msg, ai_msg in chat_history:
                  messages.append(HumanMessage(content=human_msg))
                  messages.append(AIMessage(content=ai_msg))
                  
        logger.info("[PROMPT] System message length: %d chars | History pairs: %d | Question: %.50s...", len(system_message), len(chat_history) if chat_history else 0, question)
        
        prompt = PromptTemplate.from_template(system_message)
        return prompt, messages

    @with_retry(max_retries=1)
    def get_response(
        self,
        question: str,
        chat_history: Optional[List[Tuple]] = None,
        key_start_index: int = 0
    ) -> str:
        try:
            prompt, messages = self._build_prompt_and_messages(
                question, chat_history, add_addendum=GENERAL_CHAT_ADDENDUM
            )
            
            t0 = time.perf_counter()
            result = self._invoke_llm(prompt, messages, question, key_start_index=key_start_index)
            
            logger.info("[RESPONSE] General chat: %d chars | Preview %.50s...", len(result), result)
            return result
            
        except AllowableApiFailedError:
            raise
        except Exception as e:
            logger.error("Error getting response from Groq: %s", e)
            raise RuntimeError(f"Error getting response from Groq: {e}") from e

    def stream_response(
        self,
        question: str,
        chat_history: Optional[List[Tuple]] = None,
        key_start_index: int = 0
    ) -> Iterator[str]:
        try:
            prompt, messages = self._build_prompt_and_messages(
                question, chat_history, add_addendum=GENERAL_CHAT_ADDENDUM
            )
            
            yield {"activity": "event", "context_retrieved", "message": "Retrieved relevant context from knowledge base"}
            
            yield from self._stream_llm(prompt, messages, question, key_start_index=key_start_index)
            
        except AllowableApiFailedError:
             raise
        except Exception as e:
             logger.error("Error streaming response from Groq: %s", e)
             raise RuntimeError(f"Error streaming response from Groq: {e}") from e