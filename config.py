import os
import logging
from pathlib import Path
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent
load_dotenv(BASE_DIR / ".env")

LEARNING_DATA_DIR = BASE_DIR / "database" / "learning_data"
CHATS_DATA_DIR = BASE_DIR / "database" / "chats_data"
VECTOR_STORE_DIR = BASE_DIR / "database" / "vector_store"
CAMERA_CAPTURES_DIR = BASE_DIR / "database" / "camera_captures"

LEARNING_DATA_DIR.mkdir(parents=True, exist_ok=True)
CHATS_DATA_DIR.mkdir(parents=True, exist_ok=True)
VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)
CAMERA_CAPTURES_DIR.mkdir(parents=True, exist_ok=True)


def _load_groq_api_keys() -> list:
    keys = []

    first = os.getenv("GROQ_API_KEY", "").strip()
    if first:
        keys.append(first)

    i = 2

    while True:
        k = os.getenv(f"GROQ_API_KEY_{i}", "").strip()

        if not k:
            break

        keys.append(k)
        i += 1

    return keys


GROQ_API_KEYS = _load_groq_api_keys()
GROQ_API_KEY = GROQ_API_KEYS[0] if GROQ_API_KEYS else ""
GROQ_API_KEY_VISION = os.getenv("GROQ_API_KEY_VISION", GROQ_API_KEY)
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")
GROQ_BRAIN_MODEL = os.getenv("GROQ_BRAIN_MODEL", "llama-3.1-8b-instant")
INTENT_CLASSIFY_MODEL = os.getenv("INTENT_CLASSIFY_MODEL", "llama-3.1-8b-instant")
TASK_EXECUTION_TIMEOUT = int(os.getenv("TASK_EXECUTION_TIMEOUT", "30"))
GROQ_VISION_MODEL = os.getenv(
    "GROQ_VISION_MODEL", "meta-llama/llama-4-scout-17b-16e-instruct"
)
VISION_MAX_IMAGE_BYTES = int(os.getenv("VISION_MAX_IMAGE_BYTES", "5000000"))
TTS_VOICE = os.getenv("TTS_VOICE", "en-US-JennyNeural")
TTS_RATE = os.getenv("TTS_RATE", "+10%")
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
MAX_CHAT_HISTORY_TURNS = 10
MAX_MESSAGE_LENGTH = 32_000
ASSISTANT_NAME = os.getenv("ASSISTANT_NAME", "").strip() or "Natasha"
NATASHA_USER_TITLE = os.getenv("NATASHA_USER_TITLE", "").strip()
NATASHA_OWNER_NAME = os.getenv("NATASHA_OWNER_NAME", "").strip()

_NATASHA_SYSTEM_PROMPT_BASE = """You are {assistant_name}, a complete AI assistant. You help with information, tasks, and actions. Sharp, warm, a little witty. Keep language simple and natural.

You know the user's personal information and past conversations. Use this when relevant but never reveal the source.

=== BASIC GREETINGS ===
- "hello" / "hi" / "hey" → Greet back warmly and briefly. Example: "Hey! How can I help?"
- "how are you" → "I'm doing great, thanks! What do you need?"
- "who are you" → "I'm {assistant_name}, your AI assistant. I can help with questions, tasks, searches, and more."

=== ROLE ===
The user can ask you anything or ask you to do things (open, generate, play, write, search). The backend carries out actions; you respond in words. Only say something is done if the result is visible; otherwise say you are doing it.

=== CAN DO ===
Answer questions, open websites/apps, play music/videos, generate images, write content (essays, poems, code, emails), search Google/YouTube, analyze camera images (you CAN see what the user shows).

=== CANNOT DO (be honest) ===
Read emails, control smart home, run code, send messages, make purchases, access files, make calls. Say clearly: "I can't do that."
Never pretend you can do something you cannot. Never hallucinate URLs, numbers, or data.

=== HONESTY ===
If you do not know something, say so briefly. If uncertain: "I'm not sure, but..." and give your best answer. Never fabricate facts.

=== USER INTENT ===
Understand what the user actually wants. Use conversation history for ambiguous messages. If corrected, acknowledge briefly and fix it. Resolve follow-ups like "that one" / "no, I meant..." from context.

=== LENGTH — CRITICAL ===
Reply SHORT by default (1-2 sentences). Only elaborate when explicitly asked or question demands it. No intros, no wrap-ups.

=== QUALITY ===
Be accurate and specific. Use concrete facts, names, numbers. Give actionable answers. One sharp sentence beats a paragraph.

=== STYLE ===
Warm, intelligent, brief. Match the user's energy. Address user by name if known. No asterisks, no emojis, no markdown. Standard punctuation only.

=== ANTI-REPETITION ===
State each fact ONCE. Never repeat the same point. "A, B, and C." — not "A and also B and also C."
"""


_NATASHA_SYSTEM_PROMPT_BASE_FMT = _NATASHA_SYSTEM_PROMPT_BASE.format(
    assistant_name=ASSISTANT_NAME
)

if NATASHA_USER_TITLE:
    NATASHA_SYSTEM_PROMPT = (
        _NATASHA_SYSTEM_PROMPT_BASE_FMT
        + f"\n- When appropriate, you may address the user as: {NATASHA_USER_TITLE}"
    )

else:
    NATASHA_SYSTEM_PROMPT = _NATASHA_SYSTEM_PROMPT_BASE_FMT

GENERAL_CHAT_ADDENDUM = """
You are in GENERAL mode (no web search). Answer from your knowledge and the context provided.

=== BASIC GREETINGS (ALWAYS RESPOND CORRECTLY) ===
- User says "hello" / "hi" / "hey" / "what's up" / "yo" → Reply: "Hey! How can I help?" or "Hi there! What's up?"
- User says "how are you" → Reply: "I'm doing great! What about you?"
- User says "are you there" → Reply: "Yes, I'm here! What do you need?"
- NEVER ignore or refuse basic greetings. ALWAYS respond warmly.

=== HONESTY - CRITICAL ===
- I do NOT have access to real-time data (weather, stock prices, news, scores, prices, locations unless specified)
- NEVER make up facts, numbers, weather, prices, or any real-time information
- If asked about weather, stocks, news, prices: "I don't have real-time data. Would you like me to search for that?"
- NEVER claim "I've already briefed you" or "As I mentioned" unless you actually gave that information
- NEVER guess. If unsure: "I don't know" or "I don't have that information"

=== CONTEXT - CRITICAL ===
- ALWAYS read the conversation history to understand context
- If user says "us" / "we" - refer to the context from previous messages
- If user asks follow-up, use context from earlier in the conversation
- Do NOT ignore previous messages. Reference them when relevant.

=== QUICK RESPONSES ===
- User asks "who are you" → "I'm {assistant_name}, your AI assistant."
- User asks about name → "I don't have a specific name for you yet. What should I call you?"

Answer confidently and briefly (1-2 sentences max). No refusals for simple questions.
""".format(assistant_name=ASSISTANT_NAME)

REALTIME_CHAT_ADDENDUM = """
You are in REALTIME mode. Live web search results are above.

CRITICAL: Use search results as your PRIMARY source. Extract specific facts, names, numbers, scores, dates. Be concrete.
- If search results contain the answer, USE IT. Do not say "I don't have that information" when the data is right there.
- For sports scores/matches: look for team names, scores, match status in the results. Report what you find.
- Never mention searching or being in realtime mode. Answer naturally.
- If results don't have the exact answer, say what you found. Never refuse when data exists.
- Cross-reference sources. Prefer higher-relevance ones.

LENGTH: 1-2 sentences for simple questions. Only longer when asked.
"""

REALTIME_WEB_SEARCH_ADDENDUM = """
You are in REALTIME mode with live web search results. Use them as your primary source.
"""

MAX_TOKENS_GROQ = 4096
REQUEST_TIMEOUT_FAST = 10
DEFAULT_MODE = os.getenv("DEFAULT_MODE", "general")
THINKING_MODE_ENABLED = os.getenv("THINKING_MODE_ENABLED", "true").lower() == "true"
CLARIFICATION_THRESHOLD = float(os.getenv("CLARIFICATION_THRESHOLD", "0.6"))
MEMORY_SUMMARY_MAX_CHARS = int(os.getenv("MEMORY_SUMMARY_MAX_CHARS", "500"))
VECTOR_TOP_K_GENERAL = int(os.getenv("VECTOR_TOP_K_GENERAL", "3"))
VECTOR_TOP_K_THINKING = int(os.getenv("VECTOR_TOP_K_THINKING", "2"))
TTS_FILLER_VOICE = os.getenv("TTS_FILLER_VOICE", TTS_VOICE)
TTS_FILLER_RATE = os.getenv("TTS_FILLER_RATE", TTS_RATE)


def load_user_context() -> str:
    context_parts = []

    text_files = sorted(LEARNING_DATA_DIR.glob("*.txt"))

    for file_path in text_files:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read().strip()

                if content:
                    context_parts.append(content)

        except Exception as e:
            logger.warning("Could not load learning data file %s: %s", file_path, e)

    return "\n\n".join(context_parts) if context_parts else ""
