from pydantic import BaseModel, Field
from typing import List, Optional, Any, Dict


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=32_000)
    session_id: Optional[str] = None
    tts: bool = False
    img_base64: Optional[str] = None
    mode: Optional[str] = None
    thinking_mode: bool = False
    force_clarify: bool = False
    clarification_choice: Optional[str] = None


class ChatResponse(BaseModel):
    response: str
    session_id: str


class ChatHistory(BaseModel):
    session_id: str
    messages: List[ChatMessage]


class NatashaIntent(BaseModel):
    actions: List[str] = []
    plays: List[str] = []
    images: List[str] = []
    contents: List[str] = []
    google_searches: List[str] = []
    youtube_searches: List[str] = []
    cam: Optional[Dict] = None


class TTSRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000)
