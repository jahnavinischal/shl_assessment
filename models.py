from pydantic import BaseModel
from typing import List, Optional


class Message(BaseModel):
    role: str  # "user" or "assistant"
    content: str


class ChatRequest(BaseModel):
    messages: List[Message]


class Recommendation(BaseModel):
    name: str
    url: str
    test_type: str   # single letter
    keys: Optional[List[str]] = None       
    duration: Optional[str] = None          
    languages: Optional[List[str]] = None  


class ChatResponse(BaseModel):
    reply: str
    recommendations: Optional[List[Recommendation]]  # null when gathering context
    end_of_conversation: bool