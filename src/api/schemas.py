from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1)


class ChatResponse(BaseModel):
    meeting_id: str
    conversation_id: str
    question: str
    answer: str
    citations: list


class MeetingResponse(BaseModel):
    meeting_id: str
    status: str
    duration: float | None = None
    transcript_segments: int | None = None


class TranscriptResponse(BaseModel):
    meeting_id: str
    segments: list