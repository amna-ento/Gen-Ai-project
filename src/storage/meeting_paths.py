import re
from pathlib import Path


BASE_DIR = Path("data/meetings/valid_input")


def validate_meeting_id(meeting_id: str) -> str:
    if not re.fullmatch(r"M-\d{3}", meeting_id):
        raise ValueError(
            f"Invalid meeting_id: {meeting_id}. "
            "Expected format M-001."
        )

    return meeting_id


def get_meeting_dir(meeting_id: str) -> Path:
    meeting_id = validate_meeting_id(meeting_id)

    meeting_dir = BASE_DIR / meeting_id

    if not meeting_dir.exists():
        raise FileNotFoundError(
            f"Meeting directory does not exist: {meeting_dir}"
        )

    return meeting_dir


def get_chunks_path(meeting_id: str) -> Path:
    return (
        get_meeting_dir(meeting_id)
        / "chunks"
        / "chunks.json"
    )


def get_chroma_dir(meeting_id: str) -> Path:
    return (
        get_meeting_dir(meeting_id)
        / "embeddings"
        / "chroma"
    )


def get_evaluation_path(meeting_id: str) -> Path:
    return (
        get_meeting_dir(meeting_id)
        / "evaluation"
        / "retrieval_questions.json"
    )


def get_conversation_path(meeting_id: str) -> Path:
    return (
        get_meeting_dir(meeting_id)
        / "conversation"
        / "conversation.json"
    )
    
    