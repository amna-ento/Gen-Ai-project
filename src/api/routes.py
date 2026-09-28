import json
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile
from starlette.concurrency import run_in_threadpool

from .pipeline import run_meeting_pipeline
from .schemas import (
    ChatRequest,
    ChatResponse,
    MeetingResponse,
    TranscriptResponse,
)

from src.storage.meeting_paths import (
    validate_meeting_id,
    get_meeting_dir,
)

from src.conversation.conversation_manager import ConversationManager
from src.conversation.multi_turn_chat import process_turn
from src.reranking.colbert_reranker import load_colbert_model


router = APIRouter(
    prefix="/meetings",
    tags=["Meetings"],
)


USER_INPUT_DIR = Path("data/meetings/user_input")

ALLOWED_EXTENSIONS = {
    ".wav",
    ".mp3",
    ".m4a",
    ".flac",
    ".ogg",
    ".aac",
}


_colbert_model = None


def get_colbert_model():
    global _colbert_model

    if _colbert_model is None:
        print("Loading ColBERT model...")
        _colbert_model = load_colbert_model()
        print("ColBERT model loaded.")

    return _colbert_model


@router.post("/upload", response_model=MeetingResponse)
async def upload_meeting(
    file: UploadFile = File(...)
):
    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="No audio file provided.",
        )

    filename = Path(file.filename).name
    extension = Path(filename).suffix.lower()

    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported audio format: {extension}. "
                f"Supported formats: {sorted(ALLOWED_EXTENSIONS)}"
            ),
        )

    USER_INPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    input_path = USER_INPUT_DIR / filename

    if input_path.exists():
        raise HTTPException(
            status_code=409,
            detail=f"File already exists: {filename}",
        )

    try:
        file_data = await file.read()

        if not file_data:
            raise HTTPException(
                status_code=400,
                detail="Uploaded audio file is empty.",
            )

        input_path.write_bytes(file_data)

        result = await run_in_threadpool(
            run_meeting_pipeline,
            str(input_path),
        )

        meeting_id = result["meeting_id"]

        meeting_dir = get_meeting_dir(
            meeting_id
        )

        transcript_path = (
            meeting_dir
            / "transcript"
            / "transcript.json"
        )

        with transcript_path.open(
            "r",
            encoding="utf-8",
        ) as file_handle:
            transcript = json.load(file_handle)

        duration = transcript.get(
            "audio",
            {},
        ).get(
            "duration"
        )

        segment_count = len(
            transcript.get(
                "segments",
                [],
            )
        )

        return MeetingResponse(
            meeting_id=meeting_id,
            status=result["status"],
            duration=duration,
            transcript_segments=segment_count,
        )

    except HTTPException:
        raise

    except Exception as exc:
        if input_path.exists():
            input_path.unlink()

        raise HTTPException(
            status_code=500,
            detail=f"Meeting processing failed: {exc}",
        ) from exc


@router.post(
    "/{meeting_id}/chat",
    response_model=ChatResponse,
)
async def chat(
    meeting_id: str,
    request: ChatRequest,
):
    try:
        meeting_id = validate_meeting_id(
            meeting_id
        )

        meeting_dir = get_meeting_dir(
            meeting_id
        )

        if not meeting_dir.exists():
            raise HTTPException(
                status_code=404,
                detail=f"Meeting not found: {meeting_id}",
            )

        question = request.question.strip()

        if not question:
            raise HTTPException(
                status_code=400,
                detail="Question cannot be empty.",
            )

        manager = ConversationManager(
            meeting_id=meeting_id
        )

        colbert_model = await run_in_threadpool(
            get_colbert_model
        )

        result = await run_in_threadpool(
            process_turn,
            manager,
            question,
            colbert_model,
        )

        return ChatResponse(
            meeting_id=meeting_id,
            conversation_id=manager.conversation_id,
            question=question,
            answer=result["answer"],
            citations=result.get(
                "citations",
                [],
            ),
        )

    except FileNotFoundError:
        raise HTTPException(
            status_code=404,
            detail=f"Meeting not found: {meeting_id}",
        )

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    except HTTPException:
        raise

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Chat processing failed: {exc}",
        ) from exc


@router.get(
    "/{meeting_id}",
    response_model=MeetingResponse,
)
async def get_meeting(
    meeting_id: str,
):
    try:
        meeting_id = validate_meeting_id(
            meeting_id
        )

        meeting_dir = get_meeting_dir(
            meeting_id
        )

        transcript_path = (
            meeting_dir
            / "transcript"
            / "transcript.json"
        )

        if not transcript_path.exists():
            raise HTTPException(
                status_code=404,
                detail=(
                    f"Transcript not found "
                    f"for meeting: {meeting_id}"
                ),
            )

        with transcript_path.open(
            "r",
            encoding="utf-8",
        ) as file_handle:
            transcript = json.load(file_handle)

        duration = transcript.get(
            "audio",
            {},
        ).get(
            "duration"
        )

        segments = transcript.get(
            "segments",
            [],
        )

        chunks_path = (
            meeting_dir
            / "chunks"
            / "chunks.json"
        )

        chroma_dir = (
            meeting_dir
            / "embeddings"
            / "chroma"
        )

        ready = (
            chunks_path.exists()
            and chroma_dir.exists()
        )

        return MeetingResponse(
            meeting_id=meeting_id,
            status="ready" if ready else "processing",
            duration=duration,
            transcript_segments=len(segments),
        )

    except FileNotFoundError:
        raise HTTPException(
            status_code=404,
            detail=f"Meeting not found: {meeting_id}",
        )

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc


@router.get(
    "/{meeting_id}/transcript",
    response_model=TranscriptResponse,
)
async def get_transcript(
    meeting_id: str,
):
    try:
        meeting_id = validate_meeting_id(
            meeting_id
        )

        meeting_dir = get_meeting_dir(
            meeting_id
        )

        speaker_transcript_path = (
            meeting_dir
            / "transcript"
            / "speaker_transcript.json"
        )

        if not speaker_transcript_path.exists():
            raise HTTPException(
                status_code=404,
                detail=(
                    f"Speaker transcript not found "
                    f"for meeting: {meeting_id}"
                ),
            )

        with speaker_transcript_path.open(
            "r",
            encoding="utf-8",
        ) as file_handle:
            transcript = json.load(file_handle)

        return TranscriptResponse(
            meeting_id=meeting_id,
            segments=transcript.get(
                "segments",
                [],
            ),
        )

    except FileNotFoundError:
        raise HTTPException(
            status_code=404,
            detail=f"Meeting not found: {meeting_id}",
        )

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc