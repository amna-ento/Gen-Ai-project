
import json
from pathlib import Path

from src.transcription.pipeline import process_new_meeting
from src.diarization.process_diarization import process_diarization
from src.analysis.meeting_analyzer import MeetingAnalyzer
from src.cleaning.process_cleaning import process_cleaning
from src.chunking.process_chunking import process_chunking
from src.embeddings.embed_chunks import build_embeddings
from src.embeddings.build_chroma_store import (
    load_data,
    validate_data,
    build_chroma_store,
)


def run_meeting_pipeline(audio_path: str) -> dict:
    print("\n" + "=" * 80)
    print("MEETING PIPELINE STARTED")
    print("=" * 80)

    # Phase 1
    print("\n[PHASE 1] Audio ingestion + transcription")

    phase1_result = process_new_meeting(
        audio_path
    )

    if phase1_result["segment_count"] == 0:
        raise ValueError(
            "No speech detected in the uploaded audio."
        )

    meeting_id = phase1_result[
        "meeting_id"
    ]

    meeting_dir = (
        Path("data/meetings/valid_input")
        / meeting_id
    )

    print(
        f"Meeting created: {meeting_id}"
    )

    # Phase 2
    print("\n[PHASE 2] Diarization + alignment")

    process_diarization(
        meeting_dir
    )

    # Phase 3
    print("\n[PHASE 3] Meeting analysis")

    transcript_path = (
        meeting_dir
        / "transcript"
        / "speaker_transcript.json"
    )

    with transcript_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        transcript = json.load(file)

    analyzer = MeetingAnalyzer()

    analyzer.analyze(
        transcript,
        meeting_id,
    )

    # Phase 4
    print("\n[PHASE 4] Transcript cleaning")

    process_cleaning(
        meeting_dir
    )

    # Phase 5
    print("\n[PHASE 5] Chunking")

    process_chunking(
        meeting_id
    )

    # Phase 6A
    print("\n[PHASE 6] BGE embedding generation")

    embeddings_path = build_embeddings(
        meeting_dir
    )

    # Phase 6B
    print("\n[PHASE 6] Chroma indexing")

    embeddings, chunks = load_data(
        meeting_id
    )

    validate_data(
        embeddings,
        chunks,
        meeting_id,
    )

    collection = build_chroma_store(
        meeting_id,
        embeddings,
        chunks,
    )

    print("\n" + "=" * 80)
    print("MEETING PIPELINE COMPLETED")
    print("=" * 80)

    return {
        "meeting_id": meeting_id,
        "audio_path": phase1_result[
            "audio_path"
        ],
        "transcript_path": phase1_result[
            "transcript_path"
        ],
        "embedding_path": str(
            embeddings_path
        ),
        "chroma_vectors": collection.count(),
        "status": "ready",
    }


if __name__ == "__main__":
    audio_path = input(
        "Enter audio path: "
    ).strip()

    result = run_meeting_pipeline(
        audio_path
    )

    print("\nPipeline result:")
    print(json.dumps(
        result,
        indent=2,
    ))
