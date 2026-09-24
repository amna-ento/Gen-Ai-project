import json
from pathlib import Path

from src.chunking.chunker import MeetingChunker


MEETING_ID = "M-001"

MEETING_DIR = Path(
    f"data/meetings/valid_input/{MEETING_ID}"
)

TRANSCRIPT_PATH = (
    MEETING_DIR
    / "transcript"
    / "cleaned_transcript.json"
)

ANALYSIS_PATH = (
    MEETING_DIR
    / "analysis"
    / "final_analysis.json"
)

OUTPUT_DIR = MEETING_DIR / "chunks"

OUTPUT_PATH = OUTPUT_DIR / "chunks.json"


def load_json(path):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def get_transcript(data):
    if isinstance(data, list):
        return data

    if isinstance(data, dict) and "segments" in data:
        return data["segments"]

    raise ValueError(
        "cleaned_transcript.json does not contain transcript segments."
    )


def process_chunking():
    transcript_data = load_json(TRANSCRIPT_PATH)
    analysis = load_json(ANALYSIS_PATH)

    transcript = get_transcript(transcript_data)

    chunker = MeetingChunker(
        target_words=160,
        max_words=220,
        min_words=60,
        max_duration=90,
    )

    chunks = chunker.create_chunks(
        meeting_id=MEETING_ID,
        transcript=transcript,
        analysis=analysis,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    output = {
        "meeting_id": MEETING_ID,
        "total_source_segments": len(transcript),
        "total_chunks": len(chunks),
        "chunking_config": {
            "target_words": 160,
            "max_words": 220,
            "min_words": 60,
            "max_duration_seconds": 90,
        },
        "chunks": chunks,
    }

    with open(
        OUTPUT_PATH,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            output,
            file,
            indent=2,
            ensure_ascii=False,
        )

    print("Chunking completed.")
    print(f"Input transcript: {TRANSCRIPT_PATH}")
    print(f"Input analysis:   {ANALYSIS_PATH}")
    print(f"Output:           {OUTPUT_PATH}")
    print(f"Source segments:  {len(transcript)}")
    print(f"Total chunks:     {len(chunks)}")


if __name__ == "__main__":
    process_chunking()