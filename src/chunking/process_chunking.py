import json
from pathlib import Path

from src.chunking.chunker import MeetingChunker
from src.storage.meeting_paths import validate_meeting_id


BASE_DIR = Path("data/meetings/valid_input")


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


def process_chunking(meeting_id: str):
    meeting_id = validate_meeting_id(meeting_id)

    meeting_dir = BASE_DIR / meeting_id

    if not meeting_dir.exists():
        raise FileNotFoundError(
            f"Meeting directory does not exist: {meeting_dir}"
        )

    transcript_path = (
        meeting_dir
        / "transcript"
        / "cleaned_transcript.json"
    )

    analysis_path = (
        meeting_dir
        / "analysis"
        / "final_analysis.json"
    )

    output_dir = meeting_dir / "chunks"
    output_path = output_dir / "chunks.json"

    transcript_data = load_json(
        transcript_path
    )

    analysis = load_json(
        analysis_path
    )

    transcript = get_transcript(
        transcript_data
    )

    chunker = MeetingChunker(
        target_words=160,
        max_words=220,
        min_words=60,
        max_duration=90,
    )

    chunks = chunker.create_chunks(
        meeting_id=meeting_id,
        transcript=transcript,
        analysis=analysis,
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output = {
        "meeting_id": meeting_id,
        "total_source_segments": len(
            transcript
        ),
        "total_chunks": len(chunks),
        "chunking_config": {
            "target_words": 160,
            "max_words": 220,
            "min_words": 60,
            "max_duration_seconds": 90,
        },
        "chunks": chunks,
    }

    with output_path.open(
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
    print(f"Meeting ID:        {meeting_id}")
    print(f"Input transcript:  {transcript_path}")
    print(f"Input analysis:    {analysis_path}")
    print(f"Output:            {output_path}")
    print(f"Source segments:   {len(transcript)}")
    print(f"Total chunks:      {len(chunks)}")

    return output_path


if __name__ == "__main__":
    meeting_id = input(
        "Enter meeting ID: "
    ).strip()

    process_chunking(
        meeting_id
    )