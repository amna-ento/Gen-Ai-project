import json
from pathlib import Path

from src.cleaning.transcript_cleaner import TranscriptCleaner


def process_cleaning(meeting_dir):
    meeting_dir = Path(meeting_dir)

    input_path = meeting_dir / "transcript" / "speaker_transcript.json"
    output_path = meeting_dir / "transcript" / "cleaned_transcript.json"

    if not input_path.exists():
        raise FileNotFoundError(
            f"Speaker transcript not found: {input_path}"
        )

    with open(input_path, "r", encoding="utf-8") as file:
        transcript = json.load(file)

    cleaner = TranscriptCleaner()

    cleaned_segments = []

    for segment in transcript["segments"]:
        cleaned_segment = {
            "speaker": segment["speaker"],
            "start": segment["start"],
            "end": segment["end"],
            "text": cleaner.clean_text(segment["text"]),
        }

        cleaned_segments.append(cleaned_segment)

    cleaned_transcript = {
        "meeting_id": transcript["meeting_id"],
        "segments": cleaned_segments,
    }

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(
            cleaned_transcript,
            file,
            indent=2,
            ensure_ascii=False
        )

    print("Cleaning completed.")
    print(f"Input:  {input_path}")
    print(f"Output: {output_path}")
    print(f"Segments processed: {len(cleaned_segments)}")


if __name__ == "__main__":
    process_cleaning("data/meetings/valid_input/M-001")