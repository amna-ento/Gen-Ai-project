import json
from pathlib import Path


def save_speaker_transcript(meeting_dir, aligned_segments):
    meeting_dir = Path(meeting_dir)
    transcript_dir = meeting_dir / "transcript"
    transcript_dir.mkdir(parents=True, exist_ok=True)

    output_path = transcript_dir / "speaker_transcript.json"

    data = {
        "meeting_id": meeting_dir.name,
        "segments": aligned_segments,
    }

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, ensure_ascii=False)

    return output_path