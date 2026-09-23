from pathlib import Path
import argparse
import json
import shutil

from src.transcription.audio_validator import validate_audio_file
from src.transcription.transcriber import MeetingTranscriber


MEETINGS_DIR = Path("data/meetings")
USER_INPUT_DIR = MEETINGS_DIR / "user_input"
VALID_INPUT_DIR = MEETINGS_DIR / "valid_input"


def generate_meeting_id() -> str:
    meeting_numbers = []

    if VALID_INPUT_DIR.exists():
        for path in VALID_INPUT_DIR.iterdir():
            if path.is_dir() and path.name.startswith("M-"):
                number = path.name[2:]

                if number.isdigit():
                    meeting_numbers.append(int(number))

    next_number = max(meeting_numbers, default=0) + 1

    return f"M-{next_number:03d}"


def process_new_meeting(audio_path: str) -> dict:
    source_audio = Path(audio_path)

    if not source_audio.exists():
        raise FileNotFoundError(
            f"Audio file does not exist: {source_audio}"
        )

    USER_INPUT_DIR.mkdir(parents=True, exist_ok=True)
    VALID_INPUT_DIR.mkdir(parents=True, exist_ok=True)

    if source_audio.parent.resolve() != USER_INPUT_DIR.resolve():
        raise ValueError(
            f"Audio file must be inside {USER_INPUT_DIR}: {source_audio}"
        )

    audio_info = validate_audio_file(str(source_audio))

    meeting_id = generate_meeting_id()

    meeting_dir = VALID_INPUT_DIR / meeting_id
    audio_dir = meeting_dir / "audio"
    transcript_dir = meeting_dir / "transcript"
    analysis_dir = meeting_dir / "analysis"
    conversation_dir = meeting_dir / "conversation"

    audio_dir.mkdir(parents=True, exist_ok=True)
    transcript_dir.mkdir(parents=True, exist_ok=True)
    analysis_dir.mkdir(parents=True, exist_ok=True)
    conversation_dir.mkdir(parents=True, exist_ok=True)

    saved_audio = audio_dir / source_audio.name

    shutil.move(
        str(source_audio),
        str(saved_audio),
    )

    transcriber = MeetingTranscriber()

    transcript = transcriber.transcribe(
        str(saved_audio)
    )

    transcript_data = {
        "meeting_id": meeting_id,
        "audio": {
            "filename": saved_audio.name,
            "path": str(saved_audio),
            "duration": audio_info["duration"],
            "sample_rate": audio_info["sample_rate"],
            "channels": audio_info["channels"],
        },
        "language": transcript["language"],
        "language_probability": transcript["language_probability"],
        "segments": transcript["segments"],
    }

    transcript_path = transcript_dir / "transcript.json"

    with transcript_path.open("w", encoding="utf-8") as file:
        json.dump(
            transcript_data,
            file,
            indent=2,
            ensure_ascii=False,
        )

    return {
        "meeting_id": meeting_id,
        "audio_path": str(saved_audio),
        "transcript_path": str(transcript_path),
        "segment_count": len(transcript["segments"]),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Process a user-uploaded meeting audio file."
    )

    parser.add_argument(
        "audio_path",
        help="Path to the uploaded audio file inside data/meetings/user_input/",
    )

    args = parser.parse_args()

    result = process_new_meeting(args.audio_path)

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()