import json
from pathlib import Path

from src.diarization.diarizer import SpeakerDiarizer
from src.diarization.aligner import SpeakerAligner
from src.diarization.save_speaker_transcript import save_speaker_transcript


def process_diarization(meeting_dir):
    meeting_dir = Path(meeting_dir)

    transcript_path = meeting_dir / "transcript" / "transcript.json"

    with open(transcript_path, "r", encoding="utf-8") as file:
        transcript = json.load(file)

    audio_path = transcript["audio"]["path"]

    diarizer = SpeakerDiarizer()
    speaker_segments = diarizer.diarize(audio_path)

    aligner = SpeakerAligner()
    aligned_segments = aligner.align(
        transcript["segments"],
        speaker_segments,
    )

    output_path = save_speaker_transcript(
        meeting_dir,
        aligned_segments,
    )

    return output_path, speaker_segments, aligned_segments


if __name__ == "__main__":
    output_path, speaker_segments, aligned_segments = process_diarization(
        "data/meetings/valid_input/M-001"
    )

    print("Speaker segments:", len(speaker_segments))
    print("Aligned segments:", len(aligned_segments))
    print("Saved:", output_path)