
import json

from src.analysis.meeting_analyzer import MeetingAnalyzer


MEETING_ID = "M-001"

INPUT_PATH = (
    f"data/meetings/valid_input/{MEETING_ID}/"
    "transcript/speaker_transcript.json"
)


def main():
    with open(INPUT_PATH, "r", encoding="utf-8") as file:
        transcript = json.load(file)

    analyzer = MeetingAnalyzer()

    result = analyzer.analyze(
        transcript=transcript,
        meeting_id=MEETING_ID,
    )

    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
