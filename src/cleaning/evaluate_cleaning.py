import json
from pathlib import Path


MEETING_DIR = Path("data/meetings/valid_input/M-001")

ORIGINAL_PATH = MEETING_DIR / "transcript" / "speaker_transcript.json"
CLEANED_PATH = MEETING_DIR / "transcript" / "cleaned_transcript.json"


def load_json(path):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def evaluate_cleaning():
    original = load_json(ORIGINAL_PATH)
    cleaned = load_json(CLEANED_PATH)

    original_segments = original["segments"]
    cleaned_segments = cleaned["segments"]

    total = len(original_segments)
    changed = 0
    empty_after_cleaning = 0
    speaker_errors = 0
    timestamp_errors = 0

    examples = []

    for original_segment, cleaned_segment in zip(
        original_segments,
        cleaned_segments
    ):
        if original_segment["speaker"] != cleaned_segment["speaker"]:
            speaker_errors += 1

        if (
            original_segment["start"] != cleaned_segment["start"]
            or original_segment["end"] != cleaned_segment["end"]
        ):
            timestamp_errors += 1

        original_text = original_segment["text"]
        cleaned_text = cleaned_segment["text"]

        if original_text != cleaned_text:
            changed += 1

            if len(examples) < 10:
                examples.append(
                    {
                        "original": original_text,
                        "cleaned": cleaned_text,
                    }
                )

        if not cleaned_text.strip():
            empty_after_cleaning += 1

    print("\nCleaning Evaluation")
    print("-" * 40)
    print(f"Original segments:       {total}")
    print(f"Cleaned segments:        {len(cleaned_segments)}")
    print(f"Changed text segments:   {changed}")
    print(f"Empty after cleaning:    {empty_after_cleaning}")
    print(f"Speaker errors:          {speaker_errors}")
    print(f"Timestamp errors:        {timestamp_errors}")

    print("\nSample Changes")
    print("-" * 40)

    if examples:
        for index, example in enumerate(examples, start=1):
            print(f"\nExample {index}")
            print(f"Original: {example['original']}")
            print(f"Cleaned:  {example['cleaned']}")
    else:
        print("No text changes detected.")


if __name__ == "__main__":
    evaluate_cleaning()