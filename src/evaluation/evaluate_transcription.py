
import argparse
import json
from pathlib import Path

import jiwer


def load_reference(reference_path: str) -> str:
    path = Path(reference_path)

    if not path.exists():
        raise FileNotFoundError(f"Reference transcript does not exist: {path}")

    return path.read_text(encoding="utf-8").strip()


def load_hypothesis(hypothesis_path: str) -> str:
    path = Path(hypothesis_path)

    if not path.exists():
        raise FileNotFoundError(f"Hypothesis transcript does not exist: {path}")

    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    segments = data.get("segments", [])

    if not segments:
        raise ValueError("Transcript contains no segments.")

    return " ".join(
        segment["text"].strip()
        for segment in segments
        if segment.get("text", "").strip()
    )



def calculate_metrics(reference: str, hypothesis: str) -> dict:
    transformation = jiwer.Compose(
        [
            jiwer.ToLowerCase(),
            jiwer.RemovePunctuation(),
            jiwer.RemoveMultipleSpaces(),
            jiwer.Strip(),
        ]
    )

    reference_clean = str(transformation(reference))
    hypothesis_clean = str(transformation(hypothesis))

    wer_score = jiwer.wer(reference_clean, hypothesis_clean)
    cer_score = jiwer.cer(reference_clean, hypothesis_clean)

    reference_words = len(reference_clean.split())
    hypothesis_words = len(hypothesis_clean.split())

    return {
        "wer": round(wer_score, 4),
        "cer": round(cer_score, 4),
        "reference_words": reference_words,
        "hypothesis_words": hypothesis_words,
    }


def check_timestamps(hypothesis_path: str) -> dict:
    with open(hypothesis_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    segments = data.get("segments", [])

    valid_timestamps = 0

    for segment in segments:
        start = segment.get("start")
        end = segment.get("end")

        if (
            isinstance(start, (int, float))
            and isinstance(end, (int, float))
            and start >= 0
            and end > start
        ):
            valid_timestamps += 1

    return {
        "total_segments": len(segments),
        "valid_timestamps": valid_timestamps,
        "timestamp_accuracy": round(
            valid_timestamps / len(segments), 4
        ) if segments else 0.0,
    }


def evaluate_transcription(
    reference_path: str,
    hypothesis_path: str,
) -> dict:
    reference = load_reference(reference_path)
    hypothesis = load_hypothesis(hypothesis_path)

    metrics = calculate_metrics(reference, hypothesis)
    timestamp_metrics = check_timestamps(hypothesis_path)

    return {
        **metrics,
        **timestamp_metrics,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Whisper transcription using WER and CER."
    )

    parser.add_argument(
        "--reference",
        required=True,
        help="Path to the ground-truth reference transcript.",
    )

    parser.add_argument(
        "--hypothesis",
        required=True,
        help="Path to the Whisper transcript JSON.",
    )

    args = parser.parse_args()

    results = evaluate_transcription(
        args.reference,
        args.hypothesis,
    )

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
