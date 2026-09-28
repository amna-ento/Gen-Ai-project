import json
from datetime import datetime
from pathlib import Path

from src.storage.meeting_paths import (
    BASE_DIR,
    get_meeting_dir,
    validate_meeting_id,
)


def get_meeting_ids():
    return sorted(
        path.name
        for path in BASE_DIR.iterdir()
        if path.is_dir() and path.name.startswith("M-")
    )


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def evaluate_meeting(meeting_id):
    meeting_dir = get_meeting_dir(meeting_id)

    result = {
        "meeting_id": meeting_id,
        "checks": {},
        "passed": True,
    }

    checks = result["checks"]

    checks["meeting_id_format"] = bool(
        validate_meeting_id(meeting_id)
    )

    checks["meeting_directory"] = meeting_dir.exists()

    audio_dir = meeting_dir / "audio"
    audio_files = (
        list(audio_dir.iterdir())
        if audio_dir.exists()
        else []
    )

    checks["audio_exists"] = any(
        path.is_file()
        for path in audio_files
    )

    transcript_dir = meeting_dir / "transcript"

    checks["transcript_exists"] = (
        transcript_dir.exists()
        and any(
            path.is_file()
            for path in transcript_dir.iterdir()
        )
    )

    chunks_path = (
        meeting_dir
        / "chunks"
        / "chunks.json"
    )

    if chunks_path.exists():
        data = load_json(chunks_path)
        chunks = data.get("chunks", [])

        chunk_metadata_valid = all(
            chunk.get("meeting_id") == meeting_id
            and chunk.get("chunk_id", "").startswith(
                f"{meeting_id}_CHUNK_"
            )
            for chunk in chunks
        )

        checks["chunk_metadata_isolation"] = (
            chunk_metadata_valid
        )

        result["chunk_count"] = len(chunks)

    else:
        checks["chunk_metadata_isolation"] = None
        result["chunk_count"] = 0

    conversation_path = (
        meeting_dir
        / "conversation"
        / "conversation.json"
    )

    if conversation_path.exists():
        conversation = load_json(
            conversation_path
        )

        checks["conversation_isolation"] = (
            conversation.get("meeting_id")
            == meeting_id
        )

    else:
        checks["conversation_isolation"] = None

    evaluated_checks = [
        value
        for value in checks.values()
        if value is not None
    ]

    result["passed"] = all(evaluated_checks)

    return result


def evaluate_all():
    meeting_ids = get_meeting_ids()

    if not meeting_ids:
        raise RuntimeError(
            "No meeting directories found."
        )

    meeting_results = []

    for meeting_id in meeting_ids:
        meeting_results.append(
            evaluate_meeting(meeting_id)
        )

    total_checks = 0
    passed_checks = 0

    for result in meeting_results:
        for value in result["checks"].values():
            if value is not None:
                total_checks += 1

                if value:
                    passed_checks += 1

    overall_passed = (
        all(
            result["passed"]
            for result in meeting_results
        )
        and total_checks > 0
    )

    return {
        "phase": 11,
        "evaluation": "Meeting Isolation + Data Integrity",
        "timestamp": datetime.now().isoformat(),
        "total_meetings": len(meeting_ids),
        "total_checks": total_checks,
        "passed_checks": passed_checks,
        "failed_checks": (
            total_checks - passed_checks
        ),
        "pass_rate": round(
            passed_checks / total_checks,
            4,
        ),
        "overall_passed": overall_passed,
        "meetings": meeting_results,
    }


def save_evaluation(result):
    evaluation_dir = BASE_DIR / "phase_11_evaluation"
    evaluation_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        evaluation_dir
        / "meeting_isolation_evaluation.json"
    )

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            result,
            file,
            indent=2,
        )

    return output_path


def display_result(result, output_path):
    print("\n" + "=" * 100)
    print("PHASE 11 - FINAL EVALUATION")
    print("=" * 100)

    print(
        f"Meetings Evaluated: "
        f"{result['total_meetings']}"
    )

    print(
        f"Checks Passed: "
        f"{result['passed_checks']}/"
        f"{result['total_checks']}"
    )

    print(
        f"Checks Failed: "
        f"{result['failed_checks']}"
    )

    print(
        f"Pass Rate: "
        f"{result['pass_rate'] * 100:.2f}%"
    )

    print(
        f"Overall Passed: "
        f"{result['overall_passed']}"
    )

    print("\nMeeting Results:")

    for meeting in result["meetings"]:
        print(
            f"\n{meeting['meeting_id']}"
        )

        print(
            f"  Chunks: "
            f"{meeting['chunk_count']}"
        )

        print(
            f"  Passed: "
            f"{meeting['passed']}"
        )

        for check, value in meeting["checks"].items():
            print(
                f"  {check}: {value}"
            )

    print(
        f"\nEvaluation saved to:\n"
        f"{output_path}"
    )

    print("=" * 100)


def main():
    result = evaluate_all()
    output_path = save_evaluation(result)
    display_result(result, output_path)


if __name__ == "__main__":
    main()
