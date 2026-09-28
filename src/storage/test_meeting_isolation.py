import json
from pathlib import Path

from src.storage.meeting_paths import (
    BASE_DIR,
    get_meeting_dir,
    validate_meeting_id,
)


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def get_meeting_ids():
    return sorted(
        path.name
        for path in BASE_DIR.iterdir()
        if path.is_dir() and path.name.startswith("M-")
    )


def test_meeting_directories():
    meeting_ids = get_meeting_ids()

    if not meeting_ids:
        raise AssertionError("No meeting directories found.")

    for meeting_id in meeting_ids:
        validate_meeting_id(meeting_id)
        meeting_dir = get_meeting_dir(meeting_id)

        if not meeting_dir.exists():
            raise AssertionError(
                f"Meeting directory missing: {meeting_id}"
            )

    print(
        f"PASS: {len(meeting_ids)} meeting directories found."
    )


def test_required_storage():
    for meeting_id in get_meeting_ids():
        meeting_dir = get_meeting_dir(meeting_id)

        required_dirs = [
            meeting_dir / "audio",
            meeting_dir / "transcript",
        ]

        for directory in required_dirs:
            if not directory.exists():
                raise AssertionError(
                    f"Required directory missing: {directory}"
                )

    print("PASS: Required storage directories exist.")


def test_audio_files():
    checked = 0

    for meeting_id in get_meeting_ids():
        audio_dir = get_meeting_dir(meeting_id) / "audio"

        audio_files = [
            path
            for path in audio_dir.iterdir()
            if path.is_file()
        ]

        if not audio_files:
            raise AssertionError(
                f"No audio file found for {meeting_id}"
            )

        checked += 1

    print(
        f"PASS: Audio files verified for {checked} meetings."
    )


def test_chunk_metadata_isolation():
    checked = 0

    for meeting_id in get_meeting_ids():
        chunks_path = (
            get_meeting_dir(meeting_id)
            / "chunks"
            / "chunks.json"
        )

        if not chunks_path.exists():
            continue

        data = load_json(chunks_path)
        chunks = data.get("chunks", [])

        for chunk in chunks:
            chunk_meeting_id = chunk.get("meeting_id")

            if chunk_meeting_id != meeting_id:
                raise AssertionError(
                    "Chunk meeting isolation violation: "
                    f"expected {meeting_id}, "
                    f"found {chunk_meeting_id}"
                )

            chunk_id = chunk.get("chunk_id", "")

            if not chunk_id.startswith(
                f"{meeting_id}_CHUNK_"
            ):
                raise AssertionError(
                    "Chunk ID isolation violation: "
                    f"{chunk_id}"
                )

            checked += 1

    print(
        f"PASS: {checked} chunks verified for meeting isolation."
    )


def test_conversation_isolation():
    checked = 0

    for meeting_id in get_meeting_ids():
        conversation_path = (
            get_meeting_dir(meeting_id)
            / "conversation"
            / "conversation.json"
        )

        if not conversation_path.exists():
            continue

        conversation = load_json(
            conversation_path
        )

        stored_meeting_id = conversation.get(
            "meeting_id"
        )

        if stored_meeting_id != meeting_id:
            raise AssertionError(
                "Conversation isolation violation: "
                f"expected {meeting_id}, "
                f"found {stored_meeting_id}"
            )

        checked += 1

    print(
        f"PASS: {checked} conversations verified."
    )


def test_no_duplicate_meeting_ids():
    meeting_ids = get_meeting_ids()

    if len(meeting_ids) != len(set(meeting_ids)):
        raise AssertionError(
            "Duplicate meeting IDs detected."
        )

    print(
        f"PASS: No duplicate meeting IDs. "
        f"Total: {len(meeting_ids)}"
    )


def main():
    print("\n" + "=" * 100)
    print("PHASE 11 - ISOLATION + DATA INTEGRITY TESTS")
    print("=" * 100)

    test_meeting_directories()
    test_required_storage()
    test_audio_files()
    test_chunk_metadata_isolation()
    test_conversation_isolation()
    test_no_duplicate_meeting_ids()

    print("\n" + "=" * 100)
    print("ALL ISOLATION AND DATA INTEGRITY TESTS PASSED")
    print("=" * 100)


if __name__ == "__main__":
    main()
