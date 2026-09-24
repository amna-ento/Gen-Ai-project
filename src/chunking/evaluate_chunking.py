import json
from pathlib import Path


MEETING_ID = "M-001"

MEETING_DIR = Path(
    f"data/meetings/valid_input/{MEETING_ID}"
)

TRANSCRIPT_PATH = (
    MEETING_DIR
    / "transcript"
    / "cleaned_transcript.json"
)

CHUNKS_PATH = (
    MEETING_DIR
    / "chunks"
    / "chunks.json"
)


def load_json(path):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def get_transcript_segments(data):
    if isinstance(data, list):
        return data

    if isinstance(data, dict) and "segments" in data:
        return data["segments"]

    raise ValueError(
        "Invalid cleaned transcript structure."
    )


def evaluate_coverage(source_segments, chunks):
    expected_ids = set(
        range(1, len(source_segments) + 1)
    )

    chunk_segment_ids = []

    for chunk in chunks:
        chunk_segment_ids.extend(
            chunk.get("source_segments", [])
        )

    actual_ids = set(chunk_segment_ids)

    missing_ids = sorted(
        expected_ids - actual_ids
    )

    duplicate_ids = sorted(
        segment_id
        for segment_id in actual_ids
        if chunk_segment_ids.count(segment_id) > 1
    )

    return {
        "expected_segments": len(expected_ids),
        "covered_segments": len(actual_ids),
        "missing_segments": len(missing_ids),
        "duplicate_segments": len(duplicate_ids),
        "missing_segment_ids": missing_ids,
        "duplicate_segment_ids": duplicate_ids,
    }


def evaluate_order(chunks):
    all_segment_ids = []

    for chunk in chunks:
        all_segment_ids.extend(
            chunk.get("source_segments", [])
        )

    expected_order = list(
        range(1, len(all_segment_ids) + 1)
    )

    return {
        "correct_order": all_segment_ids == expected_order,
    }


def evaluate_chunk_sizes(chunks, config):
    target_words = config["target_words"]
    max_words = config["max_words"]
    max_duration = config["max_duration_seconds"]

    word_counts = [
        chunk.get("word_count", 0)
        for chunk in chunks
    ]

    durations = [
        chunk.get("end_time", 0)
        - chunk.get("start_time", 0)
        for chunk in chunks
    ]

    oversized_chunks = [
        chunk["chunk_id"]
        for chunk in chunks
        if chunk.get("word_count", 0) > max_words
    ]

    long_chunks = [
        chunk["chunk_id"]
        for chunk in chunks
        if (
            chunk.get("end_time", 0)
            - chunk.get("start_time", 0)
        ) > max_duration
    ]

    return {
        "target_words": target_words,
        "max_words": max_words,
        "max_duration_seconds": max_duration,
        "average_words": round(
            sum(word_counts) / len(word_counts), 2
        ),
        "minimum_words": min(word_counts),
        "maximum_words": max(word_counts),
        "average_duration_seconds": round(
            sum(durations) / len(durations), 2
        ),
        "oversized_chunks": oversized_chunks,
        "long_chunks": long_chunks,
    }


def evaluate_chunking():
    transcript_data = load_json(TRANSCRIPT_PATH)
    chunks_data = load_json(CHUNKS_PATH)

    source_segments = get_transcript_segments(
        transcript_data
    )

    chunks = chunks_data["chunks"]

    config = chunks_data["chunking_config"]

    coverage = evaluate_coverage(
        source_segments,
        chunks,
    )

    order = evaluate_order(chunks)

    size = evaluate_chunk_sizes(
        chunks,
        config,
    )

    evaluation = {
        "meeting_id": MEETING_ID,
        "total_source_segments": len(source_segments),
        "total_chunks": len(chunks),
        "coverage": coverage,
        "order_check": order,
        "chunk_size": size,
    }

    output_path = (
        MEETING_DIR
        / "chunks"
        / "chunking_evaluation.json"
    )

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            evaluation,
            file,
            indent=2,
            ensure_ascii=False,
        )

    print("Chunking evaluation completed.")
    print()
    print(
        f"Expected segments: "
        f"{coverage['expected_segments']}"
    )
    print(
        f"Covered segments:  "
        f"{coverage['covered_segments']}"
    )
    print(
        f"Missing segments:  "
        f"{coverage['missing_segments']}"
    )
    print(
        f"Duplicate segments: "
        f"{coverage['duplicate_segments']}"
    )
    print(
        f"Correct order:     "
        f"{order['correct_order']}"
    )
    print()
    print(
        f"Average words/chunk: "
        f"{size['average_words']}"
    )
    print(
        f"Minimum words:       "
        f"{size['minimum_words']}"
    )
    print(
        f"Maximum words:       "
        f"{size['maximum_words']}"
    )
    print(
        f"Oversized chunks:    "
        f"{len(size['oversized_chunks'])}"
    )
    print(
        f"Chunks over 90 sec:  "
        f"{len(size['long_chunks'])}"
    )
    print()
    print(f"Output: {output_path}")


if __name__ == "__main__":
    evaluate_chunking()