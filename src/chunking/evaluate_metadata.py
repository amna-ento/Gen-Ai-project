import json
from pathlib import Path
from collections import defaultdict


MEETING_ID = "M-001"

MEETING_DIR = Path(
    f"data/meetings/valid_input/{MEETING_ID}"
)

TRANSCRIPT_PATH = (
    MEETING_DIR
    / "transcript"
    / "cleaned_transcript.json"
)

ANALYSIS_PATH = (
    MEETING_DIR
    / "analysis"
    / "final_analysis.json"
)

CHUNKS_PATH = (
    MEETING_DIR
    / "chunks"
    / "chunks.json"
)

OUTPUT_PATH = (
    MEETING_DIR
    / "chunks"
    / "metadata_evaluation.json"
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


def build_transcript_map(segments):
    return {
        index: segment
        for index, segment
        in enumerate(segments, start=1)
    }


def build_topic_map(topics):
    topic_map = defaultdict(set)

    for topic in topics:
        topic_name = topic.get("topic")

        for segment_id in topic.get(
            "source_segments", []
        ):
            topic_map[segment_id].add(topic_name)

    return topic_map


def build_sentiment_map(sentiments):
    sentiment_map = defaultdict(set)

    for sentiment in sentiments:
        speaker = sentiment.get("speaker")
        label = sentiment.get("label")

        for segment_id in sentiment.get(
            "source_segments", []
        ):
            sentiment_map[segment_id].add(
                (speaker, label)
            )

    return sentiment_map


def expected_topics(source_ids, topic_map):
    topics = set()

    for segment_id in source_ids:
        topics.update(
            topic_map.get(segment_id, set())
        )

    return topics


def expected_sentiments(
    source_ids,
    sentiment_map,
):
    sentiments = set()

    for segment_id in source_ids:
        sentiments.update(
            sentiment_map.get(segment_id, set())
        )

    return sentiments


def evaluate_metadata():
    transcript_data = load_json(
        TRANSCRIPT_PATH
    )

    analysis = load_json(
        ANALYSIS_PATH
    )

    chunks_data = load_json(
        CHUNKS_PATH
    )

    transcript = get_transcript_segments(
        transcript_data
    )

    chunks = chunks_data["chunks"]

    transcript_map = build_transcript_map(
        transcript
    )

    topic_map = build_topic_map(
        analysis.get("topics", [])
    )

    sentiment_map = build_sentiment_map(
        analysis.get("sentiment", [])
    )

    meeting_id_errors = []
    speaker_errors = []
    timestamp_errors = []
    topic_errors = []
    sentiment_errors = []

    for chunk in chunks:
        chunk_id = chunk["chunk_id"]
        source_ids = chunk["source_segments"]

        # Meeting ID
        if chunk.get("meeting_id") != MEETING_ID:
            meeting_id_errors.append(chunk_id)

        # Expected source segments
        source_segments = [
            transcript_map[segment_id]
            for segment_id in source_ids
        ]

        expected_speakers = {
            segment["speaker"]
            for segment in source_segments
        }

        actual_speakers = set(
            chunk.get("speakers", [])
        )

        if actual_speakers != expected_speakers:
            speaker_errors.append({
                "chunk_id": chunk_id,
                "expected": sorted(
                    expected_speakers
                ),
                "actual": sorted(
                    actual_speakers
                ),
            })

        # Timestamp validation
        expected_start = source_segments[0]["start"]
        expected_end = source_segments[-1]["end"]

        if (
            chunk.get("start_time")
            != expected_start
            or chunk.get("end_time")
            != expected_end
        ):
            timestamp_errors.append({
                "chunk_id": chunk_id,
                "expected_start": expected_start,
                "actual_start": chunk.get(
                    "start_time"
                ),
                "expected_end": expected_end,
                "actual_end": chunk.get(
                    "end_time"
                ),
            })

        # Topic validation
        expected_topic_set = expected_topics(
            source_ids,
            topic_map,
        )

        actual_topic_set = set(
            chunk.get("topic", [])
        )

        if actual_topic_set != expected_topic_set:
            topic_errors.append({
                "chunk_id": chunk_id,
                "expected": sorted(
                    expected_topic_set
                ),
                "actual": sorted(
                    actual_topic_set
                ),
            })

        # Sentiment validation
        expected_sentiment_set = expected_sentiments(
            source_ids,
            sentiment_map,
        )

        actual_sentiment_set = {
            (
                item.get("speaker"),
                item.get("label"),
            )
            for item in chunk.get(
                "sentiment",
                [],
            )
        }

        if (
            actual_sentiment_set
            != expected_sentiment_set
        ):
            sentiment_errors.append({
                "chunk_id": chunk_id,
                "expected": sorted(
                    expected_sentiment_set
                ),
                "actual": sorted(
                    actual_sentiment_set
                ),
            })

    total_chunks = len(chunks)

    evaluation = {
        "meeting_id": MEETING_ID,
        "total_chunks": total_chunks,
        "metadata_quality": {
            "meeting_id": {
                "errors": len(meeting_id_errors),
                "passed": len(meeting_id_errors) == 0,
            },
            "speakers": {
                "errors": len(speaker_errors),
                "passed": len(speaker_errors) == 0,
            },
            "timestamps": {
                "errors": len(timestamp_errors),
                "passed": len(timestamp_errors) == 0,
            },
            "topics": {
                "errors": len(topic_errors),
                "passed": len(topic_errors) == 0,
            },
            "sentiment": {
                "errors": len(sentiment_errors),
                "passed": len(sentiment_errors) == 0,
            },
        },
        "errors": {
            "meeting_id": meeting_id_errors,
            "speakers": speaker_errors,
            "timestamps": timestamp_errors,
            "topics": topic_errors,
            "sentiment": sentiment_errors,
        },
    }

    with open(
        OUTPUT_PATH,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            evaluation,
            file,
            indent=2,
            ensure_ascii=False,
        )

    print("Metadata evaluation completed.")
    print()

    print(
        "Meeting ID errors: "
        f"{len(meeting_id_errors)}"
    )

    print(
        "Speaker errors: "
        f"{len(speaker_errors)}"
    )

    print(
        "Timestamp errors: "
        f"{len(timestamp_errors)}"
    )

    print(
        "Topic errors: "
        f"{len(topic_errors)}"
    )

    print(
        "Sentiment errors: "
        f"{len(sentiment_errors)}"
    )

    print()
    print(f"Output: {OUTPUT_PATH}")


if __name__ == "__main__":
    evaluate_metadata()