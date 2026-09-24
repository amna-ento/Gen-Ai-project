
import json
import sys
from pathlib import Path


MEETING_ID = "M-001"

ANALYSIS_PATH = Path(
    f"data/meetings/valid_input/{MEETING_ID}/analysis/final_analysis.json"
)

TRANSCRIPT_PATH = Path(
    f"data/meetings/valid_input/{MEETING_ID}/transcript/speaker_transcript.json"
)


REQUIRED_TOP_LEVEL_FIELDS = {
    "topics",
    "sentiment",
    "decisions",
    "action_items",
}

REQUIRED_TOPIC_FIELDS = {
    "topic",
    "source_segments",
}

REQUIRED_DECISION_FIELDS = {
    "decision",
    "made_by",
    "source_segments",
}

REQUIRED_ACTION_FIELDS = {
    "assigned_by",
    "assigned_to",
    "task",
    "deadline",
    "source_segments",
}

VALID_SENTIMENT_LABELS = {
    "positive",
    "negative",
    "neutral",
    "mixed",
}


def load_json(path):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def get_transcript_segments(transcript):
    if not isinstance(transcript, dict):
        raise ValueError(
            "speaker_transcript.json must contain a JSON object."
        )

    segments = transcript.get("segments")

    if not isinstance(segments, list):
        raise ValueError(
            "speaker_transcript.json does not contain a valid 'segments' list."
        )

    return segments


def get_valid_segment_ids(segments):
    return set(range(1, len(segments) + 1))


def get_valid_speakers(segments):
    speakers = set()

    for segment in segments:
        if not isinstance(segment, dict):
            continue

        speaker = segment.get("speaker")

        if speaker:
            speakers.add(speaker)

    return speakers


def validate_top_level_structure(data):
    errors = []

    if not isinstance(data, dict):
        return ["Analysis output must be a JSON object"]

    missing = REQUIRED_TOP_LEVEL_FIELDS - set(data.keys())

    if missing:
        errors.append(
            f"Missing top-level fields: {sorted(missing)}"
        )

    return errors


def validate_topics(topics, valid_segment_ids):
    errors = []

    if not isinstance(topics, list):
        return ["'topics' must be a list"]

    for index, topic in enumerate(topics):
        if not isinstance(topic, dict):
            errors.append(
                f"Topic {index}: must be an object"
            )
            continue

        missing = REQUIRED_TOPIC_FIELDS - set(topic.keys())

        if missing:
            errors.append(
                f"Topic {index}: missing fields {sorted(missing)}"
            )

        topic_text = topic.get("topic")

        if not isinstance(topic_text, str) or not topic_text.strip():
            errors.append(
                f"Topic {index}: topic text is empty"
            )

        source_segments = topic.get("source_segments", [])

        if not isinstance(source_segments, list):
            errors.append(
                f"Topic {index}: source_segments must be a list"
            )
            continue

        invalid_segments = [
            segment_id
            for segment_id in source_segments
            if segment_id not in valid_segment_ids
        ]

        if invalid_segments:
            errors.append(
                f"Topic {index}: invalid source segments "
                f"{invalid_segments}"
            )

    return errors


def validate_sentiment(sentiment, valid_speakers):
    errors = []

    if isinstance(sentiment, dict):
        sentiment_items = sentiment.items()

    elif isinstance(sentiment, list):
        sentiment_items = []

        for index, item in enumerate(sentiment):
            if not isinstance(item, dict):
                errors.append(
                    f"Sentiment {index}: must be an object"
                )
                continue

            speaker = item.get("speaker")

            if speaker is None:
                errors.append(
                    f"Sentiment {index}: missing speaker"
                )
                continue

            sentiment_items.append((speaker, item))

    else:
        return [
            "'sentiment' must be an object or list"
        ]

    for speaker, value in sentiment_items:

        if speaker not in valid_speakers:
            errors.append(
                f"Sentiment: unknown speaker '{speaker}'"
            )

        if not isinstance(value, dict):
            errors.append(
                f"Sentiment for {speaker}: must be an object"
            )
            continue

        label = value.get("label")

        if label is None:
            label = value.get("sentiment")

        if label not in VALID_SENTIMENT_LABELS:
            errors.append(
                f"Sentiment for {speaker}: invalid label '{label}'"
            )

    return errors


def validate_decisions(
    decisions,
    valid_segment_ids,
    valid_speakers,
):
    errors = []

    if not isinstance(decisions, list):
        return ["'decisions' must be a list"]

    for index, decision in enumerate(decisions):
        if not isinstance(decision, dict):
            errors.append(
                f"Decision {index}: must be an object"
            )
            continue

        missing = REQUIRED_DECISION_FIELDS - set(decision.keys())

        if missing:
            errors.append(
                f"Decision {index}: missing fields {sorted(missing)}"
            )

        decision_text = decision.get("decision")

        if not isinstance(decision_text, str) or not decision_text.strip():
            errors.append(
                f"Decision {index}: decision text is empty"
            )

        made_by = decision.get("made_by", [])

        if not isinstance(made_by, list):
            errors.append(
                f"Decision {index}: made_by must be a list"
            )
        else:
            for speaker in made_by:
                if speaker not in valid_speakers:
                    errors.append(
                        f"Decision {index}: unknown speaker '{speaker}'"
                    )

        source_segments = decision.get("source_segments", [])

        if not isinstance(source_segments, list):
            errors.append(
                f"Decision {index}: source_segments must be a list"
            )
            continue

        invalid_segments = [
            segment_id
            for segment_id in source_segments
            if segment_id not in valid_segment_ids
        ]

        if invalid_segments:
            errors.append(
                f"Decision {index}: invalid source segments "
                f"{invalid_segments}"
            )

    return errors


def validate_action_items(
    action_items,
    valid_segment_ids,
    valid_speakers,
):
    errors = []

    if not isinstance(action_items, list):
        return ["'action_items' must be a list"]

    for index, action in enumerate(action_items):
        if not isinstance(action, dict):
            errors.append(
                f"Action item {index}: must be an object"
            )
            continue

        missing = REQUIRED_ACTION_FIELDS - set(action.keys())

        if missing:
            errors.append(
                f"Action item {index}: missing fields {sorted(missing)}"
            )

        for field in ["assigned_by", "assigned_to"]:
            speaker = action.get(field)

            if speaker is not None and speaker not in valid_speakers:
                errors.append(
                    f"Action item {index}: unknown speaker "
                    f"'{speaker}' in {field}"
                )

        task = action.get("task")

        if not isinstance(task, str) or not task.strip():
            errors.append(
                f"Action item {index}: task is empty"
            )

        source_segments = action.get("source_segments", [])

        if not isinstance(source_segments, list):
            errors.append(
                f"Action item {index}: source_segments must be a list"
            )
            continue

        invalid_segments = [
            segment_id
            for segment_id in source_segments
            if segment_id not in valid_segment_ids
        ]

        if invalid_segments:
            errors.append(
                f"Action item {index}: invalid source segments "
                f"{invalid_segments}"
            )

    return errors


def find_exact_duplicates(items, key):
    seen = set()
    duplicates = []

    for index, item in enumerate(items):
        if not isinstance(item, dict):
            continue

        value = item.get(key)

        if not isinstance(value, str):
            continue

        normalized = " ".join(
            value.lower().split()
        )

        if normalized in seen:
            duplicates.append(index)
        else:
            seen.add(normalized)

    return duplicates


def calculate_completeness(data):
    topics = data.get("topics", [])
    decisions = data.get("decisions", [])
    action_items = data.get("action_items", [])

    checks = {
        "topics_have_text": all(
            isinstance(item, dict)
            and isinstance(item.get("topic"), str)
            and item.get("topic", "").strip()
            for item in topics
        ),
        "decisions_have_text": all(
            isinstance(item, dict)
            and isinstance(item.get("decision"), str)
            and item.get("decision", "").strip()
            for item in decisions
        ),
        "actions_have_text": all(
            isinstance(item, dict)
            and isinstance(item.get("task"), str)
            and item.get("task", "").strip()
            for item in action_items
        ),
    }

    passed = sum(checks.values())
    total = len(checks)

    score = passed / total if total else 0.0

    return checks, score


def get_sentiment_count(sentiment):
    if isinstance(sentiment, dict):
        return len(sentiment)

    if isinstance(sentiment, list):
        return len(sentiment)

    return 0


def main():
    if not ANALYSIS_PATH.exists():
        print(
            f"ERROR: Analysis file not found: "
            f"{ANALYSIS_PATH}"
        )
        sys.exit(1)

    if not TRANSCRIPT_PATH.exists():
        print(
            f"ERROR: Transcript file not found: "
            f"{TRANSCRIPT_PATH}"
        )
        sys.exit(1)

    analysis = load_json(ANALYSIS_PATH)
    transcript = load_json(TRANSCRIPT_PATH)

    try:
        transcript_segments = get_transcript_segments(
            transcript
        )
    except ValueError as error:
        print(f"ERROR: {error}")
        sys.exit(1)

    valid_segment_ids = get_valid_segment_ids(
        transcript_segments
    )

    valid_speakers = get_valid_speakers(
        transcript_segments
    )

    errors = []

    errors.extend(
        validate_top_level_structure(analysis)
    )

    errors.extend(
        validate_topics(
            analysis.get("topics", []),
            valid_segment_ids,
        )
    )

    errors.extend(
        validate_sentiment(
            analysis.get("sentiment", {}),
            valid_speakers,
        )
    )

    errors.extend(
        validate_decisions(
            analysis.get("decisions", []),
            valid_segment_ids,
            valid_speakers,
        )
    )

    errors.extend(
        validate_action_items(
            analysis.get("action_items", []),
            valid_segment_ids,
            valid_speakers,
        )
    )

    topic_duplicates = find_exact_duplicates(
        analysis.get("topics", []),
        "topic",
    )

    decision_duplicates = find_exact_duplicates(
        analysis.get("decisions", []),
        "decision",
    )

    action_duplicates = find_exact_duplicates(
        analysis.get("action_items", []),
        "task",
    )

    completeness_checks, completeness_score = (
        calculate_completeness(analysis)
    )

    print("\n" + "=" * 60)
    print("PHASE 3 STRUCTURAL EVALUATION")
    print("=" * 60)

    print(
        f"\nMeeting ID: "
        f"{transcript.get('meeting_id', MEETING_ID)}"
    )

    print(
        f"Transcript segments: "
        f"{len(transcript_segments)}"
    )

    print(
        f"Valid segment IDs: "
        f"1-{len(transcript_segments)}"
    )

    print(
        f"Valid speakers: "
        f"{len(valid_speakers)}"
    )

    print("\n--- Output Counts ---")

    print(
        f"Topics: "
        f"{len(analysis.get('topics', []))}"
    )

    print(
        f"Decisions: "
        f"{len(analysis.get('decisions', []))}"
    )

    print(
        f"Action items: "
        f"{len(analysis.get('action_items', []))}"
    )

    print(
        f"Sentiment speakers: "
        f"{get_sentiment_count(analysis.get('sentiment'))}"
    )

    print("\n--- Duplicate Check ---")

    print(
        f"Duplicate topics: "
        f"{len(topic_duplicates)}"
    )

    print(
        f"Duplicate decisions: "
        f"{len(decision_duplicates)}"
    )

    print(
        f"Duplicate action items: "
        f"{len(action_duplicates)}"
    )

    print("\n--- Completeness ---")

    for check, passed in completeness_checks.items():
        status = "PASS" if passed else "FAIL"
        print(f"{status}: {check}")

    print(
        f"Completeness score: "
        f"{completeness_score:.4f}"
    )

    print("\n--- Validation ---")

    if errors:
        print(
            f"FAILED: {len(errors)} validation error(s)"
        )

        for error in errors:
            print(f"- {error}")
    else:
        print(
            "PASSED: All structural validations"
        )

    print("\n" + "=" * 60)

    if errors:
        sys.exit(1)


if __name__ == "__main__":
    main()
