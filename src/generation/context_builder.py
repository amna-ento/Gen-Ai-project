from typing import Any

from src.storage.meeting_paths import validate_meeting_id


def build_context(
    reranked_results: list[dict[str, Any]],
    meeting_id: str,
) -> str:

    meeting_id = validate_meeting_id(meeting_id)

    if not reranked_results:
        return ""

    context_parts = []

    for rank, result in enumerate(
        reranked_results,
        start=1,
    ):
        chunk_id = result.get(
            "chunk_id",
            "UNKNOWN",
        )

        text = result.get(
            "text",
            "",
        )

        metadata = result.get(
            "metadata",
            {},
        )

        result_meeting_id = metadata.get(
            "meeting_id"
        )

        if result_meeting_id != meeting_id:
            raise ValueError(
                "Meeting isolation violation while "
                "building context: "
                f"expected {meeting_id}, "
                f"found {result_meeting_id}."
            )

        if not chunk_id.startswith(
            f"{meeting_id}_CHUNK_"
        ):
            raise ValueError(
                "Chunk isolation violation: "
                f"{chunk_id} does not belong to {meeting_id}."
            )

        speaker = metadata.get(
            "speaker",
            "",
        )

        start_time = metadata.get(
            "start_time",
            "UNKNOWN",
        )

        end_time = metadata.get(
            "end_time",
            "UNKNOWN",
        )

        context_parts.append(
            f"[Source {rank}]\n"
            f"Meeting ID: {meeting_id}\n"
            f"Chunk ID: {chunk_id}\n"
            f"Speaker: "
            f"{speaker or 'See speaker labels in text'}\n"
            f"Time: {start_time} - {end_time}\n"
            f"Text:\n{text}"
        )

    return "\n\n".join(
        context_parts
    )


def display_context(
    context: str,
):
    print("\n" + "=" * 100)
    print("GENERATED CONTEXT")
    print("=" * 100)

    print(context)

    print("=" * 100)