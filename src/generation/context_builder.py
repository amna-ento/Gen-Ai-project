from typing import Any


def build_context(
    reranked_results: list[dict[str, Any]],
) -> str:

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

        meeting_id = metadata.get(
            "meeting_id",
            "UNKNOWN",
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