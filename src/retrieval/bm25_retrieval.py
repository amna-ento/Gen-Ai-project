import json
import time

from rank_bm25 import BM25Okapi

from src.storage.meeting_paths import (
    get_chunks_path,
    validate_meeting_id,
)


TOP_K = 10


def load_chunks(meeting_id: str):
    meeting_id = validate_meeting_id(
        meeting_id
    )

    chunks_path = get_chunks_path(
        meeting_id
    )

    with open(
        chunks_path,
        "r",
        encoding="utf-8",
    ) as f:
        chunks = json.load(f)["chunks"]

    for chunk in chunks:
        chunk_meeting_id = str(
            chunk["meeting_id"]
        )

        if chunk_meeting_id != meeting_id:
            raise ValueError(
                "Meeting isolation violation: "
                f"expected {meeting_id}, "
                f"found {chunk_meeting_id} "
                f"in chunk {chunk.get('chunk_id')}."
            )

    return chunks


def tokenize(text):
    return text.lower().split()


def build_bm25_index(chunks):
    if not chunks:
        raise ValueError(
            "Cannot build BM25 index from empty chunks."
        )

    tokenized_documents = [
        tokenize(chunk["text"])
        for chunk in chunks
    ]

    return BM25Okapi(
        tokenized_documents
    )


def search_bm25(
    bm25,
    chunks,
    question,
    meeting_id: str,
):
    meeting_id = validate_meeting_id(
        meeting_id
    )

    query_tokens = tokenize(
        question
    )

    start = time.perf_counter()

    scores = bm25.get_scores(
        query_tokens
    )

    ranked_indices = sorted(
        range(len(scores)),
        key=lambda index: scores[index],
        reverse=True,
    )[:TOP_K]

    end = time.perf_counter()

    latency_ms = (
        end - start
    ) * 1000

    results = []

    for index in ranked_indices:
        chunk = chunks[index]

        if str(chunk["meeting_id"]) != meeting_id:
            raise ValueError(
                "Meeting isolation violation: "
                f"expected {meeting_id}, "
                f"found {chunk['meeting_id']}."
            )

        results.append(
            {
                "chunk_id": chunk["chunk_id"],
                "score": float(
                    scores[index]
                ),
                "text": chunk["text"],
                "metadata": chunk,
            }
        )

    return results, latency_ms


def display_results(
    question,
    results,
    latency_ms,
    meeting_id,
):
    print("\n" + "=" * 80)
    print("BM25 KEYWORD RETRIEVAL")
    print("=" * 80)

    print(f"Meeting:  {meeting_id}")
    print(f"Top-K:    {TOP_K}")
    print(f"Question: {question}")
    print(
        f"Latency:  {latency_ms:.4f} ms"
    )

    print("=" * 80)

    for rank, result in enumerate(
        results,
        start=1,
    ):
        metadata = result["metadata"]

        print(f"\nRank: {rank}")
        print(
            f"Chunk ID: {result['chunk_id']}"
        )
        print(
            f"BM25 Score: "
            f"{result['score']:.4f}"
        )
        print(
            f"Speaker: "
            f"{metadata.get('speaker', '')}"
        )
        print(
            f"Time: "
            f"{metadata.get('start_time', 0.0):.2f}"
            f" - "
            f"{metadata.get('end_time', 0.0):.2f}"
        )
        print(
            f"Text: {result['text']}"
        )

    print("\n" + "=" * 80)


def main():
    meeting_id = input(
        "\nEnter meeting ID: "
    ).strip()

    meeting_id = validate_meeting_id(
        meeting_id
    )

    question = input(
        "Enter your question: "
    ).strip()

    if not question:
        raise ValueError(
            "Question cannot be empty."
        )

    chunks = load_chunks(
        meeting_id
    )

    bm25 = build_bm25_index(
        chunks
    )

    results, latency_ms = search_bm25(
        bm25,
        chunks,
        question,
        meeting_id,
    )

    display_results(
        question,
        results,
        latency_ms,
        meeting_id,
    )


if __name__ == "__main__":
    main()