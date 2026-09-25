
import json
import time
from pathlib import Path

from rank_bm25 import BM25Okapi


BASE_DIR = Path("data/meetings/valid_input/M-001")

CHUNKS_PATH = BASE_DIR / "chunks/chunks.json"

MEETING_ID = "M-001"
TOP_K = 10


def load_chunks():
    with open(
        CHUNKS_PATH,
        "r",
        encoding="utf-8",
    ) as f:
        chunks = json.load(f)["chunks"]

    return [
        chunk
        for chunk in chunks
        if str(chunk["meeting_id"]) == MEETING_ID
    ]


def tokenize(text):
    return text.lower().split()


def build_bm25_index(chunks):
    tokenized_documents = [
        tokenize(chunk["text"])
        for chunk in chunks
    ]

    bm25 = BM25Okapi(
        tokenized_documents
    )

    return bm25


def search_bm25(
    bm25,
    chunks,
    question,
):
    query_tokens = tokenize(question)

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
        results.append(
            {
                "chunk_id": chunks[index]["chunk_id"],
                "score": float(scores[index]),
                "text": chunks[index]["text"],
                "metadata": chunks[index],
            }
        )

    return results, latency_ms


def display_results(
    question,
    results,
    latency_ms,
):
    print("\n" + "=" * 80)
    print("BM25 KEYWORD RETRIEVAL")
    print("=" * 80)

    print(f"Meeting:  {MEETING_ID}")
    print(f"Top-K:    {TOP_K}")
    print(f"Question: {question}")
    print(f"Latency:  {latency_ms:.4f} ms")

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
            f"BM25 Score: {result['score']:.4f}"
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
    question = input(
        "\nEnter your question: "
    ).strip()

    if not question:
        raise ValueError(
            "Question cannot be empty."
        )

    chunks = load_chunks()

    if not chunks:
        raise ValueError(
            f"No chunks found for meeting "
            f"{MEETING_ID}."
        )

    bm25 = build_bm25_index(
        chunks
    )

    results, latency_ms = search_bm25(
        bm25,
        chunks,
        question,
    )

    display_results(
        question,
        results,
        latency_ms,
    )


if __name__ == "__main__":
    main()
