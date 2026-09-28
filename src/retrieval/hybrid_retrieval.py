
import time

from src.storage.meeting_paths import validate_meeting_id
from src.retrieval.semantic_retrieval import (
    load_model,
    load_collection,
    semantic_search,
)
from src.retrieval.bm25_retrieval import (
    load_chunks,
    build_bm25_index,
    search_bm25,
)


SEMANTIC_WEIGHT = 0.6
BM25_WEIGHT = 0.4
TOP_K = 10


def min_max_normalize(scores):
    if not scores:
        return []

    minimum = min(scores)
    maximum = max(scores)

    if maximum == minimum:
        return [1.0] * len(scores)

    return [
        (score - minimum) / (maximum - minimum)
        for score in scores
    ]


def build_hybrid_retrieval(
    question: str,
    meeting_id: str,
):
    meeting_id = validate_meeting_id(
        meeting_id
    )

    total_start = time.perf_counter()

    chunks = load_chunks(
        meeting_id
    )

    bm25 = build_bm25_index(
        chunks
    )

    model = load_model()

    collection = load_collection(
        meeting_id
    )

    semantic_results, semantic_latency = (
        semantic_search(
            collection,
            model,
            question,
            meeting_id,
        )
    )

    documents = semantic_results["documents"][0]
    metadatas = semantic_results["metadatas"][0]
    distances = semantic_results["distances"][0]

    if len(documents) == 0:
        raise ValueError(
            f"No semantic results found "
            f"for meeting {meeting_id}."
        )

    if not (
        len(documents)
        == len(metadatas)
        == len(distances)
    ):
        raise ValueError(
            "Semantic result size mismatch."
        )

    semantic_similarities = [
        1.0 - float(distance)
        for distance in distances
    ]

    normalized_semantic = min_max_normalize(
        semantic_similarities
    )

    semantic_items = {}

    for index in range(
        len(documents)
    ):
        metadata = metadatas[index]

        result_meeting_id = metadata[
            "meeting_id"
        ]

        if result_meeting_id != meeting_id:
            raise ValueError(
                "Meeting isolation violation: "
                f"expected {meeting_id}, "
                f"found {result_meeting_id}."
            )

        chunk_id = metadata[
            "chunk_id"
        ]

        semantic_items[chunk_id] = {
            "chunk_id": chunk_id,
            "text": documents[index],
            "metadata": metadata,
            "semantic_score": normalized_semantic[
                index
            ],
            "bm25_score": 0.0,
        }

    bm25_results, bm25_latency = (
        search_bm25(
            bm25,
            chunks,
            question,
            meeting_id,
        )
    )

    bm25_scores = [
        float(result["score"])
        for result in bm25_results
    ]

    normalized_bm25 = min_max_normalize(
        bm25_scores
    )

    for index in range(
        len(bm25_results)
    ):
        result = bm25_results[index]

        metadata = result[
            "metadata"
        ]

        result_meeting_id = metadata[
            "meeting_id"
        ]

        if result_meeting_id != meeting_id:
            raise ValueError(
                "Meeting isolation violation: "
                f"expected {meeting_id}, "
                f"found {result_meeting_id}."
            )

        chunk_id = result[
            "chunk_id"
        ]

        if chunk_id not in semantic_items:
            semantic_items[chunk_id] = {
                "chunk_id": chunk_id,
                "text": result["text"],
                "metadata": metadata,
                "semantic_score": 0.0,
                "bm25_score": normalized_bm25[
                    index
                ],
            }
        else:
            semantic_items[chunk_id][
                "bm25_score"
            ] = normalized_bm25[index]

    for result in semantic_items.values():
        result["hybrid_score"] = (
            SEMANTIC_WEIGHT
            * result["semantic_score"]
            + BM25_WEIGHT
            * result["bm25_score"]
        )

        if result["metadata"][
            "meeting_id"
        ] != meeting_id:
            raise ValueError(
                "Meeting isolation violation "
                "during hybrid fusion."
            )

    ranked_results = sorted(
        semantic_items.values(),
        key=lambda item: item[
            "hybrid_score"
        ],
        reverse=True,
    )[:TOP_K]

    total_latency = (
        time.perf_counter()
        - total_start
    ) * 1000

    return {
        "meeting_id": meeting_id,
        "results": ranked_results,
        "semantic_latency_ms": semantic_latency,
        "bm25_latency_ms": bm25_latency,
        "total_latency_ms": total_latency,
    }


def display_results(
    question,
    retrieval_output,
):
    meeting_id = retrieval_output[
        "meeting_id"
    ]

    results = retrieval_output[
        "results"
    ]

    print("\n" + "=" * 80)
    print("HYBRID RETRIEVAL")
    print("=" * 80)

    print(
        f"Meeting: {meeting_id}"
    )

    print(
        f"Question: {question}"
    )

    print(
        f"Semantic Weight: "
        f"{SEMANTIC_WEIGHT}"
    )

    print(
        f"BM25 Weight: "
        f"{BM25_WEIGHT}"
    )

    print(
        f"Semantic Latency: "
        f"{retrieval_output['semantic_latency_ms']:.2f} ms"
    )

    print(
        f"BM25 Latency: "
        f"{retrieval_output['bm25_latency_ms']:.2f} ms"
    )

    print(
        f"Total Latency: "
        f"{retrieval_output['total_latency_ms']:.2f} ms"
    )

    print("=" * 80)

    for rank, result in enumerate(
        results,
        start=1,
    ):
        metadata = result[
            "metadata"
        ]

        print(f"\nRank: {rank}")

        print(
            f"Chunk ID: "
            f"{result['chunk_id']}"
        )

        print(
            f"Hybrid Score: "
            f"{result['hybrid_score']:.4f}"
        )

        print(
            f"Semantic Score: "
            f"{result['semantic_score']:.4f}"
        )

        print(
            f"BM25 Score: "
            f"{result['bm25_score']:.4f}"
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
            f"Text: "
            f"{result['text']}"
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

    retrieval_output = (
        build_hybrid_retrieval(
            question,
            meeting_id,
        )
    )

    display_results(
        question,
        retrieval_output,
    )


if __name__ == "__main__":
    main()
