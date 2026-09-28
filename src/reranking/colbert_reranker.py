import time

from sentence_transformers import MultiVectorEncoder

from src.retrieval.hybrid_retrieval import (
    build_hybrid_retrieval,
)

from src.storage.meeting_paths import (
    validate_meeting_id,
)

from src.generation.context_builder import (
    build_context,
    display_context,
)


COLBERT_MODEL_NAME = "colbert-ir/colbertv2.0"

RERANK_TOP_K = 5


def load_colbert_model():
    print(
        f"Loading ColBERT model: "
        f"{COLBERT_MODEL_NAME}"
    )

    model = MultiVectorEncoder(
        COLBERT_MODEL_NAME
    )

    print(
        "ColBERT model loaded successfully."
    )

    return model


def rerank_with_colbert(
    model,
    question,
    hybrid_results,
    meeting_id,
):
    meeting_id = validate_meeting_id(
        meeting_id
    )

    if not hybrid_results:
        return [], 0.0

    for result in hybrid_results:
        result_meeting_id = result[
            "metadata"
        ].get("meeting_id")

        if result_meeting_id != meeting_id:
            raise ValueError(
                "Meeting isolation violation "
                "before ColBERT reranking: "
                f"expected {meeting_id}, "
                f"found {result_meeting_id}."
            )

    documents = [
        result["text"]
        for result in hybrid_results
    ]

    start_time = time.perf_counter()

    query_embeddings = model.encode_query(
        [question]
    )

    document_embeddings = model.encode_document(
        documents
    )

    scores = model.similarity(
        query_embeddings,
        document_embeddings,
    )

    scores = (
        scores[0]
        .detach()
        .cpu()
        .numpy()
    )

    elapsed_ms = (
        time.perf_counter()
        - start_time
    ) * 1000

    reranked_results = []

    for result, score in zip(
        hybrid_results,
        scores,
    ):
        reranked_results.append(
            {
                **result,
                "colbert_score": float(score),
            }
        )

    reranked_results.sort(
        key=lambda result:
        result["colbert_score"],
        reverse=True,
    )

    reranked_results = (
        reranked_results[
            :RERANK_TOP_K
        ]
    )

    for result in reranked_results:
        result_meeting_id = result[
            "metadata"
        ].get("meeting_id")

        if result_meeting_id != meeting_id:
            raise ValueError(
                "Meeting isolation violation "
                "after ColBERT reranking: "
                f"expected {meeting_id}, "
                f"found {result_meeting_id}."
            )

    return (
        reranked_results,
        elapsed_ms,
    )


def display_results(
    meeting_id,
    question,
    hybrid_results,
    reranked_results,
    latency_ms,
):
    print("\n" + "=" * 100)
    print("COLBERT RERANKING")
    print("=" * 100)

    print(
        f"Meeting: {meeting_id}"
    )

    print(
        f"Question: {question}"
    )

    print("\nHybrid Top-10:")
    print("-" * 100)

    for rank, result in enumerate(
        hybrid_results,
        start=1,
    ):
        print(
            f"{rank}. "
            f"{result['chunk_id']} | "
            f"Hybrid Score: "
            f"{result['hybrid_score']:.4f}"
        )

    print("\nColBERT Top-5:")
    print("-" * 100)

    for rank, result in enumerate(
        reranked_results,
        start=1,
    ):
        print(
            f"{rank}. "
            f"{result['chunk_id']} | "
            f"ColBERT Score: "
            f"{result['colbert_score']:.4f}"
        )

    print(
        f"\nColBERT Reranking Latency: "
        f"{latency_ms:.4f} ms"
    )

    print("=" * 100)


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

    print(
        "\nBuilding Hybrid Top-10..."
    )

    retrieval_output = (
        build_hybrid_retrieval(
            question,
            meeting_id,
        )
    )

    hybrid_results = retrieval_output[
        "results"
    ]

    if not hybrid_results:
        print(
            "No hybrid results found."
        )
        return

    print(
        "Loading ColBERT model..."
    )

    colbert_model = (
        load_colbert_model()
    )

    (
        reranked_results,
        latency_ms,
    ) = rerank_with_colbert(
        colbert_model,
        question,
        hybrid_results,
        meeting_id,
    )

    if not reranked_results:
        print(
            "No reranked results found."
        )
        return

    print(
        "\nFIRST COLBERT RESULT:"
    )

    print(
        reranked_results[0]
    )

    context = build_context(
        reranked_results
    )

    display_context(
        context
    )

    display_results(
        meeting_id,
        question,
        hybrid_results,
        reranked_results,
        latency_ms,
    )


if __name__ == "__main__":
    main()