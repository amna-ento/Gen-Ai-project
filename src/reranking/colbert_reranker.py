
import time
from sentence_transformers import MultiVectorEncoder, SentenceTransformer

from src.retrieval.hybrid_retrieval import (
    BM25_WEIGHT,
    CHUNKS_PATH,
    MEETING_ID,
    MODEL_NAME,
    SEMANTIC_WEIGHT,
    build_bm25_index,
    build_hybrid_results,
    bm25_search,
    load_chunks,
    load_chroma,
    semantic_search,
)


COLBERT_MODEL_NAME = "colbert-ir/colbertv2.0"

HYBRID_TOP_K = 10
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


def build_hybrid_retrieval(
    question,
):
    chunks = load_chunks()

    if not chunks:
        raise ValueError(
            f"No chunks found for {MEETING_ID}."
        )

    model = SentenceTransformer(
        MODEL_NAME
    )

    collection = load_chroma()

    bm25 = build_bm25_index(
        chunks
    )

    semantic_results = semantic_search(
        collection,
        model,
        question,
    )

    bm25_results = bm25_search(
        bm25,
        chunks,
        question,
    )

    hybrid_results = build_hybrid_results(
        semantic_results,
        bm25_results,
    )

    return hybrid_results


def rerank_with_colbert(
    model,
    question,
    hybrid_results,
):
    if not hybrid_results:
        return [], 0.0

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
        key=lambda result: result[
            "colbert_score"
        ],
        reverse=True,
    )

    reranked_results = reranked_results[
        :RERANK_TOP_K
    ]

    return (
        reranked_results,
        elapsed_ms,
    )


def display_results(
    question,
    hybrid_results,
    reranked_results,
    latency_ms,
):
    print("\n" + "=" * 100)
    print("COLBERT RERANKING")
    print("=" * 100)

    print(f"Meeting: {MEETING_ID}")
    print(f"Question: {question}")

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
    question = input(
        "\nEnter your question: "
    ).strip()

    if not question:
        raise ValueError(
            "Question cannot be empty."
        )

    print("\nBuilding Hybrid Top-10...")

    hybrid_results = build_hybrid_retrieval(
        question
    )

    if not hybrid_results:
        print(
            "No hybrid results found."
        )
        return

    print(
        "Loading ColBERT model..."
    )

    colbert_model = load_colbert_model()

    (
        reranked_results,
        latency_ms,
    ) = rerank_with_colbert(
        colbert_model,
        question,
        hybrid_results,
    )

    display_results(
        question,
        hybrid_results,
        reranked_results,
        latency_ms,
    )


if __name__ == "__main__":
    main()

