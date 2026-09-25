
import json
import time
from pathlib import Path

import numpy as np
from sentence_transformers import MultiVectorEncoder

from src.reranking.colbert_reranker import (
    build_hybrid_retrieval,
    RERANK_TOP_K,
)


BASE_DIR = Path("data/meetings/valid_input/M-001")

EVALUATION_PATH = (
    BASE_DIR
    / "evaluation/retrieval_questions.json"
)

MODEL_NAME = "colbert-ir/colbertv2.0"

TOP_K = 5


def load_evaluation_questions():
    with open(
        EVALUATION_PATH,
        "r",
        encoding="utf-8",
    ) as f:
        data = json.load(f)

    return data["questions"]


def load_colbert_model():
    print(
        f"Loading ColBERT model: {MODEL_NAME}"
    )

    model = MultiVectorEncoder(
        MODEL_NAME
    )

    print(
        "ColBERT model loaded successfully."
    )

    return model


def rerank_with_colbert(
    model,
    question,
    hybrid_results,
):
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

    latency_ms = (
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

    return (
        reranked_results[:TOP_K],
        latency_ms,
    )


def calculate_recall(
    results,
    relevant_chunk_ids,
):
    retrieved_ids = {
        result["chunk_id"]
        for result in results
    }

    relevant_ids = set(
        relevant_chunk_ids
    )

    if not relevant_ids:
        return 0.0

    return len(
        retrieved_ids & relevant_ids
    ) / len(relevant_ids)


def calculate_precision(
    results,
    relevant_chunk_ids,
):
    retrieved_ids = [
        result["chunk_id"]
        for result in results
    ]

    relevant_ids = set(
        relevant_chunk_ids
    )

    if not retrieved_ids:
        return 0.0

    relevant_retrieved = sum(
        chunk_id in relevant_ids
        for chunk_id in retrieved_ids
    )

    return (
        relevant_retrieved
        / len(retrieved_ids)
    )


def calculate_mrr(
    results,
    relevant_chunk_ids,
):
    relevant_ids = set(
        relevant_chunk_ids
    )

    for rank, result in enumerate(
        results,
        start=1,
    ):
        if result["chunk_id"] in relevant_ids:
            return 1.0 / rank

    return 0.0


def calculate_ndcg(
    results,
    relevant_chunk_ids,
):
    relevant_ids = set(
        relevant_chunk_ids
    )

    dcg = 0.0

    for rank, result in enumerate(
        results,
        start=1,
    ):
        if result["chunk_id"] in relevant_ids:
            dcg += 1.0 / np.log2(
                rank + 1
            )

    relevant_count = min(
        len(relevant_ids),
        len(results),
    )

    if relevant_count == 0:
        return 0.0

    ideal_dcg = sum(
        1.0 / np.log2(rank + 1)
        for rank in range(
            1,
            relevant_count + 1,
        )
    )

    return dcg / ideal_dcg


def evaluate():
    questions = load_evaluation_questions()

    print(
        f"Loaded {len(questions)} evaluation questions."
    )

    model = load_colbert_model()

    hybrid_recall = []
    colbert_recall = []

    hybrid_precision = []
    colbert_precision = []

    hybrid_mrr = []
    colbert_mrr = []

    hybrid_ndcg = []
    colbert_ndcg = []

    latencies = []

    print("\nStarting ColBERT evaluation...\n")

    for question_data in questions:
        question_id = question_data[
            "question_id"
        ]

        question = question_data[
            "question"
        ]

        relevant_chunk_ids = question_data[
            "relevant_chunk_ids"
        ]

        hybrid_results = build_hybrid_retrieval(
            question
        )

        hybrid_top_5 = hybrid_results[
            :TOP_K
        ]

        (
            colbert_results,
            latency_ms,
        ) = rerank_with_colbert(
            model,
            question,
            hybrid_results,
        )

        hybrid_recall.append(
            calculate_recall(
                hybrid_top_5,
                relevant_chunk_ids,
            )
        )

        colbert_recall.append(
            calculate_recall(
                colbert_results,
                relevant_chunk_ids,
            )
        )

        hybrid_precision.append(
            calculate_precision(
                hybrid_top_5,
                relevant_chunk_ids,
            )
        )

        colbert_precision.append(
            calculate_precision(
                colbert_results,
                relevant_chunk_ids,
            )
        )

        hybrid_mrr.append(
            calculate_mrr(
                hybrid_top_5,
                relevant_chunk_ids,
            )
        )

        colbert_mrr.append(
            calculate_mrr(
                colbert_results,
                relevant_chunk_ids,
            )
        )

        hybrid_ndcg.append(
            calculate_ndcg(
                hybrid_top_5,
                relevant_chunk_ids,
            )
        )

        colbert_ndcg.append(
            calculate_ndcg(
                colbert_results,
                relevant_chunk_ids,
            )
        )

        latencies.append(
            latency_ms
        )

        print(
            f"{question_id}: "
            f"Hybrid MRR={hybrid_mrr[-1]:.4f} | "
            f"ColBERT MRR={colbert_mrr[-1]:.4f} | "
            f"Latency={latency_ms:.2f} ms"
        )

    avg_hybrid_recall = np.mean(
        hybrid_recall
    )

    avg_colbert_recall = np.mean(
        colbert_recall
    )

    avg_hybrid_precision = np.mean(
        hybrid_precision
    )

    avg_colbert_precision = np.mean(
        colbert_precision
    )

    avg_hybrid_mrr = np.mean(
        hybrid_mrr
    )

    avg_colbert_mrr = np.mean(
        colbert_mrr
    )

    avg_hybrid_ndcg = np.mean(
        hybrid_ndcg
    )

    avg_colbert_ndcg = np.mean(
        colbert_ndcg
    )

    avg_latency = np.mean(
        latencies
    )

    print("\n" + "=" * 80)
    print("COLBERT EVALUATION RESULTS")
    print("=" * 80)

    print(
        f"\nRecall@5"
        f"\n  Hybrid:  {avg_hybrid_recall:.4f}"
        f"\n  ColBERT: {avg_colbert_recall:.4f}"
        f"\n  Change:  "
        f"{avg_colbert_recall - avg_hybrid_recall:+.4f}"
    )

    print(
        f"\nPrecision@5"
        f"\n  Hybrid:  {avg_hybrid_precision:.4f}"
        f"\n  ColBERT: {avg_colbert_precision:.4f}"
        f"\n  Change:  "
        f"{avg_colbert_precision - avg_hybrid_precision:+.4f}"
    )

    print(
        f"\nMRR@5"
        f"\n  Hybrid:  {avg_hybrid_mrr:.4f}"
        f"\n  ColBERT: {avg_colbert_mrr:.4f}"
        f"\n  Change:  "
        f"{avg_colbert_mrr - avg_hybrid_mrr:+.4f}"
    )

    print(
        f"\nNDCG@5"
        f"\n  Hybrid:  {avg_hybrid_ndcg:.4f}"
        f"\n  ColBERT: {avg_colbert_ndcg:.4f}"
        f"\n  Change:  "
        f"{avg_colbert_ndcg - avg_hybrid_ndcg:+.4f}"
    )

    print(
        f"\nAverage ColBERT Reranking Latency: "
        f"{avg_latency:.4f} ms"
    )

    print("\n" + "=" * 80)


if __name__ == "__main__":
    evaluate()


