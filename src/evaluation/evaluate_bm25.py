import json
import time
from pathlib import Path

from rank_bm25 import BM25Okapi


BASE_DIR = Path("data/meetings/valid_input/M-001")

CHUNKS_PATH = BASE_DIR / "chunks/chunks.json"
EVALUATION_PATH = (
    BASE_DIR
    / "evaluation/retrieval_questions.json"
)

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


def load_evaluation_questions():
    with open(
        EVALUATION_PATH,
        "r",
        encoding="utf-8",
    ) as f:
        return json.load(f)["questions"]


def tokenize(text):
    return text.lower().split()


def build_bm25_index(chunks):
    tokenized_documents = [
        tokenize(chunk["text"])
        for chunk in chunks
    ]

    return BM25Okapi(
        tokenized_documents
    )


def retrieve(
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

    retrieved_ids = [
        chunks[index]["chunk_id"]
        for index in ranked_indices
    ]

    return retrieved_ids, latency_ms


def calculate_metrics(
    retrieved_ids,
    relevant_ids,
):
    relevant_set = set(relevant_ids)

    retrieved_relevant = [
        chunk_id
        for chunk_id in retrieved_ids
        if chunk_id in relevant_set
    ]

    recall = (
        len(retrieved_relevant)
        / len(relevant_set)
        if relevant_set
        else 0.0
    )

    hit = (
        1.0
        if retrieved_relevant
        else 0.0
    )

    precision = (
        len(retrieved_relevant)
        / len(retrieved_ids)
        if retrieved_ids
        else 0.0
    )

    reciprocal_rank = 0.0

    for rank, chunk_id in enumerate(
        retrieved_ids,
        start=1,
    ):
        if chunk_id in relevant_set:
            reciprocal_rank = 1.0 / rank
            break

    return (
        recall,
        hit,
        precision,
        reciprocal_rank,
    )


def main():
    chunks = load_chunks()
    questions = load_evaluation_questions()

    if not chunks:
        raise ValueError(
            f"No chunks found for {MEETING_ID}."
        )

    bm25 = build_bm25_index(
        chunks
    )

    total_recall = 0.0
    total_hit = 0.0
    total_precision = 0.0
    total_mrr = 0.0
    total_latency = 0.0

    print("\n" + "=" * 100)
    print("BM25 RETRIEVAL EVALUATION")
    print("=" * 100)

    print(f"Meeting: {MEETING_ID}")
    print(f"Top-K:   {TOP_K}")
    print(f"Questions: {len(questions)}")

    print("=" * 100)

    for item in questions:
        question_id = item["question_id"]
        question = item["question"]
        relevant_ids = item[
            "relevant_chunk_ids"
        ]

        retrieved_ids, latency_ms = retrieve(
            bm25,
            chunks,
            question,
        )

        (
            recall,
            hit,
            precision,
            reciprocal_rank,
        ) = calculate_metrics(
            retrieved_ids,
            relevant_ids,
        )

        total_recall += recall
        total_hit += hit
        total_precision += precision
        total_mrr += reciprocal_rank
        total_latency += latency_ms

        print(f"\n{question_id}: {question}")

        print(
            f"Recall@{TOP_K}: "
            f"{recall:.4f}"
        )

        print(
            f"Hit@{TOP_K}: "
            f"{hit:.4f}"
        )

        print(
            f"Precision@{TOP_K}: "
            f"{precision:.4f}"
        )

        print(
            f"MRR@{TOP_K}: "
            f"{reciprocal_rank:.4f}"
        )

        print(
            f"Latency: "
            f"{latency_ms:.4f} ms"
        )

        print(
            "Retrieved: "
            + ", ".join(retrieved_ids)
        )

        print(
            "Relevant:  "
            + ", ".join(relevant_ids)
        )

    count = len(questions)

    avg_recall = total_recall / count
    avg_hit = total_hit / count
    avg_precision = total_precision / count
    avg_mrr = total_mrr / count
    avg_latency = total_latency / count

    print("\n" + "=" * 100)
    print("FINAL BM25 RESULTS")
    print("=" * 100)

    print(
        f"Recall@{TOP_K}:     "
        f"{avg_recall:.4f}"
    )

    print(
        f"Hit@{TOP_K}:        "
        f"{avg_hit:.4f}"
    )

    print(
        f"Precision@{TOP_K}:  "
        f"{avg_precision:.4f}"
    )

    print(
        f"MRR@{TOP_K}:        "
        f"{avg_mrr:.4f}"
    )

    print(
        f"Avg Latency:        "
        f"{avg_latency:.4f} ms"
    )

    print("=" * 100)


if __name__ == "__main__":
    main()