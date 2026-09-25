import json
import time
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer


BASE_DIR = Path("data/meetings/valid_input/M-001")

CHROMA_DIR = BASE_DIR / "embeddings/chroma"
EVALUATION_PATH = BASE_DIR / "evaluation/retrieval_questions.json"

COLLECTION_NAME = "meeting_chunks"
MODEL_NAME = "BAAI/bge-small-en-v1.5"

MEETING_ID = "M-001"
TOP_K = 10


def load_evaluation_data():
    with open(
        EVALUATION_PATH,
        "r",
        encoding="utf-8",
    ) as f:
        return json.load(f)


def load_model():
    return SentenceTransformer(MODEL_NAME)


def load_collection():
    client = chromadb.PersistentClient(
        path=str(CHROMA_DIR)
    )

    return client.get_collection(
        name=COLLECTION_NAME
    )


def semantic_search(
    collection,
    model,
    question,
):
    query_embedding = model.encode(
        question,
        normalize_embeddings=True,
    )

    start = time.perf_counter()

    results = collection.query(
        query_embeddings=[
            query_embedding.tolist()
        ],
        n_results=TOP_K,
        where={
            "meeting_id": MEETING_ID
        },
        include=[
            "documents",
            "metadatas",
            "distances",
        ],
    )

    end = time.perf_counter()

    latency_ms = (
        end - start
    ) * 1000

    retrieved_ids = results["ids"][0]

    return retrieved_ids, latency_ms


def calculate_metrics(
    retrieved_ids,
    relevant_ids,
):
    retrieved_set = set(retrieved_ids)
    relevant_set = set(relevant_ids)

    relevant_retrieved = (
        retrieved_set & relevant_set
    )

    recall = (
        len(relevant_retrieved)
        / len(relevant_set)
        if relevant_set
        else 0.0
    )

    hit = (
        1.0
        if relevant_retrieved
        else 0.0
    )

    precision = (
        len(relevant_retrieved)
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
    evaluation_data = load_evaluation_data()

    model = load_model()
    collection = load_collection()

    results = []

    print("\n" + "=" * 90)
    print("SEMANTIC RETRIEVAL EVALUATION")
    print("=" * 90)

    for item in evaluation_data["questions"]:
        question_id = item["question_id"]
        question = item["question"]
        relevant_ids = item["relevant_chunk_ids"]

        retrieved_ids, latency_ms = semantic_search(
            collection,
            model,
            question,
        )

        (
            recall,
            hit,
            precision,
            mrr,
        ) = calculate_metrics(
            retrieved_ids,
            relevant_ids,
        )

        results.append(
            {
                "question_id": question_id,
                "question": question,
                "recall": recall,
                "hit": hit,
                "precision": precision,
                "mrr": mrr,
                "latency_ms": latency_ms,
            }
        )

        print(
            f"\n{question_id} | "
            f"Recall@{TOP_K}: {recall:.4f} | "
            f"Hit@{TOP_K}: {hit:.4f} | "
            f"Precision@{TOP_K}: {precision:.4f} | "
            f"MRR@{TOP_K}: {mrr:.4f} | "
            f"Latency: {latency_ms:.4f} ms"
        )

        print(
            f"Retrieved: {retrieved_ids}"
        )

        print(
            f"Relevant:  {relevant_ids}"
        )

    total = len(results)

    avg_recall = (
        sum(r["recall"] for r in results)
        / total
    )

    avg_hit = (
        sum(r["hit"] for r in results)
        / total
    )

    avg_precision = (
        sum(r["precision"] for r in results)
        / total
    )

    avg_mrr = (
        sum(r["mrr"] for r in results)
        / total
    )

    avg_latency = (
        sum(r["latency_ms"] for r in results)
        / total
    )

    print("\n" + "=" * 90)
    print("FINAL SEMANTIC RESULTS")
    print("=" * 90)

    print(
        f"Recall@{TOP_K}:     {avg_recall:.4f}"
    )
    print(
        f"Hit@{TOP_K}:        {avg_hit:.4f}"
    )
    print(
        f"Precision@{TOP_K}:  {avg_precision:.4f}"
    )
    print(
        f"MRR@{TOP_K}:        {avg_mrr:.4f}"
    )
    print(
        f"Avg Latency:        {avg_latency:.4f} ms"
    )

    print("=" * 90)


if __name__ == "__main__":
    main()
