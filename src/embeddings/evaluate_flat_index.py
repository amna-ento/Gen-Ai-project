import json
import time
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer


BASE_DIR = Path("data/meetings/valid_input/M-001")
EMBEDDINGS_PATH = BASE_DIR / "embeddings/bge_small/embeddings.npy"
CHUNKS_PATH = BASE_DIR / "chunks/chunks.json"
QUERIES_PATH = BASE_DIR / "embeddings/embedding_queries.json"

MODEL_NAME = "BAAI/bge-small-en-v1.5"
TOP_K = 5


def load_data():
    embeddings = np.load(EMBEDDINGS_PATH)

    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        chunks = json.load(f)["chunks"]

    with open(QUERIES_PATH, "r", encoding="utf-8") as f:
        queries = json.load(f)

    return embeddings, chunks, queries


def build_index(embeddings):
    start_time = time.perf_counter()

    index = np.asarray(embeddings, dtype=np.float32)

    build_time = time.perf_counter() - start_time

    return index, build_time


def search(index, query_embedding):
    scores = index @ query_embedding
    ranked_indices = np.argsort(scores)[::-1][:TOP_K]

    return ranked_indices, scores[ranked_indices]


def calculate_metrics(results):
    recalls = []
    hits = []
    reciprocal_ranks = []

    for result in results:
        expected = set(result["expected_chunk_ids"])
        retrieved = result["retrieved_chunk_ids"]

        relevant = expected.intersection(retrieved)

        recall = len(relevant) / len(expected)
        hit = 1.0 if relevant else 0.0

        reciprocal_rank = 0.0

        for rank, chunk_id in enumerate(retrieved, start=1):
            if chunk_id in expected:
                reciprocal_rank = 1.0 / rank
                break

        recalls.append(recall)
        hits.append(hit)
        reciprocal_ranks.append(reciprocal_rank)

    return {
        "Recall@5": round(float(np.mean(recalls)), 4),
        "Hit@5": round(float(np.mean(hits)), 4),
        "MRR@5": round(float(np.mean(reciprocal_ranks)), 4),
    }


def main():
    embeddings, chunks, queries = load_data()

    chunk_ids = [chunk["chunk_id"] for chunk in chunks]

    index, build_time = build_index(embeddings)

    model = SentenceTransformer(MODEL_NAME)

    query_texts = [query["query"] for query in queries]

    query_embeddings = model.encode(
        query_texts,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    query_embeddings = np.asarray(
        query_embeddings,
        dtype=np.float32,
    )

    results = []
    latencies = []

    for query, query_embedding in zip(queries, query_embeddings):
        start_time = time.perf_counter()

        ranked_indices, scores = search(
            index,
            query_embedding,
        )

        latency = time.perf_counter() - start_time
        latencies.append(latency)

        retrieved_chunk_ids = [
            chunk_ids[index]
            for index in ranked_indices
        ]

        results.append({
            "query_id": query["query_id"],
            "query": query["query"],
            "expected_chunk_ids": query["expected_chunk_ids"],
            "retrieved_chunk_ids": retrieved_chunk_ids,
            "scores": [
                round(float(score), 6)
                for score in scores
            ],
        })

    metrics = calculate_metrics(results)

    index_size_bytes = index.nbytes

    output = {
        "meeting_id": "M-001",
        "index_type": "Flat",
        "embedding_model": MODEL_NAME,
        "dimension": int(index.shape[1]),
        "vector_count": int(index.shape[0]),
        "top_k": TOP_K,
        "metrics": {
            **metrics,
            "average_latency_ms": round(
                float(np.mean(latencies) * 1000),
                4,
            ),
            "build_time_seconds": round(
                float(build_time),
                6,
            ),
            "index_size_kb": round(
                index_size_bytes / 1024,
                2,
            ),
        },
        "queries": results,
    }

    output_path = BASE_DIR / "embeddings/flat_index_evaluation.json"

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print("\n" + "=" * 70)
    print("Flat Vector Index Evaluation")
    print("=" * 70)

    print(json.dumps(output["metrics"], indent=2))

    print("\nResults:")
    print(output_path)

    print("=" * 70)


if __name__ == "__main__":
    main()