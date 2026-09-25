import json
import time
from pathlib import Path

import hnswlib
import numpy as np
from sentence_transformers import SentenceTransformer


BASE_DIR = Path("data/meetings/valid_input/M-001")
EMBEDDINGS_PATH = BASE_DIR / "embeddings/bge_small/embeddings.npy"
CHUNKS_PATH = BASE_DIR / "chunks/chunks.json"
QUERIES_PATH = BASE_DIR / "embeddings/embedding_queries.json"

MODEL_NAME = "BAAI/bge-small-en-v1.5"

TOP_K = 5

M = 16
EF_CONSTRUCTION = 200
EF_SEARCH = 50
SEED = 42


def load_data():
    embeddings = np.load(EMBEDDINGS_PATH)

    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        chunks = json.load(f)["chunks"]

    with open(QUERIES_PATH, "r", encoding="utf-8") as f:
        queries = json.load(f)

    return embeddings, chunks, queries


def build_index(embeddings):
    dimension = embeddings.shape[1]
    vector_count = embeddings.shape[0]

    index = hnswlib.Index(
        space="cosine",
        dim=dimension,
    )

    start_time = time.perf_counter()

    index.init_index(
        max_elements=vector_count,
        M=M,
        ef_construction=EF_CONSTRUCTION,
        random_seed=SEED,
    )

    index.add_items(
        embeddings,
        np.arange(vector_count),
    )

    index.set_ef(EF_SEARCH)

    build_time = time.perf_counter() - start_time

    return index, build_time


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

        for rank, chunk_id in enumerate(
            retrieved,
            start=1,
        ):
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

    chunk_ids = [
        chunk["chunk_id"]
        for chunk in chunks
    ]

    index, build_time = build_index(embeddings)

    model = SentenceTransformer(MODEL_NAME)

    query_texts = [
        query["query"]
        for query in queries
    ]

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

    for query, query_embedding in zip(
        queries,
        query_embeddings,
    ):
        start_time = time.perf_counter()

        labels, distances = index.knn_query(
            query_embedding,
            k=TOP_K,
        )

        latency = time.perf_counter() - start_time
        latencies.append(latency)

        labels = labels[0]
        distances = distances[0]

        retrieved_chunk_ids = [
            chunk_ids[int(index_value)]
            for index_value in labels
        ]

        scores = [
            1.0 - float(distance)
            for distance in distances
        ]

        results.append({
            "query_id": query["query_id"],
            "query": query["query"],
            "expected_chunk_ids": query["expected_chunk_ids"],
            "retrieved_chunk_ids": retrieved_chunk_ids,
            "scores": [
                round(score, 6)
                for score in scores
            ],
        })

    metrics = calculate_metrics(results)

    index_size_bytes = (
        embeddings.nbytes
        + vector_count_index_size(index)
    )

    output = {
        "meeting_id": "M-001",
        "index_type": "HNSW",
        "embedding_model": MODEL_NAME,
        "dimension": int(embeddings.shape[1]),
        "vector_count": int(embeddings.shape[0]),
        "top_k": TOP_K,
        "parameters": {
            "M": M,
            "ef_construction": EF_CONSTRUCTION,
            "ef_search": EF_SEARCH,
            "seed": SEED,
        },
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

    output_path = (
        BASE_DIR
        / "embeddings/hnsw_index_evaluation.json"
    )

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            output,
            f,
            indent=2,
        )

    print("\n" + "=" * 70)
    print("HNSW Vector Index Evaluation")
    print("=" * 70)

    print(json.dumps(
        output["metrics"],
        indent=2,
    ))

    print("\nResults:")
    print(output_path)

    print("=" * 70)


def vector_count_index_size(index):
    try:
        return index.index_file_size()
    except AttributeError:
        return 0


if __name__ == "__main__":
    main()