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
HASH_BITS = 12
NUM_TABLES = 5
HAMMING_RADIUS = 2
SEED = 42


def load_data():
    embeddings = np.load(EMBEDDINGS_PATH)

    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        chunks = json.load(f)["chunks"]

    with open(QUERIES_PATH, "r", encoding="utf-8") as f:
        queries = json.load(f)

    return embeddings, chunks, queries


def generate_hash_planes(dimension, rng):
    return rng.normal(
        size=(NUM_TABLES, dimension, HASH_BITS)
    ).astype(np.float32)


def build_index(embeddings):
    start_time = time.perf_counter()

    rng = np.random.default_rng(SEED)

    dimension = embeddings.shape[1]

    hash_planes = generate_hash_planes(
        dimension,
        rng,
    )

    hash_tables = []

    for table in range(NUM_TABLES):
        projections = embeddings @ hash_planes[table]
        hash_codes = (projections >= 0).astype(np.uint8)

        table_dict = {}

        for vector_index, code in enumerate(hash_codes):
            key = tuple(code.tolist())

            if key not in table_dict:
                table_dict[key] = []

            table_dict[key].append(vector_index)

        hash_tables.append(table_dict)

    build_time = time.perf_counter() - start_time

    return hash_planes, hash_tables, build_time


def hamming_distance(code_a, code_b):
    return np.count_nonzero(code_a != code_b)


def get_candidates(
    query_embedding,
    hash_planes,
    hash_tables,
):
    candidate_indices = set()

    for table_index in range(NUM_TABLES):
        projections = query_embedding @ hash_planes[table_index]
        query_code = (projections >= 0).astype(np.uint8)

        table = hash_tables[table_index]

        for bucket_code, indices in table.items():
            distance = hamming_distance(
                query_code,
                np.asarray(bucket_code, dtype=np.uint8),
            )

            if distance <= HAMMING_RADIUS:
                candidate_indices.update(indices)

    return list(candidate_indices)


def search(
    embeddings,
    query_embedding,
    hash_planes,
    hash_tables,
):
    candidates = get_candidates(
        query_embedding,
        hash_planes,
        hash_tables,
    )

    if not candidates:
        return np.array([], dtype=int), np.array([]), 0

    candidate_vectors = embeddings[candidates]

    scores = candidate_vectors @ query_embedding

    order = np.argsort(scores)[::-1][:TOP_K]

    ranked_indices = np.asarray(
        [candidates[index] for index in order],
        dtype=int,
    )

    ranked_scores = scores[order]

    return ranked_indices, ranked_scores, len(candidates)


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

    hash_planes, hash_tables, build_time = build_index(
        embeddings
    )

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
    candidate_counts = []

    for query, query_embedding in zip(
        queries,
        query_embeddings,
    ):
        start_time = time.perf_counter()

        ranked_indices, scores, candidate_count = search(
            embeddings,
            query_embedding,
            hash_planes,
            hash_tables,
        )

        latency = time.perf_counter() - start_time

        latencies.append(latency)
        candidate_counts.append(candidate_count)

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
            "candidate_count": candidate_count,
        })

    metrics = calculate_metrics(results)

    hash_plane_size = hash_planes.nbytes
    table_size = sum(
        len(bucket) * 8
        for table in hash_tables
        for bucket in table.values()
    )

    index_size_bytes = (
        hash_plane_size
        + table_size
    )

    output = {
        "meeting_id": "M-001",
        "index_type": "LSH",
        "embedding_model": MODEL_NAME,
        "dimension": int(embeddings.shape[1]),
        "vector_count": int(embeddings.shape[0]),
        "top_k": TOP_K,
        "parameters": {
            "hash_bits": HASH_BITS,
            "num_tables": NUM_TABLES,
            "hamming_radius": HAMMING_RADIUS,
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
            "average_candidates": round(
                float(np.mean(candidate_counts)),
                2,
            ),
        },
        "queries": results,
    }

    output_path = (
        BASE_DIR
        / "embeddings/lsh_index_evaluation.json"
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
    print("LSH Vector Index Evaluation")
    print("=" * 70)

    print(json.dumps(
        output["metrics"],
        indent=2,
    ))

    print("\nResults:")
    print(output_path)

    print("=" * 70)


if __name__ == "__main__":
    main()