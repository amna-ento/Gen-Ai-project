import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer


BASE_DIR = Path("data/meetings/valid_input/M-001")
CHUNKS_PATH = BASE_DIR / "chunks/chunks.json"
QUERIES_PATH = BASE_DIR / "embeddings/embedding_queries.json"
EMBEDDINGS_DIR = BASE_DIR / "embeddings"

MODELS = {
    "bge_small": "BAAI/bge-small-en-v1.5",
    "bge_base": "BAAI/bge-base-en-v1.5",
    "qwen3_0.6b": "Qwen/Qwen3-Embedding-0.6B",
}

TOP_K = 5


def load_chunks():
    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    return data["chunks"]


def load_queries():
    with open(QUERIES_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def load_embeddings(model_name):
    path = EMBEDDINGS_DIR / model_name / "embeddings.npy"
    return np.load(path)


def cosine_similarity(query_embedding, embeddings):
    return embeddings @ query_embedding


def calculate_metrics(results, queries):
    recall_values = []
    hit_values = []
    reciprocal_ranks = []

    for result, query in zip(results, queries):
        expected = set(query["expected_chunk_ids"])
        retrieved = result["retrieved_chunk_ids"]

        relevant_retrieved = expected.intersection(retrieved)

        recall = len(relevant_retrieved) / len(expected)

        hit = 1.0 if relevant_retrieved else 0.0

        reciprocal_rank = 0.0

        for rank, chunk_id in enumerate(retrieved, start=1):
            if chunk_id in expected:
                reciprocal_rank = 1.0 / rank
                break

        recall_values.append(recall)
        hit_values.append(hit)
        reciprocal_ranks.append(reciprocal_rank)

    return {
        "Recall@5": round(float(np.mean(recall_values)), 4),
        "Hit@5": round(float(np.mean(hit_values)), 4),
        "MRR@5": round(float(np.mean(reciprocal_ranks)), 4),
    }


def evaluate_model(model_name, model_path, chunks, queries):
    print(f"\n{'=' * 70}")
    print(f"Evaluating: {model_name}")
    print(f"{'=' * 70}")

    embeddings = load_embeddings(model_name)

    chunk_ids = [chunk["chunk_id"] for chunk in chunks]

    model = SentenceTransformer(model_path)

    query_texts = [item["query"] for item in queries]

    query_embeddings = model.encode(
        query_texts,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    query_embeddings = np.asarray(query_embeddings, dtype=np.float32)

    results = []

    for query, query_embedding in zip(queries, query_embeddings):
        scores = cosine_similarity(query_embedding, embeddings)

        ranked_indices = np.argsort(scores)[::-1][:TOP_K]

        retrieved_chunk_ids = [
            chunk_ids[index]
            for index in ranked_indices
        ]

        retrieved_scores = [
            float(scores[index])
            for index in ranked_indices
        ]

        results.append(
            {
                "query_id": query["query_id"],
                "query": query["query"],
                "expected_chunk_ids": query["expected_chunk_ids"],
                "retrieved_chunk_ids": retrieved_chunk_ids,
                "scores": [
                    round(score, 6)
                    for score in retrieved_scores
                ],
            }
        )

    metrics = calculate_metrics(results, queries)

    return {
        "model_name": model_name,
        "model_path": model_path,
        "top_k": TOP_K,
        "metrics": metrics,
        "queries": results,
    }


def main():
    chunks = load_chunks()
    queries = load_queries()

    all_results = {}

    for model_name, model_path in MODELS.items():
        result = evaluate_model(
            model_name,
            model_path,
            chunks,
            queries,
        )

        all_results[model_name] = result

        print("\nMetrics:")
        print(json.dumps(result["metrics"], indent=2))

    output = {
        "meeting_id": "M-001",
        "total_chunks": len(chunks),
        "total_queries": len(queries),
        "top_k": TOP_K,
        "models": all_results,
    }

    output_path = EMBEDDINGS_DIR / "embedding_quality_evaluation.json"

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print(f"\n{'=' * 70}")
    print("Semantic embedding evaluation completed.")
    print(f"Results: {output_path}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()