import json
import time
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer


CHUNKS_PATH = Path("data/meetings/valid_input/M-001/chunks/chunks.json")
EMBEDDINGS_DIR = Path("data/meetings/valid_input/M-001/embeddings")

MODELS = {
    "bge_small": "BAAI/bge-small-en-v1.5",
    "bge_base": "BAAI/bge-base-en-v1.5",
    "qwen3_0.6b": "Qwen/Qwen3-Embedding-0.6B",
}


def load_chunks():
    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    chunks = data["chunks"]

    if not chunks:
        raise ValueError("No chunks found.")

    return chunks


def validate_chunks(chunks):
    required_fields = [
        "meeting_id",
        "chunk_id",
        "speakers",
        "start_time",
        "end_time",
        "text",
        "topic",
        "sentiment",
        "word_count",
        "segment_count",
    ]

    missing_metadata = 0
    duplicate_ids = 0
    empty_text = 0

    chunk_ids = [chunk.get("chunk_id") for chunk in chunks]

    duplicate_ids = len(chunk_ids) - len(set(chunk_ids))

    for chunk in chunks:
        if any(field not in chunk for field in required_fields):
            missing_metadata += 1

        if not chunk.get("text", "").strip():
            empty_text += 1

    return {
        "total_chunks": len(chunks),
        "missing_metadata": missing_metadata,
        "duplicate_chunk_ids": duplicate_ids,
        "empty_text": empty_text,
    }


def generate_embeddings(model_name, model_path, chunks):
    print(f"\n{'=' * 70}")
    print(f"Model: {model_name}")
    print(f"Path:  {model_path}")
    print(f"{'=' * 70}")

    model = SentenceTransformer(model_path)

    texts = [chunk["text"] for chunk in chunks]

    start_time = time.perf_counter()

    embeddings = model.encode(
        texts,
        batch_size=16,
        show_progress_bar=True,
        normalize_embeddings=True,
    )

    generation_time = time.perf_counter() - start_time

    embeddings = np.asarray(embeddings, dtype=np.float32)

    return model, embeddings, generation_time


def validate_embeddings(embeddings, chunks):
    expected_count = len(chunks)

    actual_count = len(embeddings)

    dimension = embeddings.shape[1] if embeddings.ndim == 2 else 0

    missing_vectors = expected_count - actual_count

    invalid_vectors = int(
        np.sum(~np.isfinite(embeddings))
    )

    zero_vectors = int(
        np.sum(np.linalg.norm(embeddings, axis=1) == 0)
    )

    norms = np.linalg.norm(embeddings, axis=1)

    return {
        "expected_vectors": expected_count,
        "actual_vectors": actual_count,
        "dimension": dimension,
        "missing_vectors": missing_vectors,
        "invalid_values": invalid_vectors,
        "zero_vectors": zero_vectors,
        "min_norm": float(np.min(norms)),
        "max_norm": float(np.max(norms)),
        "mean_norm": float(np.mean(norms)),
    }


def save_embeddings(model_name, model_path, model, embeddings, chunks, generation_time):
    model_dir = EMBEDDINGS_DIR / model_name
    model_dir.mkdir(parents=True, exist_ok=True)

    embeddings_path = model_dir / "embeddings.npy"
    metadata_path = model_dir / "metadata.json"

    np.save(embeddings_path, embeddings)

    metadata = []

    for chunk in chunks:
        metadata.append(
            {
                "meeting_id": chunk["meeting_id"],
                "chunk_id": chunk["chunk_id"],
                "speakers": chunk["speakers"],
                "start_time": chunk["start_time"],
                "end_time": chunk["end_time"],
                "topic": chunk["topic"],
                "sentiment": chunk["sentiment"],
                "word_count": chunk["word_count"],
                "segment_count": chunk["segment_count"],
            }
        )

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    embedding_validation = validate_embeddings(embeddings, chunks)

    file_size_bytes = embeddings_path.stat().st_size

    result = {
        "model_name": model_name,
        "model_path": model_path,
        "embedding_validation": embedding_validation,
        "metadata_count": len(metadata),
        "generation_time_seconds": round(generation_time, 6),
        "embedding_file_size_bytes": file_size_bytes,
        "embedding_file_size_kb": round(file_size_bytes / 1024, 2),
        "embedding_file_size_mb": round(file_size_bytes / (1024 * 1024), 4),
    }

    result_path = model_dir / "evaluation.json"

    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    return result


def main():
    print("Loading Phase 5 chunks...")

    chunks = load_chunks()

    print(f"Loaded chunks: {len(chunks)}")

    chunk_validation = validate_chunks(chunks)

    print("\nChunk validation:")
    print(json.dumps(chunk_validation, indent=2))

    all_results = []

    for model_name, model_path in MODELS.items():
        model, embeddings, generation_time = generate_embeddings(
            model_name,
            model_path,
            chunks,
        )

        result = save_embeddings(
            model_name,
            model_path,
            model,
            embeddings,
            chunks,
            generation_time,
        )

        all_results.append(result)

        print("\nEmbedding validation:")
        print(json.dumps(result["embedding_validation"], indent=2))

        print(
            f"Generation time: "
            f"{result['generation_time_seconds']} seconds"
        )

        print(
            f"Storage size: "
            f"{result['embedding_file_size_kb']} KB"
        )

    comparison = {
        "meeting_id": "M-001",
        "source_chunks": len(chunks),
        "chunk_validation": chunk_validation,
        "models": all_results,
    }

    comparison_path = EMBEDDINGS_DIR / "embedding_comparison.json"

    with open(comparison_path, "w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2)

    print(f"\n{'=' * 70}")
    print("Embedding evaluation completed.")
    print(f"Comparison: {comparison_path}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()