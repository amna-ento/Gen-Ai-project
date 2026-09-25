import json
import time
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer


MEETING_ID = "M-001"

CHUNKS_PATH = Path(
    f"data/meetings/valid_input/{MEETING_ID}/chunks/chunks.json"
)

OUTPUT_DIR = Path(
    f"data/meetings/valid_input/{MEETING_ID}/embeddings"
)


MODELS = {
    "bge_small": "BAAI/bge-small-en-v1.5",
    "bge_base": "BAAI/bge-base-en-v1.5",
    "qwen3_0.6b": "Qwen/Qwen3-Embedding-0.6B",
}


def load_chunks():
    with open(CHUNKS_PATH, "r", encoding="utf-8") as file:
        data = json.load(file)

    return data["chunks"]


def generate_embeddings(model_name, chunks):
    print(f"\nLoading model: {model_name}")

    model = SentenceTransformer(model_name)

    texts = [chunk["text"] for chunk in chunks]

    start_time = time.perf_counter()

    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True,
    )

    embedding_time = time.perf_counter() - start_time

    return embeddings, embedding_time


def validate_embeddings(chunks, embeddings):
    if len(chunks) != len(embeddings):
        raise ValueError(
            f"Chunk count {len(chunks)} != embedding count {len(embeddings)}"
        )

    if not np.isfinite(embeddings).all():
        raise ValueError("Embedding contains invalid values.")

    chunk_ids = [chunk["chunk_id"] for chunk in chunks]

    if len(chunk_ids) != len(set(chunk_ids)):
        raise ValueError("Duplicate chunk IDs found.")

    if not all(chunk["text"].strip() for chunk in chunks):
        raise ValueError("Empty chunk text found.")


def save_embeddings(model_key, model_name, chunks, embeddings, embedding_time):
    output_path = OUTPUT_DIR / f"{model_key}.json"

    result = {
        "meeting_id": MEETING_ID,
        "model": model_name,
        "embedding_dimension": int(embeddings.shape[1]),
        "vector_count": int(len(embeddings)),
        "embedding_time_seconds": round(embedding_time, 4),
        "embeddings": [],
    }

    for chunk, embedding in zip(chunks, embeddings):
        result["embeddings"].append(
            {
                "chunk_id": chunk["chunk_id"],
                "embedding": embedding.tolist(),
                "metadata": {
                    "meeting_id": chunk["meeting_id"],
                    "speaker": chunk["speakers"],
                    "start_time": chunk["start_time"],
                    "end_time": chunk["end_time"],
                    "topic": chunk["topic"],
                    "sentiment": chunk["sentiment"],
                    "source_segments": chunk["source_segments"],
                },
            }
        )

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(result, file, indent=2)

    return output_path


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    chunks = load_chunks()

    print(f"Meeting ID: {MEETING_ID}")
    print(f"Total chunks: {len(chunks)}")

    for model_key, model_name in MODELS.items():
        embeddings, embedding_time = generate_embeddings(
            model_name,
            chunks,
        )

        validate_embeddings(chunks, embeddings)

        output_path = save_embeddings(
            model_key,
            model_name,
            chunks,
            embeddings,
            embedding_time,
        )

        print(f"Model: {model_name}")
        print(f"Dimensions: {embeddings.shape[1]}")
        print(f"Vectors: {len(embeddings)}")
        print(f"Embedding time: {embedding_time:.4f} seconds")
        print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()