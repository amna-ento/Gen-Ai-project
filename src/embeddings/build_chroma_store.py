import json
from pathlib import Path

import chromadb
import numpy as np


BASE_DIR = Path("data/meetings/valid_input/M-001")

EMBEDDINGS_PATH = BASE_DIR / "embeddings/bge_small/embeddings.npy"
CHUNKS_PATH = BASE_DIR / "chunks/chunks.json"
CHROMA_DIR = BASE_DIR / "embeddings/chroma"

COLLECTION_NAME = "meeting_chunks"


def load_data():
    embeddings = np.load(EMBEDDINGS_PATH)

    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        chunks = json.load(f)["chunks"]

    return embeddings, chunks


def validate_data(embeddings, chunks):
    if len(embeddings) != len(chunks):
        raise ValueError(
            f"Embedding count ({len(embeddings)}) "
            f"does not match chunk count ({len(chunks)})."
        )

    chunk_ids = [chunk["chunk_id"] for chunk in chunks]

    if len(chunk_ids) != len(set(chunk_ids)):
        raise ValueError("Duplicate chunk IDs found.")

    if embeddings.shape[1] != 384:
        raise ValueError(
            f"Expected 384 dimensions, got {embeddings.shape[1]}."
        )

    for chunk in chunks:
        required_fields = [
            "chunk_id",
            "meeting_id",
            "text",
        ]

        for field in required_fields:
            if field not in chunk:
                raise ValueError(
                    f"Missing required field '{field}' "
                    f"in chunk {chunk.get('chunk_id', 'unknown')}."
                )


def build_metadata(chunk):
    return {
        "meeting_id": str(chunk["meeting_id"]),
        "chunk_id": str(chunk["chunk_id"]),
        "speaker": str(chunk.get("speaker", "")),
        "start_time": float(chunk.get("start_time", 0.0)),
        "end_time": float(chunk.get("end_time", 0.0)),
        "duration_seconds": float(
            chunk.get("duration_seconds", 0.0)
        ),
    }


def build_chroma_store(embeddings, chunks):
    CHROMA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    client = chromadb.PersistentClient(
        path=str(CHROMA_DIR)
    )

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={
            "hnsw:space": "cosine"
        },
    )

    ids = [
        str(chunk["chunk_id"])
        for chunk in chunks
    ]

    documents = [
        str(chunk["text"])
        for chunk in chunks
    ]

    metadatas = [
        build_metadata(chunk)
        for chunk in chunks
    ]

    collection.upsert(
        ids=ids,
        embeddings=embeddings.tolist(),
        documents=documents,
        metadatas=metadatas,
    )

    return collection


def main():
    embeddings, chunks = load_data()

    validate_data(
        embeddings,
        chunks,
    )

    collection = build_chroma_store(
        embeddings,
        chunks,
    )

    print("\n" + "=" * 70)
    print("Chroma Store Built")
    print("=" * 70)

    print(f"Collection: {COLLECTION_NAME}")
    print(f"Vectors:    {collection.count()}")
    print(f"Dimension:  {embeddings.shape[1]}")
    print(f"Database:   {CHROMA_DIR}")

    print("=" * 70)


if __name__ == "__main__":
    main()