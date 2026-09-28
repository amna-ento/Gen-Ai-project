import json
from pathlib import Path

import chromadb
import numpy as np

from src.storage.meeting_paths import validate_meeting_id


BASE_DIR = Path("data/meetings/valid_input")

COLLECTION_NAME = "meeting_chunks"


def load_data(meeting_id: str):
    meeting_id = validate_meeting_id(meeting_id)

    meeting_dir = BASE_DIR / meeting_id

    embeddings_path = (
        meeting_dir
        / "embeddings"
        / "bge_small"
        / "embeddings.npy"
    )

    chunks_path = (
        meeting_dir
        / "chunks"
        / "chunks.json"
    )

    if not embeddings_path.exists():
        raise FileNotFoundError(
            f"Embeddings file not found: {embeddings_path}"
        )

    if not chunks_path.exists():
        raise FileNotFoundError(
            f"Chunks file not found: {chunks_path}"
        )

    embeddings = np.load(embeddings_path)

    with chunks_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        data = json.load(file)

    chunks = data["chunks"]

    return embeddings, chunks


def validate_data(
    embeddings,
    chunks,
    meeting_id: str,
):
    meeting_id = validate_meeting_id(meeting_id)

    if len(embeddings) != len(chunks):
        raise ValueError(
            f"Embedding count ({len(embeddings)}) "
            f"does not match chunk count ({len(chunks)})."
        )

    if embeddings.ndim != 2:
        raise ValueError(
            "Embeddings must be a 2D array."
        )

    if embeddings.shape[1] != 384:
        raise ValueError(
            f"Expected 384 dimensions, "
            f"got {embeddings.shape[1]}."
        )

    chunk_ids = [
        str(chunk["chunk_id"])
        for chunk in chunks
    ]

    if len(chunk_ids) != len(set(chunk_ids)):
        raise ValueError(
            "Duplicate chunk IDs found."
        )

    for chunk in chunks:

        if "chunk_id" not in chunk:
            raise ValueError(
                "Missing chunk_id."
            )

        if "meeting_id" not in chunk:
            raise ValueError(
                "Missing meeting_id."
            )

        if "text" not in chunk:
            raise ValueError(
                f"Missing text in "
                f"{chunk['chunk_id']}."
            )

        if chunk["meeting_id"] != meeting_id:
            raise ValueError(
                "Meeting isolation violation: "
                f"expected {meeting_id}, "
                f"found {chunk['meeting_id']}."
            )


def build_metadata(chunk):

    speakers = chunk.get(
        "speakers",
        [],
    )

    if isinstance(speakers, list):
        speakers = ", ".join(
            str(speaker)
            for speaker in speakers
        )

    metadata = {
        "meeting_id": str(
            chunk["meeting_id"]
        ),
        "chunk_id": str(
            chunk["chunk_id"]
        ),
        "speakers": str(
            speakers
        ),
        "start_time": float(
            chunk.get(
                "start_time",
                0.0,
            )
        ),
        "end_time": float(
            chunk.get(
                "end_time",
                0.0,
            )
        ),
        "duration_seconds": float(
            chunk.get(
                "duration_seconds",
                0.0,
            )
        ),
    }

    return metadata


def build_chroma_store(
    meeting_id: str,
    embeddings,
    chunks,
):
    meeting_id = validate_meeting_id(
        meeting_id
    )

    meeting_dir = BASE_DIR / meeting_id

    chroma_dir = (
        meeting_dir
        / "embeddings"
        / "chroma"
    )

    chroma_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    client = chromadb.PersistentClient(
        path=str(chroma_dir)
    )

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={
            "hnsw:space": "cosine"
        },
    )

    ids = []

    documents = []

    metadatas = []

    for chunk in chunks:

        ids.append(
            str(chunk["chunk_id"])
        )

        documents.append(
            str(chunk["text"])
        )

        metadata = build_metadata(
            chunk
        )

        metadatas.append(
            metadata
        )

    collection.upsert(
        ids=ids,
        embeddings=embeddings.tolist(),
        documents=documents,
        metadatas=metadatas,
    )

    return collection


def main():

    meeting_id = input(
        "Enter meeting ID: "
    ).strip()

    meeting_id = validate_meeting_id(
        meeting_id
    )

    embeddings, chunks = load_data(
        meeting_id
    )

    validate_data(
        embeddings,
        chunks,
        meeting_id,
    )

    collection = build_chroma_store(
        meeting_id,
        embeddings,
        chunks,
    )

    chroma_dir = (
        BASE_DIR
        / meeting_id
        / "embeddings"
        / "chroma"
    )

    print("\n" + "=" * 70)
    print("Chroma Store Built")
    print("=" * 70)

    print(
        f"Meeting:    {meeting_id}"
    )

    print(
        f"Collection: {COLLECTION_NAME}"
    )

    print(
        f"Vectors:    {collection.count()}"
    )

    print(
        f"Dimension:  {embeddings.shape[1]}"
    )

    print(
        f"Database:   {chroma_dir}"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()