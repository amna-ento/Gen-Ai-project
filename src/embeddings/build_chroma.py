import json
import time
from pathlib import Path

import chromadb


MEETING_ID = "M-001"

EMBEDDINGS_DIR = Path(
    f"data/meetings/valid_input/{MEETING_ID}/embeddings"
)

CHROMA_DIR = Path(
    f"data/meetings/valid_input/{MEETING_ID}/chroma"
)


MODELS = {
    "bge_small": "bge_small.json",
    "bge_base": "bge_base.json",
    "qwen3_0.6b": "qwen3_0.6b.json",
}


def load_embeddings(file_name):
    path = EMBEDDINGS_DIR / file_name

    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def create_collection(model_key):
    model_chroma_dir = CHROMA_DIR / model_key
    model_chroma_dir.mkdir(parents=True, exist_ok=True)

    client = chromadb.PersistentClient(
        path=str(model_chroma_dir)
    )

    collection = client.get_or_create_collection(
        name="meeting_chunks"
    )

    return collection


def build_index(model_key, file_name):
    print(f"\nBuilding Chroma index: {model_key}")

    data = load_embeddings(file_name)
    collection = create_collection(model_key)

    records = data["embeddings"]

    ids = []
    embeddings = []
    documents = []
    metadatas = []

    for record in records:
        ids.append(record["chunk_id"])
        embeddings.append(record["embedding"])
        documents.append(
            next(
                chunk["text"]
                for chunk in load_chunks()
                if chunk["chunk_id"] == record["chunk_id"]
            )
        )

        metadata = record["metadata"]

        metadatas.append(
            {
                "meeting_id": metadata["meeting_id"],
                "speaker": json.dumps(metadata["speaker"]),
                "start_time": metadata["start_time"],
                "end_time": metadata["end_time"],
                "topic": json.dumps(metadata["topic"]),
                "sentiment": json.dumps(metadata["sentiment"]),
                "source_segments": json.dumps(
                    metadata["source_segments"]
                ),
            }
        )

    start_time = time.perf_counter()

    collection.upsert(
        ids=ids,
        embeddings=embeddings,
        documents=documents,
        metadatas=metadatas,
    )

    indexing_time = time.perf_counter() - start_time

    stored_count = collection.count()

    print(f"Model: {data['model']}")
    print(f"Embedding dimension: {data['embedding_dimension']}")
    print(f"Vectors inserted: {len(ids)}")
    print(f"Vectors stored: {stored_count}")
    print(f"Indexing time: {indexing_time:.4f} seconds")
    print(f"Chroma path: {CHROMA_DIR / model_key}")


def load_chunks():
    chunks_path = Path(
        f"data/meetings/valid_input/{MEETING_ID}/chunks/chunks.json"
    )

    with open(chunks_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    return data["chunks"]


def main():
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Meeting ID: {MEETING_ID}")

    for model_key, file_name in MODELS.items():
        build_index(model_key, file_name)

    print("\nChroma indexing completed.")


if __name__ == "__main__":
    main()