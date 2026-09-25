import time
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer


BASE_DIR = Path("data/meetings/valid_input/M-001")

CHROMA_DIR = BASE_DIR / "embeddings/chroma"

COLLECTION_NAME = "meeting_chunks"
MODEL_NAME = "BAAI/bge-small-en-v1.5"

MEETING_ID = "M-001"
TOP_K = 10


def load_model():
    return SentenceTransformer(MODEL_NAME)


def load_collection():
    client = chromadb.PersistentClient(
        path=str(CHROMA_DIR)
    )

    collection = client.get_collection(
        name=COLLECTION_NAME
    )

    return collection


def semantic_search(
    collection,
    model,
    question,
):
    query_embedding = model.encode(
        question,
        normalize_embeddings=True,
    )

    start_time = time.perf_counter()

    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
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

    end_time = time.perf_counter()

    latency_ms = (
        end_time - start_time
    ) * 1000

    return results, latency_ms


def display_results(
    question,
    results,
    latency_ms,
):
    print("\n" + "=" * 80)
    print("SEMANTIC RETRIEVAL")
    print("=" * 80)

    print(f"Meeting:  {MEETING_ID}")
    print(f"Top-K:    {TOP_K}")
    print(f"Question: {question}")
    print(f"Latency:  {latency_ms:.4f} ms")

    print("=" * 80)

    ids = results["ids"][0]
    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    for rank, (
        chunk_id,
        document,
        metadata,
        distance,
    ) in enumerate(
        zip(
            ids,
            documents,
            metadatas,
            distances,
        ),
        start=1,
    ):
        print(f"\nRank: {rank}")
        print(f"Chunk ID: {chunk_id}")
        print(f"Distance: {distance:.4f}")
        print(f"Speaker: {metadata.get('speaker', '')}")
        print(
            f"Time: "
            f"{metadata.get('start_time', 0.0):.2f}"
            f" - "
            f"{metadata.get('end_time', 0.0):.2f}"
        )
        print(f"Text: {document}")

    print("\n" + "=" * 80)


def main():
    question = input(
        "\nEnter your question: "
    ).strip()

    if not question:
        raise ValueError(
            "Question cannot be empty."
        )

    model = load_model()
    collection = load_collection()

    results, latency_ms = semantic_search(
        collection,
        model,
        question,
    )

    display_results(
        question,
        results,
        latency_ms,
    )


if __name__ == "__main__":
    main()
