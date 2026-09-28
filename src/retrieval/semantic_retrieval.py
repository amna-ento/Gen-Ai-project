import time

import chromadb
from sentence_transformers import SentenceTransformer

from src.storage.meeting_paths import (
    get_chroma_dir,
    validate_meeting_id,
)


COLLECTION_NAME = "meeting_chunks"
MODEL_NAME = "BAAI/bge-small-en-v1.5"

TOP_K = 10


def load_model():
    return SentenceTransformer(MODEL_NAME)


def load_collection(meeting_id: str):
    meeting_id = validate_meeting_id(meeting_id)

    chroma_dir = get_chroma_dir(meeting_id)

    client = chromadb.PersistentClient(
        path=str(chroma_dir)
    )

    return client.get_collection(
        name=COLLECTION_NAME
    )


def semantic_search(
    collection,
    model,
    question,
    meeting_id: str,
):
    meeting_id = validate_meeting_id(
        meeting_id
    )

    query_embedding = model.encode(
        question,
        normalize_embeddings=True,
    )

    start_time = time.perf_counter()

    results = collection.query(
        query_embeddings=[
            query_embedding.tolist()
        ],
        n_results=TOP_K,
        where={
            "meeting_id": meeting_id
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

    for metadata in results["metadatas"][0]:
        result_meeting_id = metadata.get(
            "meeting_id"
        )

        if result_meeting_id != meeting_id:
            raise ValueError(
                "Meeting isolation violation: "
                f"expected {meeting_id}, "
                f"found {result_meeting_id}."
            )

    return results, latency_ms


def display_results(
    question,
    results,
    latency_ms,
    meeting_id,
):
    print("\n" + "=" * 80)
    print("SEMANTIC RETRIEVAL")
    print("=" * 80)

    print(f"Meeting:  {meeting_id}")
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
        print(
            f"Speaker: "
            f"{metadata.get('speaker', '')}"
        )
        print(
            f"Time: "
            f"{metadata.get('start_time', 0.0):.2f}"
            f" - "
            f"{metadata.get('end_time', 0.0):.2f}"
        )
        print(f"Text: {document}")

    print("\n" + "=" * 80)


def main():
    meeting_id = input(
        "\nEnter meeting ID: "
    ).strip()

    meeting_id = validate_meeting_id(
        meeting_id
    )

    question = input(
        "Enter your question: "
    ).strip()

    if not question:
        raise ValueError(
            "Question cannot be empty."
        )

    model = load_model()

    collection = load_collection(
        meeting_id
    )

    results, latency_ms = semantic_search(
        collection,
        model,
        question,
        meeting_id,
    )

    display_results(
        question,
        results,
        latency_ms,
        meeting_id,
    )


if __name__ == "__main__":
    main()