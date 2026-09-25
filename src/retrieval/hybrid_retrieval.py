import json
import time
from pathlib import Path

import chromadb
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer


BASE_DIR = Path("data/meetings/valid_input/M-001")

CHUNKS_PATH = BASE_DIR / "chunks/chunks.json"
CHROMA_DIR = BASE_DIR / "embeddings/chroma"

COLLECTION_NAME = "meeting_chunks"

MEETING_ID = "M-001"

MODEL_NAME = "BAAI/bge-small-en-v1.5"

TOP_K = 10

SEMANTIC_WEIGHT = 0.6
BM25_WEIGHT = 0.4


def load_chunks():
    with open(
        CHUNKS_PATH,
        "r",
        encoding="utf-8",
    ) as f:
        chunks = json.load(f)["chunks"]

    return [
        chunk
        for chunk in chunks
        if str(chunk["meeting_id"]) == MEETING_ID
    ]


def tokenize(text):
    return text.lower().split()


def build_bm25_index(chunks):
    tokenized_documents = [
        tokenize(chunk["text"])
        for chunk in chunks
    ]

    return BM25Okapi(
        tokenized_documents
    )


def load_chroma():
    client = chromadb.PersistentClient(
        path=str(CHROMA_DIR)
    )

    collection = client.get_collection(
        name=COLLECTION_NAME
    )

    return collection


def normalize_scores(scores):
    min_score = min(scores)
    max_score = max(scores)

    if max_score == min_score:
        return [1.0 for _ in scores]

    return [
        (score - min_score)
        / (max_score - min_score)
        for score in scores
    ]


def semantic_search(
    collection,
    model,
    question,
):
    query_embedding = model.encode(
        question,
        normalize_embeddings=True,
    )

    results = collection.query(
        query_embeddings=[
            query_embedding.tolist()
        ],
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

    semantic_results = []

    for i, chunk_id in enumerate(
        results["ids"][0]
    ):
        distance = results["distances"][0][i]

        semantic_score = 1.0 - distance

        semantic_results.append(
            {
                "chunk_id": chunk_id,
                "score": semantic_score,
                "distance": distance,
                "text": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
            }
        )

    return semantic_results


def bm25_search(
    bm25,
    chunks,
    question,
):
    query_tokens = tokenize(question)

    scores = bm25.get_scores(
        query_tokens
    )

    ranked_indices = sorted(
        range(len(scores)),
        key=lambda index: scores[index],
        reverse=True,
    )[:TOP_K]

    bm25_results = []

    for index in ranked_indices:
        bm25_results.append(
            {
                "chunk_id": chunks[index]["chunk_id"],
                "score": float(scores[index]),
                "text": chunks[index]["text"],
                "metadata": chunks[index],
            }
        )

    return bm25_results


def build_hybrid_results(
    semantic_results,
    bm25_results,
):
    semantic_scores = [
        result["score"]
        for result in semantic_results
    ]

    bm25_scores = [
        result["score"]
        for result in bm25_results
    ]

    normalized_semantic = normalize_scores(
        semantic_scores
    )

    normalized_bm25 = normalize_scores(
        bm25_scores
    )

    combined = {}

    for i, result in enumerate(
        semantic_results
    ):
        chunk_id = result["chunk_id"]

        combined[chunk_id] = {
            "chunk_id": chunk_id,
            "text": result["text"],
            "metadata": result["metadata"],
            "semantic_score": normalized_semantic[i],
            "bm25_score": 0.0,
        }

    for i, result in enumerate(
        bm25_results
    ):
        chunk_id = result["chunk_id"]

        if chunk_id not in combined:
            combined[chunk_id] = {
                "chunk_id": chunk_id,
                "text": result["text"],
                "metadata": result["metadata"],
                "semantic_score": 0.0,
                "bm25_score": 0.0,
            }

        combined[chunk_id][
            "bm25_score"
        ] = normalized_bm25[i]

    for result in combined.values():
        result["hybrid_score"] = (
            SEMANTIC_WEIGHT
            * result["semantic_score"]
            + BM25_WEIGHT
            * result["bm25_score"]
        )

    ranked_results = sorted(
        combined.values(),
        key=lambda result: result[
            "hybrid_score"
        ],
        reverse=True,
    )

    return ranked_results[:TOP_K]


def display_results(
    question,
    results,
    latency_ms,
):
    print("\n" + "=" * 100)
    print("HYBRID RETRIEVAL")
    print("=" * 100)

    print(f"Meeting:          {MEETING_ID}")
    print(f"Top-K:            {TOP_K}")
    print(
        f"Semantic Weight:  {SEMANTIC_WEIGHT}"
    )
    print(
        f"BM25 Weight:      {BM25_WEIGHT}"
    )
    print(f"Question:         {question}")
    print(
        f"Retrieval Latency: {latency_ms:.4f} ms"
    )

    print("=" * 100)

    for rank, result in enumerate(
        results,
        start=1,
    ):
        metadata = result["metadata"]

        print(f"\nRank: {rank}")
        print(
            f"Chunk ID: "
            f"{result['chunk_id']}"
        )
        print(
            f"Hybrid Score: "
            f"{result['hybrid_score']:.4f}"
        )
        print(
            f"Semantic Score: "
            f"{result['semantic_score']:.4f}"
        )
        print(
            f"BM25 Score: "
            f"{result['bm25_score']:.4f}"
        )
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
        print(
            f"Text: {result['text']}"
        )

    print("\n" + "=" * 100)


def main():
    question = input(
        "\nEnter your question: "
    ).strip()

    if not question:
        raise ValueError(
            "Question cannot be empty."
        )

    chunks = load_chunks()

    if not chunks:
        raise ValueError(
            f"No chunks found for "
            f"{MEETING_ID}."
        )

    model = SentenceTransformer(
        MODEL_NAME
    )

    collection = load_chroma()

    bm25 = build_bm25_index(
        chunks
    )

    start = time.perf_counter()

    semantic_results = semantic_search(
        collection,
        model,
        question,
    )

    bm25_results = bm25_search(
        bm25,
        chunks,
        question,
    )

    hybrid_results = build_hybrid_results(
        semantic_results,
        bm25_results,
    )

    end = time.perf_counter()

    latency_ms = (
        end - start
    ) * 1000

    display_results(
        question,
        hybrid_results,
        latency_ms,
    )


if __name__ == "__main__":
    main()