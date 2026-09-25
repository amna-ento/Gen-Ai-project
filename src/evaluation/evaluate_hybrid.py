import json
import time
from pathlib import Path

import chromadb
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer


BASE_DIR = Path("data/meetings/valid_input/M-001")

CHUNKS_PATH = BASE_DIR / "chunks/chunks.json"
EVALUATION_PATH = (
    BASE_DIR
    / "evaluation/retrieval_questions.json"
)
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


def load_questions():
    with open(
        EVALUATION_PATH,
        "r",
        encoding="utf-8",
    ) as f:
        return json.load(f)["questions"]


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

    return client.get_collection(
        name=COLLECTION_NAME
    )


def normalize_scores(scores):
    if not scores:
        return []

    min_score = min(scores)
    max_score = max(scores)

    if max_score == min_score:
        return [1.0] * len(scores)

    return [
        (score - min_score)
        / (max_score - min_score)
        for score in scores
    ]


def get_semantic_results(
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

        semantic_results.append(
            {
                "chunk_id": chunk_id,
                "score": 1.0 - distance,
                "text": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
            }
        )

    return semantic_results


def get_bm25_results(
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


def get_hybrid_results(
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


def calculate_metrics(
    retrieved_ids,
    relevant_ids,
):
    relevant_set = set(relevant_ids)

    retrieved_relevant = [
        chunk_id
        for chunk_id in retrieved_ids
        if chunk_id in relevant_set
    ]

    recall = (
        len(retrieved_relevant)
        / len(relevant_set)
        if relevant_set
        else 0.0
    )

    hit = (
        1.0
        if retrieved_relevant
        else 0.0
    )

    precision = (
        len(retrieved_relevant)
        / len(retrieved_ids)
        if retrieved_ids
        else 0.0
    )

    reciprocal_rank = 0.0

    for rank, chunk_id in enumerate(
        retrieved_ids,
        start=1,
    ):
        if chunk_id in relevant_set:
            reciprocal_rank = 1.0 / rank
            break

    return (
        recall,
        hit,
        precision,
        reciprocal_rank,
    )


def main():
    chunks = load_chunks()
    questions = load_questions()

    if not chunks:
        raise ValueError(
            f"No chunks found for {MEETING_ID}."
        )

    model = SentenceTransformer(
        MODEL_NAME
    )

    collection = load_chroma()

    bm25 = build_bm25_index(
        chunks
    )

    total_recall = 0.0
    total_hit = 0.0
    total_precision = 0.0
    total_mrr = 0.0
    total_latency = 0.0

    print("\n" + "=" * 100)
    print("HYBRID RETRIEVAL EVALUATION")
    print("=" * 100)

    print(f"Meeting:         {MEETING_ID}")
    print(f"Top-K:           {TOP_K}")
    print(f"Questions:       {len(questions)}")
    print(
        f"Semantic Weight: {SEMANTIC_WEIGHT}"
    )
    print(
        f"BM25 Weight:     {BM25_WEIGHT}"
    )

    print("=" * 100)

    for item in questions:
        question_id = item["question_id"]
        question = item["question"]
        relevant_ids = item[
            "relevant_chunk_ids"
        ]

        start = time.perf_counter()

        semantic_results = get_semantic_results(
            collection,
            model,
            question,
        )

        bm25_results = get_bm25_results(
            bm25,
            chunks,
            question,
        )

        hybrid_results = get_hybrid_results(
            semantic_results,
            bm25_results,
        )

        end = time.perf_counter()

        latency_ms = (
            end - start
        ) * 1000

        retrieved_ids = [
            result["chunk_id"]
            for result in hybrid_results
        ]

        (
            recall,
            hit,
            precision,
            reciprocal_rank,
        ) = calculate_metrics(
            retrieved_ids,
            relevant_ids,
        )

        total_recall += recall
        total_hit += hit
        total_precision += precision
        total_mrr += reciprocal_rank
        total_latency += latency_ms

        print(
            f"\n{question_id}: {question}"
        )

        print(
            f"Recall@{TOP_K}: "
            f"{recall:.4f}"
        )

        print(
            f"Hit@{TOP_K}: "
            f"{hit:.4f}"
        )

        print(
            f"Precision@{TOP_K}: "
            f"{precision:.4f}"
        )

        print(
            f"MRR@{TOP_K}: "
            f"{reciprocal_rank:.4f}"
        )

        print(
            f"Latency: "
            f"{latency_ms:.4f} ms"
        )

        print(
            "Retrieved: "
            + ", ".join(retrieved_ids)
        )

        print(
            "Relevant:  "
            + ", ".join(relevant_ids)
        )

    count = len(questions)

    avg_recall = total_recall / count
    avg_hit = total_hit / count
    avg_precision = total_precision / count
    avg_mrr = total_mrr / count
    avg_latency = total_latency / count

    print("\n" + "=" * 100)
    print("FINAL HYBRID RESULTS")
    print("=" * 100)

    print(
        f"Recall@{TOP_K}:     "
        f"{avg_recall:.4f}"
    )

    print(
        f"Hit@{TOP_K}:        "
        f"{avg_hit:.4f}"
    )

    print(
        f"Precision@{TOP_K}:  "
        f"{avg_precision:.4f}"
    )

    print(
        f"MRR@{TOP_K}:        "
        f"{avg_mrr:.4f}"
    )

    print(
        f"Avg Latency:        "
        f"{avg_latency:.4f} ms"
    )

    print("=" * 100)


if __name__ == "__main__":
    main()