import json
from pathlib import Path

import chromadb
import numpy as np


MEETING_ID = "M-001"

CHROMA_DIR = Path(
    f"data/meetings/valid_input/{MEETING_ID}/chroma"
)

EXPECTED_CHUNKS = 25

MODELS = {
    "bge_small": 384,
    "bge_base": 768,
    "qwen3_0.6b": 1024,
}


def get_directory_size(path):
    total_size = 0

    for file in path.rglob("*"):
        if file.is_file():
            total_size += file.stat().st_size

    return total_size


def format_size(size_bytes):
    return f"{size_bytes / 1024:.2f} KB"


def evaluate_model(model_key, expected_dimension):
    model_path = CHROMA_DIR / model_key

    client = chromadb.PersistentClient(
        path=str(model_path)
    )

    collection = client.get_collection(
        name="meeting_chunks"
    )

    result = collection.get(
        include=[
            "embeddings",
            "documents",
            "metadatas",
        ]
    )

    ids = result.get("ids")

    if ids is None:
        ids = []

    documents = result.get("documents")

    if documents is None:
        documents = []

    metadatas = result.get("metadatas")

    if metadatas is None:
        metadatas = []

    embeddings = result.get("embeddings")

    if embeddings is None:
        embeddings = []

    vector_count = len(ids)

    duplicate_ids = vector_count - len(set(ids))

    missing_documents = sum(
        1
        for document in documents
        if not document or not document.strip()
    )

    missing_metadata = 0
    wrong_meeting_ids = 0

    required_metadata = [
        "meeting_id",
        "speaker",
        "start_time",
        "end_time",
        "topic",
        "sentiment",
        "source_segments",
    ]

    for metadata in metadatas:
        if metadata is None:
            metadata = {}

        if any(
            field not in metadata
            for field in required_metadata
        ):
            missing_metadata += 1

        if metadata.get("meeting_id") != MEETING_ID:
            wrong_meeting_ids += 1

    if len(embeddings) > 0:
        embedding_array = np.asarray(
            embeddings,
            dtype=np.float32,
        )

        actual_dimension = (
            embedding_array.shape[1]
            if embedding_array.ndim == 2
            else 0
        )

        invalid_values = not np.isfinite(
            embedding_array
        ).all()

        norms = np.linalg.norm(
            embedding_array,
            axis=1,
        )

        normalized_embeddings = np.allclose(
            norms,
            1.0,
            atol=0.01,
        )

    else:
        actual_dimension = 0
        invalid_values = True
        normalized_embeddings = False

    storage_size = get_directory_size(
        model_path
    )

    print(f"\n{'=' * 55}")
    print(f"MODEL: {model_key}")
    print(f"{'=' * 55}")

    print(
        f"Expected vectors:       {EXPECTED_CHUNKS}"
    )

    print(
        f"Stored vectors:         {vector_count}"
    )

    print(
        f"Expected dimension:     {expected_dimension}"
    )

    print(
        f"Actual dimension:       {actual_dimension}"
    )

    print(
        f"Duplicate IDs:          {duplicate_ids}"
    )

    print(
        f"Missing documents:      {missing_documents}"
    )

    print(
        f"Missing metadata:       {missing_metadata}"
    )

    print(
        f"Wrong meeting IDs:      {wrong_meeting_ids}"
    )

    print(
        f"Invalid vector values:  {invalid_values}"
    )

    print(
        f"Normalized vectors:     {normalized_embeddings}"
    )

    print(
        f"Storage size:           {format_size(storage_size)}"
    )

    passed = (
        vector_count == EXPECTED_CHUNKS
        and duplicate_ids == 0
        and missing_documents == 0
        and missing_metadata == 0
        and wrong_meeting_ids == 0
        and actual_dimension == expected_dimension
        and not invalid_values
        and normalized_embeddings
    )

    print(
        f"Validation status:      "
        f"{'PASS' if passed else 'FAIL'}"
    )

    return {
        "model": model_key,
        "vector_count": vector_count,
        "embedding_dimension": actual_dimension,
        "duplicate_ids": duplicate_ids,
        "missing_documents": missing_documents,
        "missing_metadata": missing_metadata,
        "wrong_meeting_ids": wrong_meeting_ids,
        "invalid_vector_values": invalid_values,
        "normalized_embeddings": normalized_embeddings,
        "storage_size_bytes": storage_size,
        "status": "PASS" if passed else "FAIL",
    }


def main():
    results = []

    for model_key, dimension in MODELS.items():
        result = evaluate_model(
            model_key,
            dimension,
        )

        results.append(result)

    output_path = Path(
        f"data/meetings/valid_input/"
        f"{MEETING_ID}/chroma/index_evaluation.json"
    )

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            {
                "meeting_id": MEETING_ID,
                "expected_vectors": EXPECTED_CHUNKS,
                "results": results,
            },
            file,
            indent=2,
        )

    print("\nIndex evaluation completed.")
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()