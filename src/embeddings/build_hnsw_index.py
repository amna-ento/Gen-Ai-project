import json
from pathlib import Path

import hnswlib
import numpy as np


BASE_DIR = Path("data/meetings/valid_input/M-001")

EMBEDDINGS_PATH = BASE_DIR / "embeddings/bge_small/embeddings.npy"
CHUNKS_PATH = BASE_DIR / "chunks/chunks.json"
OUTPUT_DIR = BASE_DIR / "embeddings/hnsw"

M = 16
EF_CONSTRUCTION = 200
EF_SEARCH = 50
SEED = 42


def load_data():
    embeddings = np.load(EMBEDDINGS_PATH)

    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        chunks = json.load(f)["chunks"]

    return embeddings, chunks


def build_index(embeddings):
    dimension = embeddings.shape[1]
    vector_count = embeddings.shape[0]

    index = hnswlib.Index(
        space="cosine",
        dim=dimension,
    )

    index.init_index(
        max_elements=vector_count,
        M=M,
        ef_construction=EF_CONSTRUCTION,
        random_seed=SEED,
    )

    labels = np.arange(vector_count)

    index.add_items(
        embeddings,
        labels,
    )

    index.set_ef(EF_SEARCH)

    return index


def main():
    embeddings, chunks = load_data()

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    index = build_index(embeddings)

    index_path = OUTPUT_DIR / "index.bin"
    metadata_path = OUTPUT_DIR / "metadata.json"

    index.save_index(str(index_path))

    metadata = {
        "meeting_id": "M-001",
        "index_type": "HNSW",
        "embedding_model": "BAAI/bge-small-en-v1.5",
        "dimension": int(embeddings.shape[1]),
        "vector_count": int(embeddings.shape[0]),
        "distance": "cosine",
        "parameters": {
            "M": M,
            "ef_construction": EF_CONSTRUCTION,
            "ef_search": EF_SEARCH,
            "seed": SEED,
        },
        "chunk_ids": [
            chunk["chunk_id"]
            for chunk in chunks
        ],
    }

    with open(
        metadata_path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            metadata,
            f,
            indent=2,
        )

    print("\n" + "=" * 70)
    print("HNSW Index Built")
    print("=" * 70)

    print(f"Vectors:   {embeddings.shape[0]}")
    print(f"Dimension: {embeddings.shape[1]}")
    print(f"Index:     {index_path}")
    print(f"Metadata:  {metadata_path}")

    print("=" * 70)


if __name__ == "__main__":
    main()