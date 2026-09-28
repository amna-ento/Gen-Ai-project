import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer


MODEL_NAME = "BAAI/bge-small-en-v1.5"


def load_chunks(meeting_dir: str | Path) -> list:
    meeting_dir = Path(meeting_dir)

    chunks_path = (
        meeting_dir
        / "chunks"
        / "chunks.json"
    )

    if not chunks_path.exists():
        raise FileNotFoundError(
            f"Chunks file not found: {chunks_path}"
        )

    with chunks_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        data = json.load(file)

    chunks = data.get("chunks", [])

    if not chunks:
        raise ValueError(
            f"No chunks found for {meeting_dir.name}."
        )

    return chunks


def generate_embeddings(
    chunks: list,
    meeting_dir: str | Path,
) -> Path:

    meeting_dir = Path(meeting_dir)

    model = SentenceTransformer(
        MODEL_NAME
    )

    texts = [
        chunk["text"]
        for chunk in chunks
    ]

    embeddings = model.encode(
        texts,
        batch_size=16,
        show_progress_bar=False,
        normalize_embeddings=True,
    )

    embeddings = np.asarray(
        embeddings,
        dtype=np.float32,
    )

    if len(embeddings) != len(chunks):
        raise ValueError(
            "Embedding count does not match "
            "chunk count."
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

    if not np.all(
        np.isfinite(embeddings)
    ):
        raise ValueError(
            "Embeddings contain invalid values."
        )

    output_dir = (
        meeting_dir
        / "embeddings"
        / "bge_small"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    embeddings_path = (
        output_dir
        / "embeddings.npy"
    )

    metadata_path = (
        output_dir
        / "metadata.json"
    )

    np.save(
        embeddings_path,
        embeddings,
    )

    metadata = []

    for chunk in chunks:
        metadata.append(
            {
                "meeting_id": chunk[
                    "meeting_id"
                ],
                "chunk_id": chunk[
                    "chunk_id"
                ],
                "speakers": chunk.get(
                    "speakers",
                    [],
                ),
                "start_time": chunk[
                    "start_time"
                ],
                "end_time": chunk[
                    "end_time"
                ],
                "topic": chunk.get(
                    "topic",
                    "",
                ),
                "sentiment": chunk.get(
                    "sentiment",
                    "",
                ),
                "word_count": chunk.get(
                    "word_count",
                    0,
                ),
                "segment_count": chunk.get(
                    "segment_count",
                    0,
                ),
            }
        )

    with metadata_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metadata,
            file,
            indent=2,
            ensure_ascii=False,
        )

    return embeddings_path


def build_embeddings(
    meeting_dir: str | Path,
) -> Path:

    chunks = load_chunks(
        meeting_dir
    )

    return generate_embeddings(
        chunks,
        meeting_dir,
    )