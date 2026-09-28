import re
from typing import Any


def extract_citations(answer: str) -> list[str]:
    pattern = r"M-\d+_CHUNK_\d+"

    citations = re.findall(pattern, answer)

    return list(dict.fromkeys(citations))


def validate_citations(
    answer: str,
    reranked_results: list[dict[str, Any]],
) -> dict[str, Any]:

    cited_chunk_ids = extract_citations(answer)

    available_chunk_ids = {
        result.get("chunk_id")
        for result in reranked_results
    }

    valid_citations = [
        chunk_id
        for chunk_id in cited_chunk_ids
        if chunk_id in available_chunk_ids
    ]

    invalid_citations = [
        chunk_id
        for chunk_id in cited_chunk_ids
        if chunk_id not in available_chunk_ids
    ]

    return {
        "cited_chunk_ids": cited_chunk_ids,
        "valid_citations": valid_citations,
        "invalid_citations": invalid_citations,
        "citation_count": len(cited_chunk_ids),
        "valid_citation_count": len(valid_citations),
        "citation_correct": (
            len(cited_chunk_ids) > 0
            and len(invalid_citations) == 0
        ),
    }


def display_citation_validation(
    validation_result: dict[str, Any],
):
    print("\n" + "=" * 100)
    print("CITATION VALIDATION")
    print("=" * 100)

    print(
        f"Citations Found: "
        f"{validation_result['citation_count']}"
    )

    print(
        f"Valid Citations: "
        f"{validation_result['valid_citation_count']}"
    )

    print(
        f"Invalid Citations: "
        f"{len(validation_result['invalid_citations'])}"
    )

    print(
        f"Citation Correct: "
        f"{validation_result['citation_correct']}"
    )

    if validation_result["valid_citations"]:
        print("\nValid Citation Sources:")

        for chunk_id in validation_result["valid_citations"]:
            print(f"- {chunk_id}")

    if validation_result["invalid_citations"]:
        print("\nInvalid Citation Sources:")

        for chunk_id in validation_result["invalid_citations"]:
            print(f"- {chunk_id}")

    print("=" * 100)