import time

from src.conversation.conversation_manager import ConversationManager

from src.reranking.colbert_reranker import (
    build_hybrid_retrieval,
    load_colbert_model,
    rerank_with_colbert,
)

from src.generation.context_builder import build_context

from src.generation.citation_validator import (
    validate_citations,
    extract_citations,
)

from src.generation.llm_generator import (
    generate_answer,
)

from src.storage.meeting_paths import (
    validate_meeting_id,
)


def simple_reference_resolution(
    question: str,
    conversation_history: str,
) -> str:
    """
    Adds recent conversation context to follow-up questions.
    """

    if not conversation_history.strip():
        return question

    reference_words = [
        "he",
        "she",
        "they",
        "him",
        "her",
        "them",
        "it",
        "this",
        "that",
        "these",
        "those",
        "his",
        "their",
    ]

    question_lower = question.lower()

    contains_reference = any(
        f" {word} " in f" {question_lower} "
        for word in reference_words
    )

    if not contains_reference:
        return question

    recent_history = conversation_history[-4000:]

    return (
        f"Previous conversation:\n"
        f"{recent_history}\n\n"
        f"Current question:\n"
        f"{question}"
    )


def verify_meeting_results(
    results: list,
    expected_meeting_id: str,
) -> None:

    expected_meeting_id = validate_meeting_id(
        expected_meeting_id
    )

    for result in results:

        metadata = result.get(
            "metadata",
            {},
        )

        meeting_id = metadata.get(
            "meeting_id"
        )

        if meeting_id != expected_meeting_id:
            raise ValueError(
                "Meeting isolation violation: "
                f"expected {expected_meeting_id}, "
                f"found {meeting_id}."
            )


def process_turn(
    manager: ConversationManager,
    question: str,
    colbert_model,
) -> dict:

    start_time = time.time()

    meeting_id = validate_meeting_id(
        manager.meeting_id
    )

    history = manager.get_history_text()

    retrieval_question = simple_reference_resolution(
        question=question,
        conversation_history=history,
    )

    print(
        "\nBuilding Hybrid Top-10..."
    )

    retrieval_output = build_hybrid_retrieval(
        retrieval_question,
        meeting_id,
    )

    hybrid_results = retrieval_output[
        "results"
    ]

    if not hybrid_results:
        raise ValueError(
            "No hybrid retrieval results found."
        )

    print(
        f"Retrieved "
        f"{len(hybrid_results)} "
        f"candidate chunks."
    )

    verify_meeting_results(
        hybrid_results,
        meeting_id,
    )

    print(
        "\nRunning ColBERT reranking..."
    )

    reranked_results, rerank_latency_ms = (
        rerank_with_colbert(
            colbert_model,
            retrieval_question,
            hybrid_results,
            meeting_id,
        )
    )

    if not reranked_results:
        raise ValueError(
            "No reranked results found."
        )

    verify_meeting_results(
        reranked_results,
        meeting_id,
    )

    context = build_context(
        reranked_results,
        meeting_id,
    )

    print(
        "\nGenerating answer..."
    )

    result = generate_answer(
        question=question,
        context=context,
        meeting_id=meeting_id,
        conversation_history=history,
    )

    answer = result["answer"]

    citation_validation = validate_citations(
        answer=answer,
        reranked_results=reranked_results,
        meeting_id=meeting_id,
    )

    citations = extract_citations(
        answer
    )

    total_latency = (
        time.time()
        - start_time
    )

    manager.add_turn(
        question=question,
        answer=answer,
        citations=citations,
        generation_provider=result["provider"],
        generation_model=result["model"],
        retrieval_question=retrieval_question,
    )

    return {
        "meeting_id": meeting_id,
        "question": question,
        "retrieval_question": retrieval_question,
        "answer": answer,
        "citations": citations,
        "citation_validation": citation_validation,
        "provider": result["provider"],
        "model": result["model"],
        "generation_latency_seconds": result[
            "latency_seconds"
        ],
        "rerank_latency_ms": round(
            rerank_latency_ms,
            3,
        ),
        "total_latency_seconds": round(
            total_latency,
            3,
        ),
    }


def display_turn_result(
    result: dict,
) -> None:

    print(
        "\n"
        + "=" * 100
    )

    print(
        "ANSWER"
    )

    print(
        "=" * 100
    )

    print(
        result["answer"]
    )

    print(
        "\nMeeting ID:",
        result["meeting_id"],
    )

    print(
        "Provider:",
        result["provider"],
    )

    print(
        "Model:",
        result["model"],
    )

    print(
        "Generation Latency:",
        result[
            "generation_latency_seconds"
        ],
        "seconds",
    )

    print(
        "ColBERT Latency:",
        result[
            "rerank_latency_ms"
        ],
        "ms",
    )

    print(
        "Total Turn Latency:",
        result[
            "total_latency_seconds"
        ],
        "seconds",
    )

    print(
        "\nCitations:"
    )

    if result["citations"]:

        for citation in result["citations"]:
            print(
                f"- {citation}"
            )

    else:

        print(
            "- No citations found."
        )

    validation = result[
        "citation_validation"
    ]

    print(
        "\nCitation Correct:",
        validation[
            "citation_correct"
        ],
    )

    print(
        "Cross-Meeting Citations:",
        len(
            validation[
                "invalid_meeting_citations"
            ]
        ),
    )

    print(
        "=" * 100
    )


def main():

    print(
        "\n"
        + "=" * 100
    )

    print(
        "PHASE 11 - MULTI-TURN MEETING ISOLATION"
    )

    print(
        "=" * 100
    )

    meeting_id = input(
        "\nEnter meeting ID: "
    ).strip()

    meeting_id = validate_meeting_id(
        meeting_id
    )

    print(
        f"\nMeeting ID: {meeting_id}"
    )

    print(
        "\nType 'exit' to end the conversation."
    )

    manager = ConversationManager(
        meeting_id=meeting_id
    )

    print(
        "\nLoading ColBERT model..."
    )

    colbert_model = load_colbert_model()

    print(
        "ColBERT model ready."
    )

    while True:

        try:

            question = input(
                "\nYou: "
            ).strip()

        except KeyboardInterrupt:

            print(
                "\n\nConversation ended."
            )

            break

        if not question:
            continue

        if question.lower() == "exit":

            print(
                "\nConversation ended."
            )

            break

        try:

            result = process_turn(
                manager=manager,
                question=question,
                colbert_model=colbert_model,
            )

            display_turn_result(
                result
            )

        except Exception as error:

            print(
                "\nERROR:"
            )

            print(
                error
            )

            print(
                "\nThe conversation was not saved for this turn."
            )


if __name__ == "__main__":
    main()