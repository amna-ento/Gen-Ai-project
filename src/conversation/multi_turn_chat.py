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


MEETING_ID = "M-001"


def simple_reference_resolution(
    question: str,
    conversation_history: str,
) -> str:
    """
    Adds recent conversation context to follow-up questions.

    This is a lightweight reference-resolution step.
    The original user question is preserved.
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

    for result in results:

        metadata = result.get(
            "metadata",
            {},
        )

        meeting_id = metadata.get(
            "meeting_id"
        )

        if meeting_id is not None:
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

    history = manager.get_history_text()

    retrieval_question = simple_reference_resolution(
        question=question,
        conversation_history=history,
    )

    print(
        "\nBuilding Hybrid Top-10..."
    )

    hybrid_results = build_hybrid_retrieval(
        retrieval_question
    )

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
        MEETING_ID,
    )

    print(
        "\nRunning ColBERT reranking..."
    )

    reranked_results, rerank_latency_ms = (
        rerank_with_colbert(
            colbert_model,
            retrieval_question,
            hybrid_results,
        )
    )

    if not reranked_results:
        raise ValueError(
            "No reranked results found."
        )

    verify_meeting_results(
        reranked_results,
        MEETING_ID,
    )

    context = build_context(
        reranked_results
    )

    print(
        "\nGenerating answer..."
    )

    result = generate_answer(
        question=question,
        context=context,
        conversation_history=history,
    )

    answer = result["answer"]

    citation_validation = validate_citations(
        answer=answer,
        reranked_results=reranked_results,
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
        "\nProvider:",
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
        "=" * 100
    )


def main():

    print(
        "\n"
        + "=" * 100
    )

    print(
        "PHASE 10 - MULTI-TURN MEETING CHAT"
    )

    print(
        "=" * 100
    )

    print(
        f"Meeting ID: {MEETING_ID}"
    )

    print(
        "\nType 'exit' to end the conversation."
    )

    manager = ConversationManager(
        meeting_id=MEETING_ID
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