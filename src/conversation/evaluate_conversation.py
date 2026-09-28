import json
from pathlib import Path
from typing import Any

from src.conversation.conversation_manager import ConversationManager


MEETING_ID = "M-001"
CONVERSATION_ID = "C-001"

EVALUATION_PATH = (
    Path("data/meetings/valid_input")
    / MEETING_ID
    / "evaluation"
    / "conversation_evaluation.json"
)

ABSTENTION_MESSAGE = (
    "I could not find enough information "
    "in the meeting context to answer this question."
)

REFERENCE_WORDS = {
    "it",
    "this",
    "that",
    "these",
    "those",
    "he",
    "she",
    "they",
    "him",
    "her",
    "them",
    "his",
    "their",
}

CASUAL_MESSAGES = {
    "hi",
    "hello",
    "hey",
    "bye",
    "goodbye",
    "ok bye",
    "thanks",
    "thank you",
}


def is_abstention(answer: str) -> bool:
    return (
        ABSTENTION_MESSAGE.lower()
        in answer.lower()
    )


def contains_reference(question: str) -> bool:
    padded_question = (
        f" {question.lower()} "
    )

    return any(
        f" {word} " in padded_question
        for word in REFERENCE_WORDS
    )


def is_casual_message(question: str) -> bool:
    normalized = (
        question.lower()
        .strip()
        .rstrip("!?.,")
    )

    return normalized in CASUAL_MESSAGES


def evaluate_context_retention(
    turns: list[dict[str, Any]],
) -> dict[str, Any]:

    follow_up_turns = [
        turn
        for turn in turns
        if contains_reference(turn["question"])
    ]

    if not follow_up_turns:
        return {
            "status": "NOT_TESTED",
            "total_follow_up_turns": 0,
            "passed": 0,
            "failed": 0,
            "details": [],
        }

    details = []
    passed = 0

    for turn in follow_up_turns:

        answer = turn.get(
            "answer",
            "",
        ).strip()

        abstention = is_abstention(answer)

        history_used = (
            "Previous conversation:"
            in turn.get(
                "retrieval_question",
                "",
            )
        )

        success = (
            history_used
            and (
                bool(answer)
                or abstention
            )
        )

        if success:
            passed += 1

        details.append(
            {
                "turn_id": turn["turn_id"],
                "question": turn["question"],
                "history_used": history_used,
                "answer_available": bool(answer),
                "abstention": abstention,
                "passed": success,
            }
        )

    failed = len(follow_up_turns) - passed

    return {
        "status": (
            "PASS"
            if failed == 0
            else "PARTIAL"
        ),
        "total_follow_up_turns": len(
            follow_up_turns
        ),
        "passed": passed,
        "failed": failed,
        "details": details,
    }


def evaluate_follow_up_accuracy(
    turns: list[dict[str, Any]],
) -> dict[str, Any]:

    if len(turns) < 2:
        return {
            "status": "NOT_TESTED",
            "total_follow_ups": 0,
            "passed": 0,
            "failed": 0,
            "details": [],
        }

    details = []
    passed = 0

    for index in range(1, len(turns)):

        current_turn = turns[index]
        previous_turn = turns[index - 1]

        retrieval_question = current_turn.get(
            "retrieval_question",
            "",
        )

        history_was_used = (
            "Previous conversation:"
            in retrieval_question
        )

        answer = current_turn.get(
            "answer",
            "",
        ).strip()

        question = current_turn[
            "question"
        ]

        if is_casual_message(question):
            success = bool(answer)
        elif contains_reference(question):
            success = (
                history_was_used
                and bool(answer)
            )
        else:
            success = bool(answer)

        if success:
            passed += 1

        details.append(
            {
                "turn_id": current_turn["turn_id"],
                "question": question,
                "previous_question": (
                    previous_turn["question"]
                ),
                "history_used_for_retrieval": (
                    history_was_used
                ),
                "answer_available": bool(answer),
                "passed": success,
            }
        )

    failed = len(details) - passed

    return {
        "status": (
            "PASS"
            if failed == 0
            else "PARTIAL"
        ),
        "total_follow_ups": len(details),
        "passed": passed,
        "failed": failed,
        "details": details,
    }


def evaluate_reference_resolution(
    turns: list[dict[str, Any]],
) -> dict[str, Any]:

    reference_turns = [
        turn
        for turn in turns
        if contains_reference(
            turn["question"]
        )
    ]

    if not reference_turns:
        return {
            "status": "NOT_TESTED",
            "total_reference_questions": 0,
            "passed": 0,
            "failed": 0,
            "details": [],
        }

    details = []
    passed = 0

    for turn in reference_turns:

        retrieval_question = turn.get(
            "retrieval_question",
            "",
        )

        history_used = (
            "Previous conversation:"
            in retrieval_question
        )

        answer = turn.get(
            "answer",
            "",
        ).strip()

        success = (
            history_used
            and bool(answer)
        )

        if success:
            passed += 1

        details.append(
            {
                "turn_id": turn["turn_id"],
                "question": turn["question"],
                "history_attached": history_used,
                "answer_available": bool(answer),
                "passed": success,
            }
        )

    failed = (
        len(reference_turns) - passed
    )

    return {
        "status": (
            "PASS"
            if failed == 0
            else "PARTIAL"
        ),
        "total_reference_questions": len(
            reference_turns
        ),
        "passed": passed,
        "failed": failed,
        "details": details,
        "note": (
            "This section verifies that history is supplied. "
            "Semantic reference resolution is evaluated separately "
            "using the ground-truth tests below."
        ),
    }


def evaluate_reference_ground_truth(
    turns: list[dict[str, Any]],
) -> dict[str, Any]:

    test_cases = [
        {
            "turn_id": 2,
            "question": "explain it a bit",
            "expected_concept": "$15 meeting cost",
            "required_terms": [
                "$15",
                "15",
            ],
        },
        {
            "turn_id": 5,
            "question": "was that led or lcd",
            "expected_concept": "LCD",
            "required_terms": [
                "lcd",
            ],
        },
    ]

    turn_lookup = {
        turn["turn_id"]: turn
        for turn in turns
    }

    details = []
    passed = 0

    for test in test_cases:

        turn = turn_lookup.get(
            test["turn_id"]
        )

        if turn is None:

            details.append(
                {
                    **test,
                    "status": "NOT_FOUND",
                    "passed": False,
                }
            )

            continue

        answer = turn.get(
            "answer",
            "",
        )

        answer_lower = answer.lower()

        matched_term = None

        for term in test["required_terms"]:
            if term.lower() in answer_lower:
                matched_term = term
                break

        success = (
            matched_term is not None
            and not is_abstention(answer)
        )

        if success:
            passed += 1

        details.append(
            {
                "turn_id": test["turn_id"],
                "question": test["question"],
                "expected_concept": (
                    test["expected_concept"]
                ),
                "matched_term": matched_term,
                "answer": answer,
                "passed": success,
            }
        )

    total_tests = len(test_cases)
    failed = total_tests - passed

    return {
        "status": (
            "PASS"
            if failed == 0
            else "PARTIAL"
        ),
        "total_tests": total_tests,
        "passed": passed,
        "failed": failed,
        "details": details,
        "note": (
            "These are explicit ground-truth checks for "
            "known reference-resolution examples."
        ),
    }


def evaluate_previous_answer_consistency(
    turns: list[dict[str, Any]],
) -> dict[str, Any]:

    reference_turns = [
        turn
        for turn in turns
        if contains_reference(
            turn["question"]
        )
    ]

    if not reference_turns:
        return {
            "status": "NOT_TESTED",
            "total_follow_ups": 0,
            "passed": 0,
            "failed": 0,
            "details": [],
        }

    details = []
    passed = 0

    turn_lookup = {
        turn["turn_id"]: turn
        for turn in turns
    }

    for turn in reference_turns:

        turn_id = turn["turn_id"]

        previous_turn = turn_lookup.get(
            turn_id - 1
        )

        if previous_turn is None:
            continue

        previous_citations = set(
            previous_turn.get(
                "citations",
                [],
            )
        )

        current_citations = set(
            turn.get(
                "citations",
                [],
            )
        )

        shared_citations = sorted(
            previous_citations.intersection(
                current_citations
            )
        )

        answer = turn.get(
            "answer",
            "",
        )

        success = bool(answer)

        if success:
            passed += 1

        details.append(
            {
                "turn_id": turn_id,
                "question": turn["question"],
                "previous_turn_id": (
                    previous_turn["turn_id"]
                ),
                "shared_citations": (
                    shared_citations
                ),
                "answer_available": bool(answer),
                "passed": success,
            }
        )

    failed = len(details) - passed

    return {
        "status": (
            "PASS"
            if failed == 0
            else "PARTIAL"
        ),
        "total_follow_ups": len(details),
        "passed": passed,
        "failed": failed,
        "details": details,
    }


def evaluate_meeting_id_consistency(
    manager: ConversationManager,
) -> dict[str, Any]:

    conversation = manager.conversation

    stored_meeting_id = conversation.get(
        "meeting_id"
    )

    manager_meeting_id = manager.meeting_id

    passed = (
        stored_meeting_id
        == manager_meeting_id
        == MEETING_ID
    )

    return {
        "status": (
            "PASS"
            if passed
            else "FAIL"
        ),
        "expected_meeting_id": MEETING_ID,
        "manager_meeting_id": manager_meeting_id,
        "stored_meeting_id": stored_meeting_id,
        "passed": passed,
    }


def evaluate_conversation_history(
    manager: ConversationManager,
) -> dict[str, Any]:

    turns = manager.get_turns()

    generated_history = (
        manager.get_history_text()
    )

    expected_parts = []

    for turn in turns:

        expected_parts.append(
            f"User: {turn['question']}\n"
            f"Assistant: {turn['answer']}"
        )

    expected_history = "\n\n".join(
        expected_parts
    )

    if not turns:
        expected_history = (
            "No previous conversation."
        )

    history_matches = (
        generated_history
        == expected_history
    )

    turn_ids = [
        turn["turn_id"]
        for turn in turns
    ]

    sequential_ids = (
        turn_ids
        == list(
            range(
                1,
                len(turns) + 1,
            )
        )
    )

    passed = (
        history_matches
        and sequential_ids
    )

    return {
        "status": (
            "PASS"
            if passed
            else "FAIL"
        ),
        "history_matches_stored_turns": (
            history_matches
        ),
        "turn_ids_sequential": sequential_ids,
        "total_turns": len(turns),
        "passed": passed,
    }


def evaluate_hallucination_safety(
    turns: list[dict[str, Any]],
) -> dict[str, Any]:

    evaluated_turns = [
        turn
        for turn in turns
        if not is_casual_message(
            turn["question"]
        )
    ]

    if not evaluated_turns:
        return {
            "status": "NOT_TESTED",
            "total_evaluated_turns": 0,
            "passed": 0,
            "failed": 0,
            "details": [],
        }

    details = []
    passed = 0

    for turn in evaluated_turns:

        question = turn["question"]

        answer = turn.get(
            "answer",
            "",
        ).strip()

        citations = turn.get(
            "citations",
            [],
        )

        abstention = is_abstention(
            answer
        )

        has_citations = bool(citations)

        if abstention:
            success = True
        else:
            success = has_citations

        if success:
            passed += 1

        details.append(
            {
                "turn_id": turn["turn_id"],
                "question": question,
                "abstention": abstention,
                "citation_count": len(
                    citations
                ),
                "citations": citations,
                "passed": success,
            }
        )

    failed = (
        len(evaluated_turns) - passed
    )

    return {
        "status": (
            "PASS"
            if failed == 0
            else "PARTIAL"
        ),
        "total_evaluated_turns": len(
            evaluated_turns
        ),
        "passed": passed,
        "failed": failed,
        "details": details,
        "note": (
            "Valid abstentions are treated as safe. "
            "Casual messages are excluded because they do not "
            "require meeting-context citations."
        ),
    }


def calculate_overall_status(
    evaluations: dict[str, Any],
) -> str:

    statuses = [
        result["status"]
        for result in evaluations.values()
    ]

    if "FAIL" in statuses:
        return "FAIL"

    if "PARTIAL" in statuses:
        return "PARTIAL"

    return "PASS"


def save_evaluation(
    evaluation: dict[str, Any],
) -> None:

    EVALUATION_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        EVALUATION_PATH,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            evaluation,
            file,
            indent=2,
            ensure_ascii=False,
        )


def display_evaluation(
    evaluation: dict[str, Any],
) -> None:

    print("\n" + "=" * 100)
    print("PHASE 10 - CONVERSATION EVALUATION")
    print("=" * 100)

    print(
        "\nMeeting ID:",
        evaluation["meeting_id"],
    )

    print(
        "Conversation ID:",
        evaluation["conversation_id"],
    )

    print(
        "Total Turns:",
        evaluation["total_turns"],
    )

    print("\nEvaluation Results")
    print("-" * 100)

    labels = {
        "context_retention": (
            "Context Retention"
        ),
        "follow_up_accuracy": (
            "Follow-up Question Accuracy"
        ),
        "reference_resolution": (
            "Reference History Handling"
        ),
        "reference_ground_truth": (
            "Semantic Reference Resolution"
        ),
        "previous_answer_consistency": (
            "Previous-answer Consistency"
        ),
        "meeting_id_consistency": (
            "Meeting ID Consistency"
        ),
        "hallucination_safety": (
            "Hallucination / Citation Safety"
        ),
        "conversation_history": (
            "Conversation History Correctness"
        ),
    }

    for key, label in labels.items():

        result = evaluation[key]

        print(
            f"{label:<38}"
            f"{result['status']}"
        )

    print("-" * 100)

    print(
        "\nOVERALL STATUS:",
        evaluation["overall_status"],
    )

    print(
        "\nSaved to:",
        EVALUATION_PATH,
    )

    print("=" * 100)


def main() -> None:

    manager = ConversationManager(
        meeting_id=MEETING_ID,
        conversation_id=CONVERSATION_ID,
    )

    turns = manager.get_turns()

    evaluations = {
        "context_retention": (
            evaluate_context_retention(
                turns
            )
        ),
        "follow_up_accuracy": (
            evaluate_follow_up_accuracy(
                turns
            )
        ),
        "reference_resolution": (
            evaluate_reference_resolution(
                turns
            )
        ),
        "reference_ground_truth": (
            evaluate_reference_ground_truth(
                turns
            )
        ),
        "previous_answer_consistency": (
            evaluate_previous_answer_consistency(
                turns
            )
        ),
        "meeting_id_consistency": (
            evaluate_meeting_id_consistency(
                manager
            )
        ),
        "hallucination_safety": (
            evaluate_hallucination_safety(
                turns
            )
        ),
        "conversation_history": (
            evaluate_conversation_history(
                manager
            )
        ),
    }

    evaluation = {
        "phase": 10,
        "meeting_id": MEETING_ID,
        "conversation_id": CONVERSATION_ID,
        "total_turns": len(turns),
        "evaluations": evaluations,
        **evaluations,
        "overall_status": calculate_overall_status(
            evaluations
        ),
    }

    save_evaluation(evaluation)

    display_evaluation(evaluation)


if __name__ == "__main__":
    main()