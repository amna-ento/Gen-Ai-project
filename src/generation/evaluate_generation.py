import json
import os
import re
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from groq import Groq
from ollama import Client

from src.generation.citation_validator import (
    validate_citations,
)
from src.generation.context_builder import (
    build_context,
)
from src.generation.llm_generator import (
    generate_answer,
)
from src.reranking.colbert_reranker import (
    build_hybrid_retrieval,
    load_colbert_model,
    rerank_with_colbert,
)


load_dotenv()


# ============================================================
# Configuration
# ============================================================

MEETING_ID = "M-001"

OUTPUT_PATH = Path(
    "data/meetings/valid_input/M-001/evaluation/"
    "generation_evaluation.json"
)

GROQ_MODEL_NAME = "openai/gpt-oss-20b"
OLLAMA_MODEL_NAME = "qwen3:8b"
OLLAMA_HOST = "http://localhost:11434"


# ============================================================
# Evaluation Dataset
# ============================================================

EVALUATION_DATASET = [
    {
        "id": "GQ001",
        "question": "What was the central idea of the meeting?",
        "answerable": True,
    },
    {
        "id": "GQ002",
        "question": "What prototype versions were discussed?",
        "answerable": True,
    },
    {
        "id": "GQ003",
        "question": (
            "What were the main design requirements "
            "for the remote control?"
        ),
        "answerable": True,
    },
    {
        "id": "GQ004",
        "question": "What was discussed about the LCD?",
        "answerable": True,
    },
    {
        "id": "GQ005",
        "question": (
            "Who was responsible for approving the final design?"
        ),
        "answerable": False,
    },
    {
        "id": "GQ006",
        "question": "What was the meeting's budget?",
        "answerable": False,
    },
]


# ============================================================
# Judge Prompt
# ============================================================

def build_judge_prompt(
    metric: str,
    question: str,
    context: str,
    answer: str,
) -> str:

    if metric == "faithfulness":

        task = """
Check whether the factual claims in the generated answer
are supported by the retrieved meeting context.

1.0 = fully supported
0.5 = partially supported
0.0 = unsupported
"""

    elif metric == "answer_relevance":

        task = """
Check whether the generated answer directly answers the question.

1.0 = directly answers
0.5 = partially answers
0.0 = does not answer
"""

    elif metric == "context_relevance":

        task = """
Check whether the retrieved meeting context contains information
relevant to the question.

1.0 = highly relevant
0.5 = partially relevant
0.0 = irrelevant
"""

    else:
        raise ValueError(
            f"Unknown evaluation metric: {metric}"
        )

    return f"""
Evaluate this meeting RAG system.

Metric:
{metric}

Task:
{task}

Question:
{question}

Retrieved Meeting Context:
{context}

Generated Answer:
{answer}

Return ONLY:

SCORE: X

X must be a number between 0.0 and 1.0.

Do not explain.
Do not summarize.
Do not answer the meeting question.
Do not think aloud.
Do not output any other text.
""".strip()


# ============================================================
# Score Parsing
# ============================================================

def parse_score(response: str) -> float:

    if not response:
        raise ValueError(
            "Judge returned an empty response."
        )

    match = re.search(
        r"SCORE\s*:\s*(0(?:\.\d+)?|1(?:\.0+)?)",
        response,
        re.IGNORECASE,
    )

    if not match:
        raise ValueError(
            "Could not parse SCORE from judge response:\n"
            f"{response}"
        )

    score = float(
        match.group(1)
    )

    if not 0.0 <= score <= 1.0:
        raise ValueError(
            f"Score must be between 0 and 1. "
            f"Got: {score}"
        )

    return score


# ============================================================
# Groq Judge
# ============================================================

def load_judge_groq():

    api_key = os.getenv(
        "GROQ_API_KEY"
    )

    if not api_key:
        raise ValueError(
            "GROQ_API_KEY is not set."
        )

    return Groq(
        api_key=api_key
    )


def generate_judge_with_groq(
    client,
    prompt: str,
) -> str:

    response = client.chat.completions.create(
        model=GROQ_MODEL_NAME,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a strict evaluation judge. "
                    "Return only SCORE: X."
                ),
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        temperature=0,
        max_tokens=20,
    )

    content = (
        response.choices[0]
        .message
        .content
    )

    if not content:
        raise ValueError(
            "Groq judge returned an empty response. "
            f"Full response: {response}"
        )

    return content.strip()


# ============================================================
# Ollama Judge
# ============================================================

def load_judge_ollama():

    return Client(
        host=OLLAMA_HOST
    )


def generate_judge_with_ollama(
    client,
    prompt: str,
) -> str:

    response = client.chat(
        model=OLLAMA_MODEL_NAME,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a strict evaluation judge. "
                    "Do not think or explain. "
                    "Return only SCORE: X."
                ),
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        think=False,
        options={
            "temperature": 0,
            "num_predict": 50,
        },
    )

    message = response.get(
        "message",
        {},
    )

    content = message.get(
        "content",
        "",
    )

    if not content:
        raise ValueError(
            "Ollama judge returned an empty response. "
            f"Full response: {response}"
        )

    return content.strip()


# ============================================================
# LLM Evaluation
# ============================================================

def evaluate_with_llm(
    metric: str,
    question: str,
    context: str,
    answer: str,
) -> float:

    prompt = build_judge_prompt(
        metric=metric,
        question=question,
        context=context,
        answer=answer,
    )

    print(
        f"\nEvaluating {metric}..."
    )

    groq_error = None
    ollama_error = None

    # --------------------------------------------------------
    # Try Groq
    # --------------------------------------------------------

    try:

        print(
            "Trying Groq judge..."
        )

        groq_client = load_judge_groq()

        response = generate_judge_with_groq(
            groq_client,
            prompt,
        )

        print(
            f"Groq raw judge response: "
            f"{response}"
        )

        score = parse_score(
            response
        )

        print(
            f"Groq judge score: "
            f"{score}"
        )

        return score

    except Exception as error:

        groq_error = error

        print(
            f"Groq judge failed: "
            f"{error}"
        )

    # --------------------------------------------------------
    # Fallback to Ollama
    # --------------------------------------------------------

    try:

        print(
            "Falling back to Ollama judge..."
        )

        ollama_client = (
            load_judge_ollama()
        )

        response = (
            generate_judge_with_ollama(
                ollama_client,
                prompt,
            )
        )

        print(
            f"Ollama raw judge response: "
            f"{response}"
        )

        score = parse_score(
            response
        )

        print(
            f"Ollama judge score: "
            f"{score}"
        )

        return score

    except Exception as error:

        ollama_error = error

        print(
            f"Ollama judge failed: "
            f"{error}"
        )

    # --------------------------------------------------------
    # Both providers failed
    # --------------------------------------------------------

    raise RuntimeError(
        "Both LLM judges failed.\n"
        f"Groq error: {groq_error}\n"
        f"Ollama error: {ollama_error}"
    )


# ============================================================
# Citation Correctness
# ============================================================

def calculate_citation_correctness(
    answer: str,
    reranked_results: list[dict[str, Any]],
) -> float:

    validation = validate_citations(
        answer,
        reranked_results,
    )

    if validation["citation_correct"]:
        return 1.0

    return 0.0


# ============================================================
# Hallucination Rate
# ============================================================

def calculate_hallucination_rate(
    faithfulness: float,
) -> float:

    return 1.0 - faithfulness


# ============================================================
# Abstention Correctness
# ============================================================

def calculate_abstention_correctness(
    answer: str,
    answerable: bool,
) -> float:

    abstention_text = (
        "I couldn't find enough information "
        "in this meeting to answer that."
    )

    normalized_answer = (
        answer.strip().lower()
    )

    abstained = (
        abstention_text.lower()
        in normalized_answer
    )

    if answerable:

        if abstained:
            return 0.0

        return 1.0

    if abstained:
        return 1.0

    return 0.0


# ============================================================
# Average Metric
# ============================================================

def average_metric(
    results: list[dict[str, Any]],
    metric: str,
) -> float | None:

    values = [
        result[metric]
        for result in results
        if result.get(metric) is not None
    ]

    if not values:
        return None

    return (
        sum(values)
        / len(values)
    )


# ============================================================
# Evaluate One Question
# ============================================================

def evaluate_question(
    question_data: dict[str, Any],
    colbert_model,
) -> dict[str, Any]:

    question_id = question_data["id"]

    question = question_data[
        "question"
    ]

    answerable = question_data[
        "answerable"
    ]

    print("\n")
    print("=" * 100)
    print(
        f"QUESTION: {question_id}"
    )
    print("=" * 100)
    print(question)

    # --------------------------------------------------------
    # Hybrid Retrieval
    # --------------------------------------------------------

    print(
        "\nBuilding Hybrid Top-10..."
    )

    hybrid_results = (
        build_hybrid_retrieval(
            question
        )
    )

    if not hybrid_results:

        raise RuntimeError(
            f"No hybrid retrieval results "
            f"for {question_id}"
        )

    # --------------------------------------------------------
    # ColBERT Reranking
    # --------------------------------------------------------

    print(
        "Running ColBERT reranking..."
    )

    reranked_results, latency_ms = (
        rerank_with_colbert(
            colbert_model,
            question,
            hybrid_results,
        )
    )

    if not reranked_results:

        raise RuntimeError(
            f"No reranked results "
            f"for {question_id}"
        )

    # --------------------------------------------------------
    # Build Context
    # --------------------------------------------------------

    print(
        "Building context..."
    )

    context = build_context(
        reranked_results
    )

    if not context:

        raise RuntimeError(
            f"No context generated "
            f"for {question_id}"
        )

    # --------------------------------------------------------
    # Generate Answer
    # --------------------------------------------------------

    print(
        "Generating answer..."
    )

    generation_result = (
        generate_answer(
            question,
            context,
        )
    )

    answer = generation_result[
        "answer"
    ]

    # --------------------------------------------------------
    # Detect Empty Answer
    # --------------------------------------------------------

    if not answer or not answer.strip():

        raise RuntimeError(
            f"LLM generation returned "
            f"an empty answer for {question_id}."
        )

    print(
        "\nGenerated Answer:"
    )
    print(answer)

    # --------------------------------------------------------
    # Base Result
    # --------------------------------------------------------

    result = {
        "question_id": question_id,
        "question": question,
        "answerable": answerable,
        "answer": answer,
        "generation_provider": (
            generation_result[
                "provider"
            ]
        ),
        "generation_model": (
            generation_result[
                "model"
            ]
        ),
        "colbert_latency_ms": latency_ms,
        "faithfulness": None,
        "answer_relevance": None,
        "context_relevance": None,
        "citation_correctness": None,
        "hallucination_rate": None,
        "abstention_correctness": None,
    }

    # --------------------------------------------------------
    # Context Relevance
    # --------------------------------------------------------

    result[
        "context_relevance"
    ] = evaluate_with_llm(
        metric="context_relevance",
        question=question,
        context=context,
        answer=answer,
    )

    # --------------------------------------------------------
    # Answerable Questions
    # --------------------------------------------------------

    if answerable:

        result[
            "faithfulness"
        ] = evaluate_with_llm(
            metric="faithfulness",
            question=question,
            context=context,
            answer=answer,
        )

        result[
            "answer_relevance"
        ] = evaluate_with_llm(
            metric="answer_relevance",
            question=question,
            context=context,
            answer=answer,
        )

        result[
            "citation_correctness"
        ] = (
            calculate_citation_correctness(
                answer,
                reranked_results,
            )
        )

        result[
            "hallucination_rate"
        ] = (
            calculate_hallucination_rate(
                result["faithfulness"]
            )
        )

    # --------------------------------------------------------
    # Unanswerable Questions
    # --------------------------------------------------------

    else:

        result[
            "abstention_correctness"
        ] = (
            calculate_abstention_correctness(
                answer,
                answerable=False,
            )
        )

    return result


# ============================================================
# Overall Metrics
# ============================================================

def calculate_overall_metrics(
    results: list[dict[str, Any]],
) -> dict[str, Any]:

    answerable_results = [
        result
        for result in results
        if result["answerable"]
    ]

    unanswerable_results = [
        result
        for result in results
        if not result["answerable"]
    ]

    return {

        # ----------------------------------------------------
        # Counts
        # ----------------------------------------------------

        "answerable_question_count": len(
            answerable_results
        ),

        "unanswerable_question_count": len(
            unanswerable_results
        ),

        # ----------------------------------------------------
        # Answerable Metrics
        # ----------------------------------------------------

        "faithfulness": average_metric(
            answerable_results,
            "faithfulness",
        ),

        "answer_relevance": average_metric(
            answerable_results,
            "answer_relevance",
        ),

        "citation_correctness": average_metric(
            answerable_results,
            "citation_correctness",
        ),

        "hallucination_rate": average_metric(
            answerable_results,
            "hallucination_rate",
        ),

        # ----------------------------------------------------
        # Context Relevance
        # ----------------------------------------------------

        "context_relevance_answerable": (
            average_metric(
                answerable_results,
                "context_relevance",
            )
        ),

        "context_relevance_all": (
            average_metric(
                results,
                "context_relevance",
            )
        ),

        # ----------------------------------------------------
        # Unanswerable Metrics
        # ----------------------------------------------------

        "abstention_correctness": (
            average_metric(
                unanswerable_results,
                "abstention_correctness",
            )
        ),
    }


# ============================================================
# Save Results
# ============================================================

def save_results(
    results: list[dict[str, Any]],
    overall_metrics: dict[str, Any],
):

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output = {
        "meeting_id": MEETING_ID,
        "evaluation_type": "generation",
        "dataset_size": len(
            EVALUATION_DATASET
        ),
        "results": results,
        "overall_metrics": overall_metrics,
    }

    with open(
        OUTPUT_PATH,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            output,
            file,
            indent=4,
            ensure_ascii=False,
        )

    print(
        "\nEvaluation saved to:"
    )
    print(OUTPUT_PATH)


# ============================================================
# Display Overall Metrics
# ============================================================

def display_overall_metrics(
    overall_metrics: dict[str, Any],
):

    print("\n")
    print("=" * 100)
    print(
        "OVERALL GENERATION EVALUATION"
    )
    print("=" * 100)

    print(
        "\nAnswerable Questions:"
    )

    faithfulness = (
        overall_metrics[
            "faithfulness"
        ]
    )

    answer_relevance = (
        overall_metrics[
            "answer_relevance"
        ]
    )

    context_relevance_answerable = (
        overall_metrics[
            "context_relevance_answerable"
        ]
    )

    citation_correctness = (
        overall_metrics[
            "citation_correctness"
        ]
    )

    hallucination_rate = (
        overall_metrics[
            "hallucination_rate"
        ]
    )

    abstention_correctness = (
        overall_metrics[
            "abstention_correctness"
        ]
    )

    context_relevance_all = (
        overall_metrics[
            "context_relevance_all"
        ]
    )

    if faithfulness is not None:

        print(
            f"Faithfulness: "
            f"{faithfulness:.4f}"
        )

    if answer_relevance is not None:

        print(
            f"Answer Relevance: "
            f"{answer_relevance:.4f}"
        )

    if context_relevance_answerable is not None:

        print(
            f"Context Relevance: "
            f"{context_relevance_answerable:.4f}"
        )

    if citation_correctness is not None:

        print(
            f"Citation Correctness: "
            f"{citation_correctness:.4f}"
        )

    if hallucination_rate is not None:

        print(
            f"Hallucination Rate: "
            f"{hallucination_rate:.4f}"
        )

    print(
        "\nUnanswerable Questions:"
    )

    if abstention_correctness is not None:

        print(
            f"Abstention Correctness: "
            f"{abstention_correctness:.4f}"
        )

    print(
        "\nAll Questions:"
    )

    if context_relevance_all is not None:

        print(
            f"Context Relevance: "
            f"{context_relevance_all:.4f}"
        )

    print("=" * 100)


# ============================================================
# Main
# ============================================================

def main():

    print("\n")
    print("=" * 100)
    print(
        "PHASE 9 - GENERATION EVALUATION"
    )
    print("=" * 100)

    print(
        f"\nMeeting ID: {MEETING_ID}"
    )

    print(
        f"Evaluation Questions: "
        f"{len(EVALUATION_DATASET)}"
    )

    # --------------------------------------------------------
    # Load ColBERT Once
    # --------------------------------------------------------

    print(
        "\nLoading ColBERT model..."
    )

    colbert_model = (
        load_colbert_model()
    )

    # --------------------------------------------------------
    # Evaluate Questions
    # --------------------------------------------------------

    results = []

    for question_data in (
        EVALUATION_DATASET
    ):

        result = evaluate_question(
            question_data,
            colbert_model,
        )

        results.append(result)

        # ----------------------------------------------------
        # Display Question Metrics
        # ----------------------------------------------------

        print(
            "\nQuestion Metrics:"
        )

        if result[
            "faithfulness"
        ] is not None:

            print(
                f"Faithfulness: "
                f"{result['faithfulness']:.4f}"
            )

        if result[
            "answer_relevance"
        ] is not None:

            print(
                f"Answer Relevance: "
                f"{result['answer_relevance']:.4f}"
            )

        if result[
            "context_relevance"
        ] is not None:

            print(
                f"Context Relevance: "
                f"{result['context_relevance']:.4f}"
            )

        if result[
            "citation_correctness"
        ] is not None:

            print(
                f"Citation Correctness: "
                f"{result['citation_correctness']:.4f}"
            )

        if result[
            "hallucination_rate"
        ] is not None:

            print(
                f"Hallucination Rate: "
                f"{result['hallucination_rate']:.4f}"
            )

        if result[
            "abstention_correctness"
        ] is not None:

            print(
                f"Abstention Correctness: "
                f"{result['abstention_correctness']:.4f}"
            )

    # --------------------------------------------------------
    # Overall Metrics
    # --------------------------------------------------------

    overall_metrics = (
        calculate_overall_metrics(
            results
        )
    )

    display_overall_metrics(
        overall_metrics
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    save_results(
        results,
        overall_metrics,
    )


if __name__ == "__main__":

    main()