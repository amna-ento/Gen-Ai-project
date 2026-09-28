
import os

from dotenv import load_dotenv
from groq import Groq
from ollama import Client

from src.reranking.colbert_reranker import (
    build_hybrid_retrieval,
    load_colbert_model,
    rerank_with_colbert,
)

from src.generation.context_builder import (
    build_context,
)

from src.generation.citation_validator import (
    validate_citations,
    display_citation_validation,
)

from generation.generation_guardrails import (
    validate_user_input,
    validate_meeting_context,
    enforce_output_guardrails,
)


load_dotenv()


GROQ_MODEL_NAME = "openai/gpt-oss-20b"
OLLAMA_MODEL_NAME = "qwen3:8b"
OLLAMA_HOST = "http://localhost:11434"

MEETING_ID = "M-001"


def build_prompt(question, context):
    return f"""
You are a meeting assistant.

Answer the user's question using ONLY the provided meeting context.

Rules:
1. Do not use outside knowledge.
2. Do not invent facts.
3. If the answer cannot be found in the context, say:
   "I couldn't find enough information in this meeting to answer that."
4. Keep the answer concise and factual.
5. Support important claims with citations.
6. Every citation must use this exact format:
   [Source: M-001_CHUNK_001]
7. Only cite Chunk IDs that appear in the provided context.
8. Do not create or guess Chunk IDs.
9. Do not cite information that is not supported by the cited chunk.

Meeting Context:
{context}

User Question:
{question}

Answer:
"""


def load_groq():
    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise ValueError(
            "GROQ_API_KEY is not set."
        )

    return Groq(
        api_key=api_key
    )


def generate_with_groq(
    client,
    prompt,
):
    response = client.chat.completions.create(
        model=GROQ_MODEL_NAME,
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        temperature=0,
    )

    return response.choices[0].message.content.strip()


def load_ollama():
    return Client(
        host=OLLAMA_HOST
    )


def generate_with_ollama(
    client,
    prompt,
):
    response = client.chat(
        model=OLLAMA_MODEL_NAME,
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        options={
            "temperature": 0,
        },
    )

    return response["message"]["content"].strip()


def generate_answer(
    question,
    context,
):
    prompt = build_prompt(
        question,
        context,
    )

    try:
        print("\nTrying Groq...")

        groq_client = load_groq()

        answer = generate_with_groq(
            groq_client,
            prompt,
        )

        print(
            "Groq generation successful."
        )

        return {
            "answer": answer,
            "model": GROQ_MODEL_NAME,
            "provider": "Groq",
        }

    except Exception as groq_error:

        print(
            f"\nGroq generation failed: "
            f"{groq_error}"
        )

        print(
            "Falling back to Ollama..."
        )

        try:
            ollama_client = load_ollama()

            answer = generate_with_ollama(
                ollama_client,
                prompt,
            )

            print(
                "Ollama generation successful."
            )

            return {
                "answer": answer,
                "model": OLLAMA_MODEL_NAME,
                "provider": "Ollama",
            }

        except Exception as ollama_error:

            raise RuntimeError(
                "Both LLM providers failed.\n"
                f"Groq error: {groq_error}\n"
                f"Ollama error: {ollama_error}"
            )


def main():

    # ========================================================
    # Input
    # ========================================================

    question = input(
        "\nEnter your question: "
    ).strip()

    # ========================================================
    # Input Guardrail
    # ========================================================

    print(
        "\nRunning input guardrail..."
    )

    input_validation = validate_user_input(
        question
    )

    if not input_validation["valid"]:
        raise ValueError(
            "Input guardrail rejected the question: "
            f"{input_validation['reason']}"
        )

    print(
        "Input guardrail passed."
    )

    # ========================================================
    # Hybrid Retrieval
    # ========================================================

    print(
        "\nBuilding Hybrid Top-10..."
    )

    hybrid_results = build_hybrid_retrieval(
        question
    )

    if not hybrid_results:

        print(
            "No hybrid results found."
        )

        return

    # ========================================================
    # ColBERT Reranking
    # ========================================================

    print(
        "Loading ColBERT model..."
    )

    colbert_model = load_colbert_model()

    reranked_results, latency_ms = (
        rerank_with_colbert(
            colbert_model,
            question,
            hybrid_results,
        )
    )

    if not reranked_results:

        print(
            "No reranked results found."
        )

        return

    # ========================================================
    # Meeting Isolation Guardrail
    # ========================================================

    print(
        "\nRunning meeting isolation guardrail..."
    )

    meeting_validation = validate_meeting_context(
        reranked_results,
        expected_meeting_id=MEETING_ID,
    )

    if not meeting_validation["valid"]:

        raise ValueError(
            "Meeting isolation guardrail failed: "
            f"{meeting_validation['reason']}"
        )

    print(
        "Meeting isolation guardrail passed."
    )

    # ========================================================
    # Context Building
    # ========================================================

    print(
        "\nBuilding context..."
    )

    context = build_context(
        reranked_results
    )

    if not context:

        print(
            "No context available."
        )

        return

    # ========================================================
    # Generation
    # ========================================================

    print(
        "Generating answer..."
    )

    result = generate_answer(
        question,
        context,
    )

    # ========================================================
    # Output Guardrails
    # ========================================================

    print(
        "\nRunning output guardrails..."
    )

    guardrailed_answer = (
        enforce_output_guardrails(
            answer=result["answer"],
            reranked_results=reranked_results,
            expected_meeting_id=MEETING_ID,
        )
    )

    result["answer"] = guardrailed_answer

    print(
        "Output guardrails passed."
    )

    # ========================================================
    # Citation Validation
    # ========================================================

    citation_result = validate_citations(
        result["answer"],
        reranked_results,
    )

    # ========================================================
    # Final Answer
    # ========================================================

    print(
        "\n" + "=" * 100
    )

    print(
        "FINAL ANSWER"
    )

    print(
        "=" * 100
    )

    print(
        result["answer"]
    )

    print(
        "\n" + "-" * 100
    )

    print(
        f"Generation Provider: "
        f"{result['provider']}"
    )

    print(
        f"Generation Model: "
        f"{result['model']}"
    )

    print(
        f"ColBERT Reranking Latency: "
        f"{latency_ms:.4f} ms"
    )

    print(
        "-" * 100
    )

    display_citation_validation(
        citation_result
    )


if __name__ == "__main__":
    main()
