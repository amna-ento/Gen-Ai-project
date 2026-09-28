import os
import time

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

from src.storage.meeting_paths import (
    validate_meeting_id,
)


load_dotenv()


GROQ_MODEL_NAME = "openai/gpt-oss-20b"
OLLAMA_MODEL_NAME = "qwen3:8b"
OLLAMA_HOST = "http://localhost:11434"

ABSTENTION_MESSAGE = (
    "I could not find enough information in the meeting context to answer this question."
)


def build_prompt(
    question: str,
    context: str,
    meeting_id: str,
    conversation_history: str = "",
) -> str:

    meeting_id = validate_meeting_id(
        meeting_id
    )

    history_section = ""

    if conversation_history.strip():
        history_section = f"""
PREVIOUS CONVERSATION:

{conversation_history}

Use the previous conversation only to understand references such as:
- he / she / they
- him / her / them
- this / that / these / those
- previous topics or entities

Do not use the previous conversation as an independent source of facts.

All factual claims in the answer must be supported by the CURRENT MEETING CONTEXT.
"""

    prompt = f"""
You are a meeting question-answering assistant.

You are answering questions only about meeting {meeting_id}.

Answer the user's question using ONLY the CURRENT MEETING CONTEXT.

{history_section}

CURRENT MEETING CONTEXT:

{context}

CURRENT USER QUESTION:

{question}

RULES:

1. Answer only from meeting {meeting_id}.

2. Do not use outside knowledge.

3. If the context does not contain enough information, respond exactly with:

{ABSTENTION_MESSAGE}

4. Keep the answer concise and factual.

5. Every factual claim should be supported by the current meeting context.

6. Use citations in this exact format:

[Source: {meeting_id}_CHUNK_001]

7. Only cite chunk IDs belonging to meeting {meeting_id}.

8. Only cite chunk IDs that actually appear in the CURRENT MEETING CONTEXT.

9. Never invent a citation.

10. Never cite another meeting.

11. Do not cite information from the previous conversation.

12. Previous conversation may only help resolve references in the current question.

13. If the question is a follow-up question, answer the follow-up directly.

14. Do not repeat the entire previous answer unless necessary.

Return only the final answer.
"""

    return prompt


def load_groq() -> Groq:

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


def generate_with_groq(
    question: str,
    context: str,
    meeting_id: str,
    conversation_history: str = "",
) -> str:

    prompt = build_prompt(
        question=question,
        context=context,
        meeting_id=meeting_id,
        conversation_history=conversation_history,
    )

    client = load_groq()

    response = client.chat.completions.create(
        model=GROQ_MODEL_NAME,
        messages=[
            {
                "role": "system",
                "content": (
                    "You answer questions about "
                    "meeting transcripts. "
                    "Use only the supplied "
                    "meeting context."
                ),
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        temperature=0,
    )

    answer = (
        response
        .choices[0]
        .message
        .content
    )

    if not answer:
        raise ValueError(
            "Groq returned an empty response."
        )

    return answer.strip()


def load_ollama() -> Client:

    return Client(
        host=OLLAMA_HOST
    )


def generate_with_ollama(
    question: str,
    context: str,
    meeting_id: str,
    conversation_history: str = "",
) -> str:

    prompt = build_prompt(
        question=question,
        context=context,
        meeting_id=meeting_id,
        conversation_history=conversation_history,
    )

    client = load_ollama()

    response = client.chat(
        model=OLLAMA_MODEL_NAME,
        messages=[
            {
                "role": "system",
                "content": (
                    "You answer questions about "
                    "meeting transcripts. "
                    "Use only the supplied "
                    "meeting context."
                ),
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        options={
            "temperature": 0,
        },
    )

    answer = (
        response[
            "message"
        ][
            "content"
        ]
    )

    if not answer:
        raise ValueError(
            "Ollama returned an empty response."
        )

    return answer.strip()


def generate_answer(
    question: str,
    context: str,
    meeting_id: str,
    conversation_history: str = "",
) -> dict:

    meeting_id = validate_meeting_id(
        meeting_id
    )

    start_time = time.time()

    groq_error = None

    try:

        answer = generate_with_groq(
            question=question,
            context=context,
            meeting_id=meeting_id,
            conversation_history=conversation_history,
        )

        latency = (
            time.time()
            - start_time
        )

        return {
            "answer": answer,
            "provider": "Groq",
            "model": GROQ_MODEL_NAME,
            "latency_seconds": round(
                latency,
                3,
            ),
        }

    except Exception as error:

        groq_error = str(error)

    try:

        answer = generate_with_ollama(
            question=question,
            context=context,
            meeting_id=meeting_id,
            conversation_history=conversation_history,
        )

        latency = (
            time.time()
            - start_time
        )

        return {
            "answer": answer,
            "provider": "Ollama",
            "model": OLLAMA_MODEL_NAME,
            "latency_seconds": round(
                latency,
                3,
            ),
        }

    except Exception as ollama_error:

        raise RuntimeError(
            "Both Groq and Ollama generation failed.\n"
            f"Groq error: {groq_error}\n"
            f"Ollama error: {ollama_error}"
        )


def main():

    print(
        "\n"
        + "=" * 100
    )

    print(
        "PHASE 11 - GENERATION + CITATION ISOLATION"
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

    question = input(
        "Enter your question: "
    ).strip()

    if not question:
        print(
            "Question cannot be empty."
        )
        return

    print(
        "\nBuilding Hybrid Top-10..."
    )

    retrieval_output = (
        build_hybrid_retrieval(
            question,
            meeting_id,
        )
    )

    hybrid_results = retrieval_output[
        "results"
    ]

    if not hybrid_results:

        print(
            "\nNo hybrid results found."
        )

        return

    print(
        f"Retrieved "
        f"{len(hybrid_results)} "
        f"candidate chunks."
    )

    print(
        "\nLoading ColBERT model..."
    )

    colbert_model = (
        load_colbert_model()
    )

    (
        reranked_results,
        rerank_latency_ms,
    ) = rerank_with_colbert(
        colbert_model,
        question,
        hybrid_results,
        meeting_id,
    )

    if not reranked_results:

        print(
            "\nNo reranked results found."
        )

        return

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
    )

    answer = result[
        "answer"
    ]

    citation_validation = (
        validate_citations(
            answer=answer,
            reranked_results=reranked_results,
            meeting_id=meeting_id,
        )
    )

    print(
        "\n"
        + "=" * 100
    )

    print(
        "FINAL ANSWER"
    )

    print(
        "=" * 100
    )

    print(
        answer
    )

    print(
        "\nMeeting ID:",
        meeting_id,
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
        result["latency_seconds"],
        "seconds",
    )

    print(
        "ColBERT Reranking Latency:",
        round(
            rerank_latency_ms,
            3,
        ),
        "ms",
    )

    display_citation_validation(
        citation_validation
    )


if __name__ == "__main__":
    main()