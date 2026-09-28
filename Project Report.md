# Meeting RAG: Meeting Intelligence and Question Answering System

## Professional Project Report

---

## 1. Executive Summary

The **Meeting RAG** project is an end-to-end system that converts meeting audio into a searchable, structured, and conversational knowledge source. It validates and assigns a unique meeting ID to each audio file, generates timestamped transcripts using **Faster-Whisper**, performs speaker diarization and alignment with **pyannote**, and extracts topics, sentiment, decisions, and action items. The transcript is then cleaned, divided into metadata-rich chunks, embedded using **BAAI/bge-small-en-v1.5**, and stored in **Chroma**.

For querying, the system combines **semantic retrieval (BGE + Chroma)** with **BM25 lexical retrieval** using a `0.6/0.4` weighting, selects Top-10 candidates, and reranks them with **ColBERT** to obtain the final context. Answers are generated using **Groq (`openai/gpt-oss-20b`)**, with **Ollama (`qwen3:8b`)** as a fallback. Citation validation ensures that cited information belongs to the retrieved context and correct meeting, while multi-turn conversations are maintained through a conversation manager and JSON storage. **Meeting isolation** is enforced throughout the pipeline using IDs such as `M-001` and `M-002`, preventing data from different meetings from being mixed.


---

# 2. Introduction

Meetings contain important organizational knowledge, including discussions, decisions, action items, technical explanations, project updates, and responsibilities. However, meeting recordings are difficult to search directly, while raw transcripts can become very large and difficult to navigate.

Traditional keyword search is also insufficient for questions that use different wording from the original conversation. For example, a meeting may contain the statement:

> "The prototype should use the smaller LCD because of the cost."

A user might later ask:

> "Why did the team choose the smaller display?"

A simple keyword search may not identify the relationship between these statements.

The Meeting RAG system addresses this problem by combining:

* Speech recognition
* Speaker diarization
* Transcript processing
* Semantic embeddings
* Lexical retrieval
* Vector search
* Reranking
* Large language model generation
* Citation validation
* Multi-turn conversation
* Meeting-level isolation

The result is a system that allows users to interact with meeting recordings through natural-language questions.

---

# 3. Problem Statement

Raw meeting recordings contain valuable information but are difficult to search, analyze, and query.

The project addresses the following problems:

1. Audio cannot be directly searched efficiently using natural-language questions.
2. Long transcripts contain too much information for direct LLM processing.
3. Different speakers need to be identified.
4. Important information may be expressed using different words from the user's question.
5. Retrieval systems can return irrelevant chunks.
6. Generated answers may contain unsupported information.
7. Multiple meetings create a risk of cross-meeting information leakage.
8. Follow-up questions require conversation history.
9. External LLM providers may occasionally fail or become unavailable.

The project therefore aims to build a complete pipeline from:

```text
Meeting Audio
      ↓
Structured Meeting Knowledge
      ↓
Searchable Knowledge Base
      ↓
Grounded Question Answering
      ↓
Multi-turn Conversation
```

---

# 4. Project Objectives

The main objectives of the project are:

* Convert meeting audio into timestamped transcripts.
* Identify speakers within the meeting.
* Align speakers with transcript segments.
* Extract structured meeting information.
* Clean transcript data before retrieval processing.
* Divide transcripts into meaningful retrieval chunks.
* Attach useful metadata to every chunk.
* Generate semantic embeddings.
* Store embeddings in a vector database.
* Combine semantic and lexical retrieval.
* Improve retrieval ordering using reranking.
* Generate grounded answers using an LLM.
* Provide source citations for generated answers.
* Validate generated citations.
* Support multi-turn conversations.
* Maintain strict meeting-level data isolation.
* Provide an API interface through FastAPI and Swagger.
* Evaluate individual stages instead of evaluating only the final answer.

---

# 5. System Requirements

## 5.1 Functional Requirements

The system must:

1. Accept supported meeting audio files.
2. Validate uploaded audio.
3. Generate sequential meeting IDs.
4. Store each meeting independently.
5. Transcribe the meeting.
6. Detect speakers.
7. Align speakers with transcript segments.
8. Analyze meeting content.
9. Clean transcript data.
10. Generate chunks.
11. Generate embeddings.
12. Store embeddings in Chroma.
13. Perform semantic retrieval.
14. Perform BM25 retrieval.
15. Combine retrieval results.
16. Rerank retrieved candidates.
17. Build context for the LLM.
18. Generate an answer.
19. Validate citations.
20. Store conversation history.
21. Support follow-up questions.
22. Prevent cross-meeting retrieval.

---

## 5.2 Non-Functional Requirements

The system should provide:


* Meeting isolation
* Source attribution
* Retrieval accuracy
* Generation grounding
* Provider fallback
* Evaluation at individual stages


---

# 6. Technology Stack

| Component                 | Technology                           |
| ------------------------- | ------------------------------------ |
| Programming Language      | Python                               |
| API Framework             | FastAPI                              |
| API Documentation         | Swagger/OpenAPI                      |
| Speech Recognition        | Faster-Whisper                       |
| Diarization               | pyannote.audio                       |
| Transcript Processing     | Custom Python modules                |
| Embeddings                | BAAI/bge-small-en-v1.5               |
| Vector Database           | Chroma                               |
| Lexical Retrieval         | BM25                                 |
| Hybrid Retrieval          | Custom score fusion                  |
| Reranking                 | ColBERT v2                           |
| Primary LLM               | Groq `openai/gpt-oss-20b`            |
| Fallback LLM              | Ollama `qwen3:8b`                    |
| Conversation Storage      | JSON                                 |
| Environment Configuration | python-dotenv                        |
| Audio Processing          | soundfile / FFmpeg-supported formats |

---

# 8. Project Architecture by Phase

## Phase 1 — Audio Ingestion and Transcription

### Purpose

Convert raw meeting audio into a timestamped transcript.

### Flow

```text
User Audio
    ↓
validate_audio_file()
    ↓
generate_meeting_id()
    ↓
Create Meeting Directory
    ↓
Move Audio
    ↓
MeetingTranscriber
    ↓
Faster-Whisper
    ↓
transcript.json
```

### Implementation

The main entry point is:

```text
src/transcription/pipeline.py
```

Important functions/classes:

```text
process_new_meeting()
generate_meeting_id()
validate_audio_file()
MeetingTranscriber
```

### Audio Validation

`audio_validator.py` checks:

* File existence
* File type
* Supported extension
* Audio readability
* Duration

Supported formats include:

```text
.wav
.mp3
.m4a
.flac
.ogg
.aac
```

### Meeting ID Generation

Meeting IDs are sequential:

```text
M-001
M-002
M-003
M-004
...
```

The system scans existing meeting directories and determines the next available number.

### Transcription Model

```text
Faster-Whisper
Model: small
Device: CPU
Compute Type: int8
VAD: enabled
```

### Output

```text
data/meetings/valid_input/M-001/transcript/transcript.json
```

### Evaluation

```text
WER
CER
```

These metrics measure how accurately the generated transcript matches reference text.

---

# 9. Phase 2 — Speaker Diarization and Alignment

## Purpose

Determine who spoke during each part of the meeting and associate speaker labels with transcript segments.

### Architecture

```text
transcript.json
       │
       ├──────────────┐
       │              │
       ▼              ▼
Transcript        Audio
Segments             │
                     ▼
             SpeakerDiarizer
                     │
                     ▼
             pyannote Community-1
                     │
                     ▼
             Speaker Segments
                     │
Transcript ──────────┘
       │
       ▼
SpeakerAligner
       │
       ▼
Dominant Time Overlap
       │
       ▼
speaker_transcript.json
```

### Main Files

```text
src/diarization/diarizer.py
src/diarization/aligner.py
src/diarization/process_diarization.py
src/diarization/save_speaker_transcript.py
```

### Diarization

The system uses:

```text
pyannote/speaker-diarization-community-1
```

The implementation uses:

```text
MPS → if available
CPU → otherwise
```

### Speaker Alignment

For every transcript segment:

1. Compare transcript start/end timestamps.
2. Compare with diarization segments.
3. Calculate overlap.
4. Add overlap duration for each speaker.
5. Select the speaker with the largest overlap.

### Output

```text
speaker_transcript.json
```

Each segment contains:

```text
speaker
start
end
text
```

### Evaluation

```text
DER
Timestamp Accuracy
```

---

# 10. Phase 3 — Meeting Analysis

## Purpose

Convert the speaker-aware transcript into structured meeting information.

### Flow

```text
speaker_transcript.json
        ↓
MeetingAnalyzer
        ↓
Structured Analysis
        ↓
final_analysis.json
```

The analysis contains information such as:

```text
Topics
Sentiment
Decisions
Action Items
```

### Output

```text
analysis/final_analysis.json
```

### Evaluation

The analysis stage is evaluated for:

* Structure correctness
* Field correctness
* Extracted meeting information accuracy

---

# 11. Phase 4 — Transcript Cleaning

## Purpose

Clean the transcript before creating retrieval chunks.

### Flow

```text
speaker_transcript.json
        ↓
TranscriptCleaner
        ↓
Cleaned Transcript
        ↓
cleaned_transcript.json
```

### Why Cleaning Comes Before Chunking

If the transcript contains unnecessary artifacts and those artifacts are embedded directly, they become part of the retrieval representation.

Therefore:

```text
Raw Transcript
      ↓
Cleaning
      ↓
Chunking
      ↓
Embedding
```

rather than:

```text
Raw Transcript
      ↓
Chunking
      ↓
Embedding
      ↓
Cleaning
```

### Evaluation

The cleaning stage checks whether the transcript was cleaned without introducing unwanted changes.

---

# 12. Phase 5 — Chunking and Metadata

## Purpose

Convert the cleaned transcript into retrieval-ready units.

A complete meeting transcript is too large to use directly for retrieval and generation.

Therefore:

```text
Cleaned Transcript
       ↓
Chunking
       ↓
Chunk 1
Chunk 2
Chunk 3
...
Chunk N
```

Each chunk contains metadata.

Typical metadata includes:

```text
meeting_id
chunk_id
speaker
start_time
end_time
topics
sentiment
text
```

### Example

```text
M-001_CHUNK_001

Meeting ID: M-001
Speaker: SPEAKER_01
Time: 00:02:10 - 00:03:20

Text:
...
```

### Output

```text
data/meetings/valid_input/M-001/chunks/chunks.json
```

### Evaluation

Two dedicated evaluations are used:

```text
Chunk Quality
Metadata Quality
```

---

# 13. Phase 6 — Embeddings and Vector Storage

## Purpose

Convert text chunks into numerical vectors that capture semantic meaning.

### Selected Embedding Model

```text
BAAI/bge-small-en-v1.5
```

Embedding dimension:

```text
384
```

### Flow

```text
chunks.json
     ↓
build_embeddings()
     ↓
BGE-small
     ↓
embeddings.npy
     ↓
Chroma
     ↓
meeting_chunks
```

### Chroma

The project uses persistent Chroma storage.

The collection name is:

```text
meeting_chunks
```

### Meeting Isolation

Chroma retrieval uses:

```text
where = {
    "meeting_id": meeting_id
}
```

Therefore the semantic retrieval layer only retrieves vectors belonging to the requested meeting.

### Experiments

The project also experimented with:

```text
BGE-small
BGE-base
Qwen3-Embedding-0.6B
```

and index approaches including:

```text
Flat
HNSW
IVF
LSH
```

### Selected Configuration

```text
Embedding: BGE-small
Vector DB: Chroma
```

### Evaluation

Embedding and index evaluation includes retrieval quality and performance metrics such as:

```text
Recall@K
Hit@K
MRR
Precision
Latency
```

---

# 14. Phase 7 — Hybrid Retrieval

## Purpose

Improve retrieval by combining semantic similarity with lexical matching.

A purely semantic system can miss exact terms.

A purely lexical system can miss semantic relationships.

Therefore the project combines both.

---

## 14.1 Semantic Retrieval

```text
Question
   ↓
BGE-small
   ↓
Query Embedding
   ↓
Chroma
   ↓
Top-10 Semantic Results
```

File:

```text
src/retrieval/semantic_retrieval.py
```

The retrieval query uses:

```text
meeting_id
```

as a Chroma filter.

---

## 14.2 BM25 Retrieval

```text
Question
   ↓
Tokenization
   ↓
BM25
   ↓
Top-10 Lexical Results
```

File:

```text
src/retrieval/bm25_retrieval.py
```

---

## 14.3 Hybrid Fusion

The project uses:

```text
Semantic Weight = 0.6
BM25 Weight     = 0.4
```

The final score is:

```text
Hybrid Score =
    0.6 × Semantic Score
    +
    0.4 × BM25 Score
```

The scores are normalized before fusion.

### Final Result

```text
Question
   │
   ├── Semantic Retrieval
   │
   └── BM25 Retrieval
          │
          ▼
     Score Fusion
          │
          ▼
     Hybrid Top-10
```

### Evaluation

```text
Recall@K
Hit@K
MRR
Precision
Latency
```

---

# 15. Phase 8 — ColBERT Reranking

## Purpose

Hybrid retrieval generates candidates, but the candidates still need better ordering.

The project therefore uses ColBERT.

### Flow

```text
Hybrid Top-10
      ↓
ColBERT
      ↓
Query / Document Encoding
      ↓
Similarity
      ↓
Sort by ColBERT Score
      ↓
Top-5
```

### Model

```text
colbert-ir/colbertv2.0
```

### Implementation

```text
src/reranking/colbert_reranker.py
```

### Why Reranking Is Separate

Retrieval is optimized for finding candidates efficiently.

Reranking is optimized for ordering those candidates more accurately.

Therefore:

```text
Retrieve broadly
       ↓
Rerank deeply
       ↓
Generate from strongest evidence
```

### Evaluation

Reranking is evaluated using:

```text
Recall
MRR
Latency
```

The project also experimented with other reranking approaches including Cross-Encoder, Bi-Encoder, and LLM-based reranking.

---

# 16. Phase 9 — Context Construction and Generation

## 16.1 Context Builder

The Top-5 reranked chunks are converted into structured context.

Each source contains:

```text
Source number
Meeting ID
Chunk ID
Speaker
Start time
End time
Text
```

Example:

```text
[Source 1]
Meeting ID: M-001
Chunk ID: M-001_CHUNK_001
Speaker: SPEAKER_01
Time: 120.2 - 145.8

Text:
...
```

This provides the LLM with both content and provenance.

---

# 17. Generation

The generation prompt explicitly instructs the model to:

* Answer only about the requested meeting.
* Use only the current meeting context.
* Avoid outside knowledge.
* Abstain when sufficient information is unavailable.
* Cite actual chunks.
* Never cite another meeting.
* Never invent citations.
* Use conversation history only for resolving references.

---

## Primary Generator

```text
Provider: Groq
Model: openai/gpt-oss-20b
Temperature: 0
```

---

## Ollama Fallback

If Groq generation fails:

```text
Groq
  │
  ├── Success → Answer
  │
  └── Failure
         ↓
      Ollama
         ↓
      qwen3:8b
```

The fallback logic is implemented inside:

```text
generate_answer()
```

Therefore the system does not immediately fail when the primary cloud generation provider encounters an exception.

If both providers fail, a runtime error is raised containing the failure information from both providers.

---

# 18. Citation Validation

After generation, the answer is passed to:

```text
validate_citations()
```

The validator:

1. Extracts chunk IDs from the answer.
2. Determines which chunks were actually retrieved.
3. Checks whether the cited chunks belong to the requested meeting.
4. Detects invalid citations.
5. Detects cross-meeting citations.
6. Produces a validation result.

The result contains:

```text
meeting_id
cited_chunk_ids
valid_citations
invalid_citations
invalid_meeting_citations
citation_count
valid_citation_count
citation_correct
```

This creates an additional verification layer after generation.

---

# 19. Phase 10 — Multi-Turn Conversation

The project supports conversations rather than isolated questions.

### Conversation Architecture

```text
Question
   ↓
ConversationManager
   ↓
Previous History
   ↓
Reference Resolution
   ↓
Retrieval Question
   ↓
Hybrid Retrieval
   ↓
ColBERT
   ↓
Context
   ↓
Generation
   ↓
Citation Validation
   ↓
Save Turn
```

---

## Conversation Manager

`ConversationManager` handles:

* Loading a conversation
* Creating a conversation
* Retrieving turns
* Building history text
* Adding new turns
* Saving turns
* Clearing a conversation

The default conversation ID is:

```text
C-001
```

---

## Conversation Storage

Conversations are stored as:

```text
data/meetings/valid_input/M-001/conversation/conversation.json
```

Each turn stores:

```text
turn_id
question
retrieval_question
answer
citations
generation_provider
generation_model
timestamp
```

---

# 20. Follow-Up Question Handling

The project includes a lightweight reference-resolution mechanism.

For example:

```text
User:
What did they decide about the prototype?

Assistant:
...

User:
What about its cost?
```

The system detects reference words such as:

```text
he
she
they
him
her
them
it
this
that
these
those
his
their
```

When a reference is detected, recent conversation history is added to the retrieval question.

The important distinction is:

```text
Conversation History
        ↓
Reference Resolution
        ↓
Better Retrieval Question
```

but:

```text
Current Meeting Context
        ↓
Actual factual source
```

Conversation history is therefore not intended to become an independent factual knowledge source.

---

# 21. Phase 11 — Meeting Isolation and Storage

Meeting isolation is one of the most important architectural requirements.

Every meeting receives a unique ID:

```text
M-001
M-002
M-003
M-004
...
```

Each meeting has its own storage:

```text
valid_input/
├── M-001/
├── M-002/
├── M-003/
└── M-004/
```

---

## Isolation Checks

Isolation is enforced at multiple stages.

```text
Requested Meeting ID
        ↓
validate_meeting_id()
        ↓
Meeting-specific filesystem
        ↓
Chroma meeting_id filter
        ↓
Retrieval result validation
        ↓
Reranking result validation
        ↓
Context validation
        ↓
Prompt restriction
        ↓
Citation validation
        ↓
Conversation validation
```

This creates defense-in-depth rather than depending on a single check.

---

# 22. Meeting Path Management

The project centralizes meeting path logic in:

```text
src/storage/meeting_paths.py
```

Important functions include:

```text
validate_meeting_id()
get_meeting_dir()
get_chunks_path()
get_chroma_dir()
get_evaluation_path()
get_conversation_path()
```

The expected meeting ID format is:

```text
M-\d{3}
```

Examples:

```text
M-001
M-002
M-999
```

Invalid IDs are rejected.

---

# 23. API Architecture

The application uses FastAPI.

Main application:

```text
src/api/main.py
```

The application:

```text
FastAPI(...)
        ↓
include_router(router)
```

The API exposes the Meeting RAG functionality through the router layer.

The project also defines Pydantic schemas:

```text
ChatRequest
ChatResponse
MeetingResponse
TranscriptResponse
```

### Chat Request

```text
question: str
```

The question must contain at least one character.

### Chat Response

Contains:

```text
meeting_id
conversation_id
question
answer
citations
```

### Meeting Response

Contains:

```text
meeting_id
status
duration
transcript_segments
```

### Transcript Response

Contains:

```text
meeting_id
segments
```


---

# 26. Project Directory Structure

```text
.
├── chunks/
├── data/
│   └── meetings/
│       ├── user_input/
│       └── valid_input/
│           ├── M-001/
│           │   ├── analysis/
│           │   ├── audio/
│           │   ├── chunks/
│           │   ├── conversation/
│           │   ├── embeddings/
│           │   ├── evaluation/
│           │   ├── reference/
│           │   └── transcript/
│           │
│           ├── M-002/
│           ├── M-003/
│           └── M-004/
│
├── src/
│   ├── analysis/
│   ├── api/
│   ├── chunking/
│   ├── cleaning/
│   ├── conversation/
│   ├── diarization/
│   ├── embeddings/
│   ├── evaluation/
│   ├── generation/
│   ├── reranking/
│   ├── retrieval/
│   ├── storage/
│   └── transcription/
│
├── project_plan.md
├── readme.md
├── requirements.txt
└── week 9 report/
```

---

              |

---

# 28. Evaluation Strategy

A major feature of the project is that evaluation is not limited to the final generated answer.

Each important stage has its own evaluation.

| Phase    | Evaluation                       |
| -------- | -------------------------------- |
| Phase 1  | WER / CER                        |
| Phase 2  | DER / Timestamp Accuracy         |
| Phase 3  | Structure / Analysis Accuracy    |
| Phase 4  | Cleaning Quality                 |
| Phase 5  | Chunk Quality / Metadata Quality |
| Phase 6  | Embedding / Index Metrics        |
| Phase 7  | Retrieval Metrics                |
| Phase 8  | Reranking Metrics                |
| Phase 9  | Generation / Citation Evaluation |
| Phase 10 | Conversation Evaluation          |
| Phase 11 | Meeting Isolation Evaluation     |

This makes the project easier to debug because a poor final answer can be traced back to an earlier stage.



---


# 40. Complete End-to-End Data Flow

```text
                        MEETING AUDIO
                             │
                             ▼
                     Audio Validation
                             │
                             ▼
                       Meeting ID
                             │
                             ▼
                         Whisper
                             │
                             ▼
                      Transcript
                             │
                             ▼
                       Pyannote
                             │
                             ▼
                    Speaker Alignment
                             │
                             ▼
                  Speaker Transcript
                             │
                             ▼
                   Meeting Analysis
                             │
                             ▼
                  Transcript Cleaning
                             │
                             ▼
                         Chunking
                             │
                             ▼
                       Metadata
                             │
                             ▼
                      BGE Embedding
                             │
                             ▼
                          Chroma
                             │
                     ┌───────┴────────┐
                     │                │
                  Question        Conversation
                     │                │
                     └───────┬────────┘
                             ▼
                    Reference Resolution
                             │
                             ▼
                    Hybrid Retrieval
                     ┌───────┴───────┐
                     │               │
                  Semantic         BM25
                     │               │
                     └───────┬───────┘
                             ▼
                       Hybrid Top-10
                             │
                             ▼
                         ColBERT
                             │
                             ▼
                          Top-5
                             │
                             ▼
                     Context Builder
                             │
                             ▼
                    Groq / Ollama
                             │
                             ▼
                         Answer
                             │
                             ▼
                  Citation Validation
                             │
                             ▼
                   Save Conversation
                             │
                             ▼
                       API Response
```

---


# Github

The complete project link is here: 

# 41. Final Project Summary


The **Meeting RAG** project is a complete end-to-end system that transforms meeting recordings into an interactive, searchable knowledge base using **Faster-Whisper** for transcription, **pyannote** for speaker diarization, transcript analysis and cleaning, metadata-based chunking, **BGE-small + Chroma** for semantic retrieval, **BM25** for lexical retrieval, and **ColBERT** for reranking. At query time, hybrid retrieval combines semantic and BM25 results using a `0.6 / 0.4` weighting, selects and reranks relevant chunks, and provides structured context to **Groq (`openai/gpt-oss-20b`)**, with **Ollama (`qwen3:8b`)** as a fallback. The system supports citation validation and multi-turn conversations with stored questions, answers, retrieval queries, citations, providers, models, and timestamps. A core architectural feature is **strict meeting isolation**, where each meeting receives a unique sequential ID and that ID is enforced throughout storage, retrieval, reranking, context construction, generation, and citation validation to prevent information from different meetings from being mixed.
