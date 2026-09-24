import json
import os
import re
import time
import urllib.error
import urllib.request
from pathlib import Path

from groq import Groq, RateLimitError


class MeetingAnalyzer:
    def __init__(
        self,
        model: str = "openai/gpt-oss-20b",
        max_input_tokens: int = 1200,
        max_segments_per_batch: int = 30,
        overlap_segments: int = 3,
        max_output_tokens: int = 4000,
        ollama_model: str = "qwen3:8b",
        ollama_url: str = "http://127.0.0.1:11434/api/chat",
    ):
        api_key = os.getenv("GROQ_API_KEY")

        self.client = None

        if api_key:
            self.client = Groq(api_key=api_key)

        self.model = model
        self.ollama_model = ollama_model
        self.ollama_url = ollama_url

        self.max_input_tokens = max_input_tokens
        self.max_segments_per_batch = max_segments_per_batch
        self.overlap_segments = overlap_segments
        self.max_output_tokens = max_output_tokens

        self.groq_daily_limit_reached = False

        self.speaker_mapping = {}
        self.canonical_speakers = []
        self.valid_segment_ids = set()

        if self.client:
            print("LLM provider: Groq")
        else:
            print(
                "GROQ_API_KEY not found. "
                "Using Ollama Qwen3 8B."
            )

    def analyze(self, transcript, meeting_id):
        segments = transcript.get("segments", [])

        print(f"Total transcript segments: {len(segments)}")

        self._build_speaker_mapping(segments)

        prepared_segments = self._prepare_segments(segments)

        self.valid_segment_ids = {
            segment["segment_id"]
            for segment in prepared_segments
        }

        batches = self._create_batches(prepared_segments)

        print(f"Total batches: {len(batches)}")

        analysis_dir = Path(
            f"data/meetings/valid_input/{meeting_id}/analysis"
        )
        analysis_dir.mkdir(parents=True, exist_ok=True)

        batch_results = []

        for batch_number, batch in enumerate(batches, start=1):
            checkpoint_path = (
                analysis_dir / f"batch_{batch_number:03d}.json"
            )

            if checkpoint_path.exists():
                print(
                    f"Batch {batch_number}/{len(batches)} "
                    f"already completed. Skipping..."
                )

                with open(
                    checkpoint_path,
                    "r",
                    encoding="utf-8",
                ) as file:
                    saved_result = json.load(file)

                batch_results.append(saved_result)
                continue

            start_segment = batch[0]["segment_id"]
            end_segment = batch[-1]["segment_id"]

            print(
                f"Analyzing batch {batch_number}/{len(batches)} "
                f"(segments {start_segment}-{end_segment})..."
            )

            formatted_segments = self._format_segments(batch)

            prompt = self._build_prompt(
                formatted_segments
            )

            result = self._call_llm(prompt)

            normalized_result = self._normalize_result(result)

            with open(
                checkpoint_path,
                "w",
                encoding="utf-8",
            ) as file:
                json.dump(
                    normalized_result,
                    file,
                    indent=2,
                    ensure_ascii=False,
                )

            print(
                f"Batch {batch_number}/{len(batches)} saved."
            )

            batch_results.append(normalized_result)

        final_result = self._merge_results(batch_results)

        final_result = self._normalize_result(final_result)

        final_path = analysis_dir / "final_analysis.json"

        with open(
            final_path,
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                final_result,
                file,
                indent=2,
                ensure_ascii=False,
            )

        print(f"\nFinal analysis saved to: {final_path}")

        return final_result

    def _build_speaker_mapping(self, segments):
        raw_speakers = []

        for segment in segments:
            speaker = segment.get("speaker")

            if speaker:
                raw_speakers.append(str(speaker))

        unique_speakers = sorted(set(raw_speakers))

        mapping = {}

        for speaker in unique_speakers:
            mapping[speaker] = self._canonicalize_speaker(speaker)

        self.speaker_mapping = mapping

        self.canonical_speakers = sorted(
            set(mapping.values())
        )

    def _canonicalize_speaker(self, speaker):
        if speaker is None:
            return None

        speaker = str(speaker).strip()

        match = re.fullmatch(
            r"speaker[\s_-]*(\d+)",
            speaker,
            flags=re.IGNORECASE,
        )

        if match:
            number = int(match.group(1))
            return f"SPEAKER_{number:02d}"

        if speaker.upper() == "UNKNOWN":
            return "UNKNOWN"

        return speaker

    def _prepare_segments(self, segments):
        prepared = []

        for index, segment in enumerate(segments, start=1):
            text = str(segment.get("text", "")).strip()

            if not text:
                continue

            raw_speaker = segment.get("speaker")

            speaker = self._normalize_speaker_name(
                raw_speaker
            )

            prepared.append(
                {
                    "segment_id": index,
                    "speaker": speaker,
                    "text": text,
                }
            )

        return prepared

    def _create_batches(self, segments):
        batches = []

        current_batch = []
        current_tokens = 0

        start_index = 0

        while start_index < len(segments):
            current_batch = []
            current_tokens = 0

            index = start_index

            while index < len(segments):
                segment = segments[index]

                estimated_tokens = max(
                    1,
                    len(segment["text"]) // 4,
                )

                if (
                    current_batch
                    and (
                        len(current_batch)
                        >= self.max_segments_per_batch
                        or current_tokens
                        + estimated_tokens
                        > self.max_input_tokens
                    )
                ):
                    break

                current_batch.append(segment)
                current_tokens += estimated_tokens

                index += 1

            batches.append(current_batch)

            if index >= len(segments):
                break

            next_start = index - self.overlap_segments

            if next_start <= start_index:
                next_start = index

            start_index = next_start

        return batches

    def _format_segments(self, segments):
        lines = []

        for segment in segments:
            lines.append(
                f"[Segment {segment['segment_id']}] "
                f"{segment['speaker']}: "
                f"{segment['text']}"
            )

        return "\n".join(lines)

    def _build_prompt(self, formatted_segments):
        speaker_list = ", ".join(
            self.canonical_speakers
        )

        return f"""
Analyze the following meeting transcript segments.

VALID SPEAKER NAMES:
{speaker_list}

Use ONLY these exact speaker names.

Do not create variations such as:
- Speaker 3
- Speaker_03
- speaker 3
- SPEAKER 3

Use:
SPEAKER_03

Return ONLY valid JSON.

Required structure:

{{
  "topics": [
    {{
      "topic": "short topic name",
      "source_segments": [1, 2]
    }}
  ],
  "sentiment": [
    {{
      "speaker": "SPEAKER_00",
      "label": "positive",
      "source_segments": [1, 2]
    }}
  ],
  "decisions": [
    {{
      "decision": "explicit agreed or finalized outcome",
      "made_by": ["SPEAKER_00"],
      "source_segments": [1, 2]
    }}
  ],
  "action_items": [
    {{
      "assigned_by": "SPEAKER_00",
      "assigned_to": "SPEAKER_01",
      "task": "task description",
      "deadline": null,
      "source_segments": [1, 2]
    }}
  ]
}}

RULES:

TOPICS:
- Extract important discussion topics.
- Maximum 5 topics from this batch.
- Do not invent topics.

SENTIMENT:
- Use ONLY:
  positive
  negative
  neutral
  mixed
- Use only a few representative source segments per speaker.
- Maximum 5 source segments per speaker.
- Use the exact speaker names provided above.

DECISIONS:
- A decision must represent an explicit agreed, selected,
  approved, rejected, finalized, or committed outcome.
- Normal statements are NOT decisions.
- Opinions are NOT decisions.
- Suggestions are NOT necessarily decisions.
- Questions are NOT decisions.
- Explanations are NOT decisions.
- Ordinary instructions are NOT necessarily decisions.
- Hypothetical possibilities are NOT decisions.
- "made_by" must ALWAYS be a list.
- Maximum 5 decisions.

ACTION ITEMS:
- Include actual tasks or follow-up work.
- Use exact speaker names when the assignee is a diarized speaker.
- Maximum 10 action items.

SOURCE SEGMENTS:
- Use only segment IDs that actually appear in the input.

TRANSCRIPT:
{formatted_segments}
"""

    def _call_llm(self, prompt):
        if self.groq_daily_limit_reached:
            return self._call_ollama(prompt)

        if self.client is None:
            return self._call_ollama(prompt)

        try:
            return self._call_groq(prompt)

        except RateLimitError as error:
            if self._is_daily_token_limit(error):
                self.groq_daily_limit_reached = True

                print(
                    "\nGroq daily token limit reached."
                )
                print(
                    "Switching to Ollama Qwen3 8B..."
                )

                return self._call_ollama(prompt)

            raise

    def _call_groq(self, prompt):
        client = self.client

        if client is None:
            raise RuntimeError(
                "Groq client is not configured."
            )

        max_retries = 5
        retry_delay = 10

        for attempt in range(max_retries):
            try:
                response = client.chat.completions.create(
                    model=self.model,
                    temperature=0,
                    max_tokens=self.max_output_tokens,
                    response_format={
                        "type": "json_object"
                    },
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "You are a meeting analysis "
                                "assistant. Return only valid JSON."
                            ),
                        },
                        {
                            "role": "user",
                            "content": prompt,
                        },
                    ],
                )

                choice = response.choices[0]

                content = choice.message.content

                if choice.finish_reason == "length":
                    raise ValueError(
                        "Groq response was truncated because "
                        "the maximum output token limit was reached."
                    )

                if not content:
                    raise ValueError(
                        "Groq returned an empty response."
                    )

                try:
                    return self._parse_json(content)

                except json.JSONDecodeError as error:
                    print("Groq returned invalid JSON.")
                    print(f"Response: {content}")

                    raise ValueError(
                        "Groq response was not valid JSON."
                    ) from error

            except RateLimitError as error:
                if self._is_daily_token_limit(error):
                    raise

                if attempt == max_retries - 1:
                    raise

                wait_time = retry_delay * (attempt + 1)

                print(
                    "Groq rate limit reached. "
                    f"Retrying in {wait_time} seconds..."
                )

                time.sleep(wait_time)

    def _call_ollama(self, prompt):
        print(
            f"Using Ollama model: {self.ollama_model}"
        )

        payload = {
            "model": self.ollama_model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a meeting analysis "
                        "assistant. Return only valid JSON."
                    ),
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            "stream": False,
            "format": "json",
            "think": False,
            "options": {
                "temperature": 0,
                "num_predict": self.max_output_tokens,
            },
        }

        request = urllib.request.Request(
            self.ollama_url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(
                request,
                timeout=300,
            ) as response:
                response_data = json.loads(
                    response.read().decode("utf-8")
                )

        except urllib.error.URLError as error:
            raise RuntimeError(
                "Ollama is not available. "
                "Make sure the Ollama server is running."
            ) from error

        message = response_data.get("message", {})

        content = message.get("content")

        if not content:
            raise ValueError(
                "Ollama returned an empty response."
            )

        try:
            return self._parse_json(content)

        except json.JSONDecodeError as error:
            print("Ollama returned invalid JSON.")
            print(f"Response: {content}")

            raise ValueError(
                "Ollama response was not valid JSON."
            ) from error

    def _parse_json(self, content):
        content = str(content).strip()

        if content.startswith("```"):
            content = re.sub(
                r"^```(?:json)?\s*",
                "",
                content,
                flags=re.IGNORECASE,
            )

            content = re.sub(
                r"\s*```$",
                "",
                content,
            )

        return json.loads(content)

    def _is_daily_token_limit(self, error):
        message = str(error).lower()

        daily_indicators = [
            "tokens per day",
            "token per day",
            "tpd",
            "daily token",
            "daily limit",
        ]

        return any(
            indicator in message
            for indicator in daily_indicators
        )

    def _normalize_result(self, result):
        normalized = {
            "topics": [],
            "sentiment": [],
            "decisions": [],
            "action_items": [],
        }

        for topic in result.get("topics", []):
            normalized_topic = self._normalize_topic(topic)

            if normalized_topic:
                normalized["topics"].append(
                    normalized_topic
                )

        for sentiment in result.get("sentiment", []):
            normalized_sentiment = (
                self._normalize_sentiment(sentiment)
            )

            if normalized_sentiment:
                normalized["sentiment"].append(
                    normalized_sentiment
                )

        for decision in result.get("decisions", []):
            normalized_decision = (
                self._normalize_decision(decision)
            )

            if normalized_decision:
                normalized["decisions"].append(
                    normalized_decision
                )

        for action_item in result.get(
            "action_items",
            [],
        ):
            normalized_action = (
                self._normalize_action_item(action_item)
            )

            if normalized_action:
                normalized["action_items"].append(
                    normalized_action
                )

        return normalized

    def _normalize_topic(self, topic):
        if not isinstance(topic, dict):
            return None

        text = str(
            topic.get("topic", "")
        ).strip()

        if not text:
            return None

        source_segments = self._validate_source_segments(
            topic.get("source_segments", [])
        )

        return {
            "topic": text,
            "source_segments": source_segments,
        }

    def _normalize_sentiment(self, sentiment):
        if not isinstance(sentiment, dict):
            return None

        speaker = self._normalize_speaker_name(
            sentiment.get("speaker")
        )

        if not speaker:
            return None

        label = sentiment.get("label")

        if not label:
            label = sentiment.get("sentiment")

        label = str(
            label or "neutral"
        ).lower().strip()

        allowed_labels = {
            "positive",
            "negative",
            "neutral",
            "mixed",
        }

        if label not in allowed_labels:
            label = "neutral"

        source_segments = self._validate_source_segments(
            sentiment.get("source_segments", []),
            max_items=5,
        )

        return {
            "speaker": speaker,
            "label": label,
            "source_segments": source_segments,
        }

    def _normalize_decision(self, decision):
        if not isinstance(decision, dict):
            return None

        text = str(
            decision.get("decision", "")
        ).strip()

        if not text:
            return None

        made_by = decision.get("made_by", [])

        if made_by is None:
            made_by = []

        if not isinstance(made_by, list):
            made_by = [made_by]

        normalized_made_by = []

        for speaker in made_by:
            normalized_speaker = (
                self._normalize_speaker_name(speaker)
            )

            if normalized_speaker:
                if normalized_speaker not in normalized_made_by:
                    normalized_made_by.append(
                        normalized_speaker
                    )

        source_segments = self._validate_source_segments(
            decision.get("source_segments", [])
        )

        return {
            "decision": text,
            "made_by": normalized_made_by,
            "source_segments": source_segments,
        }

    def _normalize_action_item(self, action_item):
        if not isinstance(action_item, dict):
            return None

        task = str(
            action_item.get("task", "")
        ).strip()

        if not task:
            return None

        assigned_by = (
            self._normalize_speaker_name(
                action_item.get("assigned_by")
            )
        )

        assigned_to = action_item.get(
            "assigned_to"
        )

        if assigned_to:
            assigned_to = (
                self._normalize_speaker_name(
                    assigned_to
                )
            )

        source_segments = self._validate_source_segments(
            action_item.get("source_segments", [])
        )

        return {
            "assigned_by": assigned_by,
            "assigned_to": assigned_to,
            "task": task,
            "deadline": action_item.get("deadline"),
            "source_segments": source_segments,
        }

    def _normalize_speaker_name(self, speaker):
        if speaker is None:
            return None

        speaker = str(speaker).strip()

        if speaker in self.speaker_mapping:
            return self.speaker_mapping[speaker]

        return self._canonicalize_speaker(speaker)

    def _validate_source_segments(
        self,
        source_segments,
        max_items=None,
    ):
        if source_segments is None:
            return []

        if not isinstance(source_segments, list):
            source_segments = [source_segments]

        valid = set()

        for segment_id in source_segments:
            try:
                segment_id = int(segment_id)
            except (TypeError, ValueError):
                continue

            if segment_id in self.valid_segment_ids:
                valid.add(segment_id)

        result = sorted(valid)

        if max_items is not None:
            result = result[:max_items]

        return result

    def _merge_results(self, results):
        merged = {
            "topics": [],
            "sentiment": [],
            "decisions": [],
            "action_items": [],
        }

        for result in results:
            for topic in result.get("topics", []):
                self._merge_topic(
                    merged["topics"],
                    topic,
                )

            for sentiment in result.get(
                "sentiment",
                [],
            ):
                self._merge_sentiment(
                    merged["sentiment"],
                    sentiment,
                )

            for decision in result.get(
                "decisions",
                [],
            ):
                self._merge_decision(
                    merged["decisions"],
                    decision,
                )

            for action_item in result.get(
                "action_items",
                [],
            ):
                self._merge_action_item(
                    merged["action_items"],
                    action_item,
                )

        return merged

    def _merge_topic(self, topics, new_topic):
        new_text = self._normalize_text(
            new_topic["topic"]
        )

        for existing in topics:
            existing_text = self._normalize_text(
                existing["topic"]
            )

            if existing_text == new_text:
                existing["source_segments"] = sorted(
                    set(
                        existing["source_segments"]
                        + new_topic["source_segments"]
                    )
                )
                return

        topics.append(new_topic)

    def _merge_sentiment(
        self,
        sentiments,
        new_sentiment,
    ):
        speaker = self._normalize_speaker_name(
            new_sentiment["speaker"]
        )

        for existing in sentiments:
            if existing["speaker"] == speaker:
                existing["source_segments"] = sorted(
                    set(
                        existing["source_segments"]
                        + new_sentiment["source_segments"]
                    )
                )[:5]

                return

        sentiments.append(
            {
                "speaker": speaker,
                "label": new_sentiment["label"],
                "source_segments": new_sentiment[
                    "source_segments"
                ][:5],
            }
        )

    def _merge_decision(
        self,
        decisions,
        new_decision,
    ):
        new_text = self._normalize_text(
            new_decision["decision"]
        )

        for existing in decisions:
            existing_text = self._normalize_text(
                existing["decision"]
            )

            if existing_text == new_text:
                existing["source_segments"] = sorted(
                    set(
                        existing["source_segments"]
                        + new_decision["source_segments"]
                    )
                )

                existing["made_by"] = sorted(
                    set(
                        existing["made_by"]
                        + new_decision["made_by"]
                    )
                )

                return

        decisions.append(new_decision)

    def _merge_action_item(
        self,
        action_items,
        new_action_item,
    ):
        new_text = self._normalize_text(
            new_action_item["task"]
        )

        for existing in action_items:
            existing_text = self._normalize_text(
                existing["task"]
            )

            if existing_text == new_text:
                existing["source_segments"] = sorted(
                    set(
                        existing["source_segments"]
                        + new_action_item["source_segments"]
                    )
                )

                return

        action_items.append(new_action_item)

    def _normalize_text(self, text):
        return re.sub(
            r"\s+",
            " ",
            str(text).lower().strip(),
        )


if __name__ == "__main__":
    print("MeetingAnalyzer module loaded.")