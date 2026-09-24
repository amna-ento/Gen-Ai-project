from collections import defaultdict


class MeetingChunker:
    def __init__(
        self,
        target_words=160,
        max_words=220,
        min_words=60,
        max_duration=90,
    ):
        self.target_words = target_words
        self.max_words = max_words
        self.min_words = min_words
        self.max_duration = max_duration

    def build_topic_map(self, topics):
        topic_map = defaultdict(list)

        for topic in topics:
            topic_name = topic.get("topic")
            source_segments = topic.get("source_segments", [])

            if not topic_name:
                continue

            for segment_id in source_segments:
                if topic_name not in topic_map[segment_id]:
                    topic_map[segment_id].append(topic_name)

        return topic_map

    def build_sentiment_map(self, sentiments):
        sentiment_map = defaultdict(list)

        for sentiment in sentiments:
            speaker = sentiment.get("speaker")
            label = sentiment.get("label")
            source_segments = sentiment.get("source_segments", [])

            if not speaker or not label:
                continue

            for segment_id in source_segments:
                item = {
                    "speaker": speaker,
                    "label": label,
                }

                if item not in sentiment_map[segment_id]:
                    sentiment_map[segment_id].append(item)

        return sentiment_map

    def count_words(self, segments):
        return sum(
            len(segment["text"].split())
            for segment in segments
        )

    def calculate_duration(self, segments):
        if not segments:
            return 0

        return (
            segments[-1]["end"]
            - segments[0]["start"]
        )

    def is_natural_boundary(
        self,
        current_segment,
        next_segment,
        topic_map,
    ):
        current_text = current_segment["text"].strip()

        punctuation_boundary = current_text.endswith(
            (".", "?", "!")
        )

        speaker_change = (
            current_segment["speaker"]
            != next_segment["speaker"]
        )

        current_topics = set(
            topic_map.get(
                current_segment["segment_id"],
                [],
            )
        )

        next_topics = set(
            topic_map.get(
                next_segment["segment_id"],
                [],
            )
        )

        topic_change = (
            bool(current_topics)
            and bool(next_topics)
            and current_topics != next_topics
        )

        return (
            punctuation_boundary
            or speaker_change
            or topic_change
        )

    def build_chunk(
        self,
        meeting_id,
        chunk_number,
        segments,
        topic_map,
        sentiment_map,
    ):
        source_segments = [
            segment["segment_id"]
            for segment in segments
        ]

        speakers = []

        for segment in segments:
            if segment["speaker"] not in speakers:
                speakers.append(segment["speaker"])

        text = "\n".join(
            f'{segment["speaker"]}: '
            f'{segment["text"].strip()}'
            for segment in segments
        )

        topics = []

        for segment_id in source_segments:
            for topic in topic_map.get(
                segment_id,
                [],
            ):
                if topic not in topics:
                    topics.append(topic)

        sentiment_by_speaker = {}

        for segment_id in source_segments:
            for sentiment in sentiment_map.get(
                segment_id,
                [],
            ):
                sentiment_by_speaker[
                    sentiment["speaker"]
                ] = sentiment["label"]

        sentiment = [
            {
                "speaker": speaker,
                "label": label,
            }
            for speaker, label
            in sentiment_by_speaker.items()
        ]

        return {
            "meeting_id": meeting_id,
            "chunk_id": (
                f"{meeting_id}_CHUNK_"
                f"{chunk_number:03d}"
            ),
            "source_segments": source_segments,
            "speakers": speakers,
            "start_time": segments[0]["start"],
            "end_time": segments[-1]["end"],
            "text": text,
            "topic": topics,
            "sentiment": sentiment,
            "word_count": self.count_words(segments),
            "segment_count": len(segments),
        }

    def create_chunks(
        self,
        meeting_id,
        transcript,
        analysis,
    ):
        topic_map = self.build_topic_map(
            analysis.get("topics", [])
        )

        sentiment_map = self.build_sentiment_map(
            analysis.get("sentiment", [])
        )

        segments = []

        for index, segment in enumerate(
            transcript,
            start=1,
        ):
            segments.append(
                {
                    "segment_id": index,
                    "speaker": segment["speaker"],
                    "start": segment["start"],
                    "end": segment["end"],
                    "text": segment["text"],
                }
            )

        chunks = []
        current_chunk = []

        for index, segment in enumerate(segments):
            if not current_chunk:
                current_chunk.append(segment)
                continue

            current_words = self.count_words(
                current_chunk
            )

            current_duration = self.calculate_duration(
                current_chunk
            )

            next_words = len(
                segment["text"].split()
            )

            next_duration = (
                segment["end"]
                - current_chunk[0]["start"]
            )

            would_exceed_words = (
                current_words + next_words
                > self.max_words
            )

            would_exceed_duration = (
                next_duration
                > self.max_duration
            )

            natural_boundary = self.is_natural_boundary(
                current_chunk[-1],
                segment,
                topic_map,
            )

            target_reached = (
                current_words
                >= self.target_words
            )

            minimum_reached = (
                current_words
                >= self.min_words
            )

            should_split = False

            if minimum_reached and (
                would_exceed_words
                or would_exceed_duration
            ):
                should_split = True

            elif (
                minimum_reached
                and target_reached
                and natural_boundary
            ):
                should_split = True

            if should_split:
                chunks.append(
                    self.build_chunk(
                        meeting_id,
                        len(chunks) + 1,
                        current_chunk,
                        topic_map,
                        sentiment_map,
                    )
                )

                current_chunk = [segment]

            else:
                current_chunk.append(segment)

        if current_chunk:
            chunks.append(
                self.build_chunk(
                    meeting_id,
                    len(chunks) + 1,
                    current_chunk,
                    topic_map,
                    sentiment_map,
                )
            )

        return chunks