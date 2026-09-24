from collections import defaultdict


class SpeakerAligner:
    def align(self, transcript_segments, speaker_segments):
        aligned_segments = []

        for transcript_segment in transcript_segments:
            speaker = self._find_dominant_speaker(
                transcript_segment["start"],
                transcript_segment["end"],
                speaker_segments,
            )

            aligned_segments.append(
                {
                    "speaker": speaker,
                    "start": transcript_segment["start"],
                    "end": transcript_segment["end"],
                    "text": transcript_segment["text"],
                }
            )

        return aligned_segments

    def _find_dominant_speaker(self, start, end, speaker_segments):
        speaker_overlap = defaultdict(float)

        for segment in speaker_segments:
            overlap = self._calculate_overlap(
                start,
                end,
                segment["start"],
                segment["end"],
            )

            if overlap > 0:
                speaker_overlap[segment["speaker"]] += overlap

        if not speaker_overlap:
            return "UNKNOWN"

        return max(
            speaker_overlap,
            key=lambda speaker: speaker_overlap[speaker],
        )

    def _calculate_overlap(self, start_a, end_a, start_b, end_b):
        overlap_start = max(start_a, start_b)
        overlap_end = min(end_a, end_b)

        return max(0.0, overlap_end - overlap_start)