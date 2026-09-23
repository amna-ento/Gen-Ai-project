from faster_whisper import WhisperModel


class MeetingTranscriber:
    def __init__(
        self,
        model_size: str = "small",
        device: str = "cpu",
        compute_type: str = "int8",
    ):
        self.model = WhisperModel(
            model_size,
            device=device,
            compute_type=compute_type,
        )

    def transcribe(self, audio_path: str, language: str | None = None) -> dict:
        segments, info = self.model.transcribe(
            audio_path,
            language=language,
            vad_filter=True,
        )

        transcript_segments = []

        for segment in segments:
            text = segment.text.strip()

            if not text:
                continue

            transcript_segments.append(
                {
                    "start": round(segment.start, 3),
                    "end": round(segment.end, 3),
                    "text": text,
                }
            )

        return {
            "language": info.language,
            "language_probability": round(info.language_probability, 4),
            "duration": round(info.duration, 3),
            "segments": transcript_segments,
        }