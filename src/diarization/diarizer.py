from pathlib import Path
from typing import Any

import torch
from pyannote.audio import Pipeline


class SpeakerDiarizer:
    def __init__(self, model_name="pyannote/speaker-diarization-community-1"):
        self.model_name = model_name

        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        self.pipeline: Any = Pipeline.from_pretrained(self.model_name)
        self.pipeline.to(self.device)

    def diarize(self, audio_path):
        audio_path = Path(audio_path)

        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        output: Any = self.pipeline(str(audio_path))

        diarization = output.exclusive_speaker_diarization

        segments = []

        for turn, _, speaker in diarization.itertracks(yield_label=True):
            segments.append(
                {
                    "speaker": speaker,
                    "start": round(turn.start, 3),
                    "end": round(turn.end, 3),
                }
            )

        return segments