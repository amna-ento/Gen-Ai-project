from pathlib import Path

import soundfile as sf


SUPPORTED_EXTENSIONS = {
    ".wav",
    ".mp3",
    ".m4a",
    ".flac",
    ".ogg",
    ".aac",
}


def validate_audio_file(audio_path: str) -> dict:
    """
    Validate an audio file before transcription.
    """

    path = Path(audio_path)

    if not path.exists():
        raise FileNotFoundError(f"Audio file does not exist: {path}")

    if not path.is_file():
        raise ValueError(f"Path is not a file: {path}")

    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported audio format: {path.suffix}. "
            f"Supported formats: {sorted(SUPPORTED_EXTENSIONS)}"
        )

    try:
        info = sf.info(str(path))
    except Exception as exc:
        raise ValueError(f"Unable to read audio file: {path}") from exc

    if info.duration <= 0:
        raise ValueError(f"Audio file has no duration: {path}")

    return {
        "path": str(path),
        "filename": path.name,
        "extension": path.suffix.lower(),
        "duration": info.duration,
        "sample_rate": info.samplerate,
        "channels": info.channels,
    }