import re


class TranscriptCleaner:

    def clean_text(self, text):
        if not text:
            return ""

        text = self._remove_non_speech_artifacts(text)
        text = self._normalize_whitespace(text)
        text = self._remove_repeated_words(text)
        text = self._normalize_punctuation(text)

        return text.strip()

    def _remove_non_speech_artifacts(self, text):
        text = re.sub(
            r"\[(?:music|noise|silence|inaudible)\]",
            "",
            text,
            flags=re.IGNORECASE
        )

        text = re.sub(
            r"\((?:music|noise|silence|inaudible)\)",
            "",
            text,
            flags=re.IGNORECASE
        )

        return text

    def _normalize_whitespace(self, text):
        return re.sub(r"\s+", " ", text)

    def _remove_repeated_words(self, text):
        pattern = r"\b(\w+)(?:\s+\1\b)+"
        return re.sub(pattern, r"\1", text, flags=re.IGNORECASE)

    def _normalize_punctuation(self, text):
        text = re.sub(r"\s+([,.!?;:])", r"\1", text)

        return text
    
    
    