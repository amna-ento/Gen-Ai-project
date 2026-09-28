from datetime import datetime, timezone
from typing import Any

from src.conversation.conversation_storage import (
    create_conversation,
    load_conversation,
    save_conversation,
)


class ConversationManager:

    def __init__(
        self,
        meeting_id: str,
        conversation_id: str = "C-001",
    ):

        self.meeting_id = meeting_id
        self.conversation_id = conversation_id

        conversation = load_conversation(
            meeting_id
        )

        if conversation is None:

            conversation = create_conversation(
                meeting_id=meeting_id,
                conversation_id=conversation_id,
            )

            save_conversation(
                conversation
            )

        if conversation.get(
            "meeting_id"
        ) != meeting_id:

            raise ValueError(
                "Conversation belongs to a "
                "different meeting."
            )

        self.conversation = conversation

    def get_turns(
        self,
    ) -> list[dict[str, Any]]:

        return self.conversation.get(
            "turns",
            [],
        )

    def get_history_text(
        self,
    ) -> str:

        turns = self.get_turns()

        if not turns:
            return "No previous conversation."

        history_parts = []

        for turn in turns:

            history_parts.append(
                f"User: {turn['question']}\n"
                f"Assistant: {turn['answer']}"
            )

        return "\n\n".join(
            history_parts
        )

    def add_turn(
        self,
        question: str,
        answer: str,
        citations: list[str],
        generation_provider: str,
        generation_model: str,
        retrieval_question: str | None = None,
    ) -> dict[str, Any]:

        turns = self.get_turns()

        turn_id = len(turns) + 1

        turn = {
            "turn_id": turn_id,
            "question": question,
            "retrieval_question": (
                retrieval_question
                if retrieval_question
                else question
            ),
            "answer": answer,
            "citations": citations,
            "generation_provider": (
                generation_provider
            ),
            "generation_model": (
                generation_model
            ),
            "timestamp": (
                datetime.now(
                    timezone.utc
                ).isoformat()
            ),
        }

        turns.append(turn)

        self.conversation["turns"] = turns

        save_conversation(
            self.conversation
        )

        return turn

    def clear(
        self,
    ) -> None:

        self.conversation = create_conversation(
            meeting_id=self.meeting_id,
            conversation_id=self.conversation_id,
        )

        save_conversation(
            self.conversation
        )