import json

from pathlib import Path
from typing import Any


BASE_PATH = Path(
    "data/meetings/valid_input"
)


def get_conversation_path(
    meeting_id: str,
) -> Path:

    return (
        BASE_PATH
        / meeting_id
        / "conversation"
        / "conversation.json"
    )


def create_conversation(
    meeting_id: str,
    conversation_id: str,
) -> dict[str, Any]:

    return {
        "meeting_id": meeting_id,
        "conversation_id": conversation_id,
        "turns": [],
    }


def save_conversation(
    conversation: dict[str, Any],
) -> None:

    meeting_id = conversation["meeting_id"]

    path = get_conversation_path(
        meeting_id
    )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            conversation,
            file,
            indent=4,
            ensure_ascii=False,
        )


def load_conversation(
    meeting_id: str,
) -> dict[str, Any] | None:

    path = get_conversation_path(
        meeting_id
    )

    if not path.exists():
        return None

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as file:

        conversation = json.load(file)

    if conversation.get(
        "meeting_id"
    ) != meeting_id:

        raise ValueError(
            "Conversation meeting_id does not "
            "match requested meeting_id."
        )

    return conversation