"""Tiny offline LLM fallback used before external providers are configured."""

from models import ChatMessage, ModelChoice

OFFLINE_TINY_MODEL = "autumn/offline-tiny"


def generate_response(messages: list[ChatMessage], choice: ModelChoice | None = None) -> ChatMessage:
    last_user = next((message for message in reversed(messages) if message.role == "user"), None)
    prompt = last_user.text if last_user is not None else ""
    model = choice.name if choice is not None else OFFLINE_TINY_MODEL
    return ChatMessage(
        role="assistant",
        text=f"Offline local response: {prompt}",
        model=model,
    )
