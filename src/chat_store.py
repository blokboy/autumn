"""Crash-safe, per-session persistence for dashboard chat transcripts."""

import json
import os
import uuid
from pathlib import Path

from models import ChatMessage
from procutil import pid_alive


def new_session_id() -> str:
    return uuid.uuid4().hex


def session_path(sessions_root: Path, session_id: str) -> Path:
    return sessions_root / f"{session_id}.json"


def _message_to_dict(message: ChatMessage) -> dict:
    raw = {"role": message.role, "text": message.text}
    if message.model is not None:
        raw["model"] = message.model
    if message.participant_name is not None:
        raw["participant_name"] = message.participant_name
    return raw


def _message_from_dict(raw: object) -> ChatMessage | None:
    if not isinstance(raw, dict):
        return None
    role = raw.get("role")
    text = raw.get("text")
    model = raw.get("model")
    participant_name = raw.get("participant_name")
    if role not in ("user", "assistant"):
        return None
    if not isinstance(text, str) or not text:
        return None
    if model is not None and not isinstance(model, str):
        return None
    if participant_name is not None and (
        not isinstance(participant_name, str) or not participant_name
    ):
        return None
    return ChatMessage(
        role=role,
        text=text,
        model=model,
        participant_name=participant_name,
    )


def persist_chat(path: Path, messages: list[ChatMessage], pid: int | None = None) -> None:
    if not messages:
        path.unlink(missing_ok=True)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "pid": pid if pid is not None else os.getpid(),
        "messages": [_message_to_dict(message) for message in messages],
    }
    path.write_text(json.dumps(payload))


def _load_session(path: Path) -> dict | None:
    try:
        with path.open() as f:
            payload = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None
    if not isinstance(payload, dict):
        return None
    pid = payload.get("pid")
    messages = payload.get("messages")
    if not isinstance(pid, int) or not isinstance(messages, list):
        return None
    return {"pid": pid, "messages": messages}


def load_chat(path: Path) -> list[ChatMessage]:
    session = _load_session(path)
    if session is None:
        return []
    parsed = [_message_from_dict(raw) for raw in session["messages"]]
    return [message for message in parsed if message is not None]


def discover_resumable(sessions_root: Path, own_path: Path) -> list[Path]:
    if not sessions_root.is_dir():
        return []

    resumable = []
    for path in sessions_root.glob("*.json"):
        if path == own_path:
            continue
        session = _load_session(path)
        if session is None:
            continue
        if pid_alive(session["pid"]):
            continue
        parsed = [_message_from_dict(raw) for raw in session["messages"]]
        if not any(message is not None for message in parsed):
            continue
        resumable.append(path)
    resumable.sort(key=lambda p: p.stat().st_mtime)
    return resumable


def load_and_merge(paths: list[Path]) -> list[ChatMessage]:
    merged: list[ChatMessage] = []
    for path in paths:
        merged.extend(load_chat(path))
    return merged


def delete_files(paths: list[Path]) -> None:
    for path in paths:
        path.unlink(missing_ok=True)
