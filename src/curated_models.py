"""Curated list of small, ungated GGUF models offered by the first-run
model picker (see screens/model_picker_screen.py) when Autumn's local model
catalog is empty."""

from dataclasses import dataclass


@dataclass(frozen=True)
class CuratedModel:
    """One downloadable option in the first-run model picker. `repo` +
    `filename` resolve to a direct, unauthenticated download URL:
    https://huggingface.co/<repo>/resolve/main/<filename>."""

    name: str
    vendor: str
    repo: str
    filename: str
    approx_size_gb: float
    context_window: int | None = None


CURATED_MODELS: list[CuratedModel] = [
    CuratedModel(
        name="Llama 3.2 3B Instruct",
        vendor="Meta",
        repo="bartowski/Llama-3.2-3B-Instruct-GGUF",
        filename="Llama-3.2-3B-Instruct-Q4_K_M.gguf",
        approx_size_gb=2.02,
        context_window=131072,
    ),
    CuratedModel(
        name="Qwen2.5 3B Instruct",
        vendor="Alibaba",
        repo="bartowski/Qwen2.5-3B-Instruct-GGUF",
        filename="Qwen2.5-3B-Instruct-Q4_K_M.gguf",
        approx_size_gb=1.93,
        context_window=32768,
    ),
    CuratedModel(
        name="Phi-3.5-mini-instruct",
        vendor="Microsoft",
        repo="bartowski/Phi-3.5-mini-instruct-GGUF",
        filename="Phi-3.5-mini-instruct-Q4_K_M.gguf",
        approx_size_gb=2.39,
        context_window=131072,
    ),
    CuratedModel(
        # Google's own gemma-2-2b-it-GGUF repo gates downloads behind a
        # license click-through that blocks anonymous HTTPS fetches; this is
        # bartowski's ungated re-quantization of the same weights instead.
        name="Gemma 2 2B Instruct",
        vendor="Google",
        repo="bartowski/gemma-2-2b-it-GGUF",
        filename="gemma-2-2b-it-Q4_K_M.gguf",
        approx_size_gb=1.71,
        context_window=8192,
    ),
    CuratedModel(
        name="TinyLlama 1.1B Chat",
        vendor="community",
        repo="TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF",
        filename="tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
        approx_size_gb=0.67,
        context_window=2048,
    ),
]
