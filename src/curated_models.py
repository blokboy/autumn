"""Curated list of small, ungated GGUF models offered by the first-run
model picker (see screens/model_picker_screen.py) when Autumn's local model
catalog is empty."""

from dataclasses import dataclass


@dataclass(frozen=True)
class CuratedModel:
    """One downloadable option in the first-run model picker. `repo` +
    `filename` resolve to a direct, unauthenticated download URL:
    https://huggingface.co/<repo>/resolve/main/<filename>.

    `sha256` pins the expected checksum of that file so
    `model_downloader.download_and_install` can verify a completed download
    before installing it -- see that module's `_sha256_file` /
    `ChecksumMismatchError`. Values below come from Hugging Face's
    `tree/main` API (`lfs.oid`), which reports the SHA-256 the Git LFS
    pointer for each file was uploaded with -- the same value `git lfs`
    itself checks a fetched blob against -- fetched without downloading the
    multi-GB file bytes locally to hash them. See the source URL noted next
    to each entry to re-verify."""

    name: str
    vendor: str
    repo: str
    filename: str
    approx_size_gb: float
    sha256: str
    context_window: int | None = None


CURATED_MODELS: list[CuratedModel] = [
    CuratedModel(
        name="Llama 3.2 3B Instruct",
        vendor="Meta",
        repo="bartowski/Llama-3.2-3B-Instruct-GGUF",
        filename="Llama-3.2-3B-Instruct-Q4_K_M.gguf",
        approx_size_gb=2.02,
        # https://huggingface.co/api/models/bartowski/Llama-3.2-3B-Instruct-GGUF/tree/main
        sha256="6c1a2b41161032677be168d354123594c0e6e67d2b9227c84f296ad037c728ff",
        context_window=131072,
    ),
    CuratedModel(
        name="Qwen2.5 3B Instruct",
        vendor="Alibaba",
        repo="bartowski/Qwen2.5-3B-Instruct-GGUF",
        filename="Qwen2.5-3B-Instruct-Q4_K_M.gguf",
        approx_size_gb=1.93,
        # https://huggingface.co/api/models/bartowski/Qwen2.5-3B-Instruct-GGUF/tree/main
        sha256="9c9f56a391a3abbd5b89d0245bf6106081bcc3173119d4229235dd9d23253f94",
        context_window=32768,
    ),
    CuratedModel(
        name="Phi-3.5-mini-instruct",
        vendor="Microsoft",
        repo="bartowski/Phi-3.5-mini-instruct-GGUF",
        filename="Phi-3.5-mini-instruct-Q4_K_M.gguf",
        approx_size_gb=2.39,
        # https://huggingface.co/api/models/bartowski/Phi-3.5-mini-instruct-GGUF/tree/main
        sha256="e4165e3a71af97f1b4820da61079826d8752a2088e313af0c7d346796c38eff5",
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
        # https://huggingface.co/api/models/bartowski/gemma-2-2b-it-GGUF/tree/main
        sha256="e0aee85060f168f0f2d8473d7ea41ce2f3230c1bc1374847505ea599288a7787",
        context_window=8192,
    ),
    CuratedModel(
        name="TinyLlama 1.1B Chat",
        vendor="community",
        repo="TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF",
        filename="tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
        approx_size_gb=0.67,
        # https://huggingface.co/api/models/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/tree/main
        sha256="9fecc3b3cd76bba89d504f29b616eedf7da85b96540e490ca5824d3f7d2776a0",
        context_window=2048,
    ),
]
