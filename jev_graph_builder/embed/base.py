"""EmbeddingProvider port (§8.5). Neither Jev nor the harnesses provide embeddings."""

from __future__ import annotations

from typing import Any, Protocol

from jev_graph_builder.registry.loader import RegistryError


class EmbeddingProvider(Protocol):
    model_id: str
    dim: int
    max_tokens: int

    async def embed(self, texts: list[str]) -> list[list[float]]: ...


def build_embedder(profile: dict[str, Any]) -> EmbeddingProvider:
    kind = profile["kind"]
    if kind == "sentence_transformers":
        from jev_graph_builder.embed.local import SentenceTransformersEmbedder

        return SentenceTransformersEmbedder(profile)
    if kind == "openai_compatible":
        from jev_graph_builder.embed.openai_compat import OpenAICompatibleEmbedder

        return OpenAICompatibleEmbedder(profile)
    if kind == "hashing":
        from jev_graph_builder.embed.hashing import HashingEmbedder

        return HashingEmbedder(profile)
    raise RegistryError(f"unknown embedding provider kind `{kind}`")
