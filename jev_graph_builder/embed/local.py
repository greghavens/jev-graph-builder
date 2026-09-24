"""Local `sentence-transformers` embeddings (the §21 Q1 default)."""

from __future__ import annotations

import asyncio
from typing import Any


class SentenceTransformersEmbedder:
    def __init__(self, profile: dict[str, Any]) -> None:
        from sentence_transformers import SentenceTransformer

        self.model_id = profile["model_id"]
        self.batch_size = profile["batch_size"]
        self.normalize = profile["normalize"]
        self._model = SentenceTransformer(self.model_id, device=profile.get("device"))
        self.dim = int(self._model.get_sentence_embedding_dimension())
        if self.dim != profile["dim"]:
            raise ValueError(f"profile says dim {profile['dim']} but {self.model_id} produces {self.dim}")
        self.max_tokens = min(profile["max_tokens"], int(self._model.max_seq_length))

    async def embed(self, texts: list[str]) -> list[list[float]]:
        def _run() -> list[list[float]]:
            vecs = self._model.encode(texts, batch_size=self.batch_size, normalize_embeddings=self.normalize, show_progress_bar=False)
            return [v.tolist() for v in vecs]

        return await asyncio.to_thread(_run)
