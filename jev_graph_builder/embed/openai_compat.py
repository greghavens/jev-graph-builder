"""Any OpenAI-compatible `/v1/embeddings` endpoint (§8.5)."""

from __future__ import annotations

from typing import Any

import httpx

from jev_graph_builder.config import secret


class OpenAICompatibleEmbedder:
    def __init__(self, profile: dict[str, Any], transport: httpx.AsyncBaseTransport | None = None) -> None:
        self.model_id = profile["model_id"]
        self.dim = profile["dim"]
        self.max_tokens = profile["max_tokens"]
        self.batch_size = profile["batch_size"]
        self.endpoint = profile["endpoint"]
        key = secret(profile["key_env"]) if profile.get("key_env") else None
        headers = {"Authorization": f"Bearer {key}"} if key else {}
        self._http = httpx.AsyncClient(timeout=profile["timeout_s"], headers=headers, transport=transport)
        self._send_dim = profile.get("send_dimensions", False)

    async def embed(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for i in range(0, len(texts), self.batch_size):
            body: dict[str, Any] = {"model": self.model_id, "input": texts[i : i + self.batch_size]}
            if self._send_dim:
                body["dimensions"] = self.dim
            resp = await self._http.post(self.endpoint, json=body)
            resp.raise_for_status()
            data = sorted(resp.json()["data"], key=lambda d: d["index"])
            for d in data:
                if len(d["embedding"]) != self.dim:
                    raise ValueError(f"endpoint returned dim {len(d['embedding'])}, profile says {self.dim}")
                out.append(d["embedding"])
        return out
