"""Busca 1:N dos embeddings cadastrados, em memória."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from checkpoint.face import EMBEDDING_SIZE


@dataclass(frozen=True)
class Match:
    employee_id: int
    name: str
    score: float


class Gallery:
    """Embeddings de todos os funcionários numa matriz; a busca é um produto de matrizes."""

    def __init__(self) -> None:
        self._names: dict[int, str] = {}
        self._labels = np.empty(0, dtype=np.int64)
        self._matrix = np.empty((0, EMBEDDING_SIZE), dtype=np.float32)

    def __len__(self) -> int:
        return len(self._names)

    def add(self, employee_id: int, name: str, embeddings: list[np.ndarray]) -> None:
        if not embeddings:
            return
        rows = np.vstack([np.asarray(e, dtype=np.float32).reshape(1, -1) for e in embeddings])
        if rows.shape[1] != EMBEDDING_SIZE:
            raise ValueError(f"embedding com {rows.shape[1]} posições, esperado {EMBEDDING_SIZE}")
        self._names[employee_id] = name
        self._matrix = np.vstack([self._matrix, rows])
        self._labels = np.concatenate([self._labels, np.full(len(rows), employee_id)])

    def scores(self, embedding: np.ndarray) -> list[Match]:
        """Melhor similaridade de cada funcionário, da maior para a menor."""
        if not self._names:
            return []
        sims = self._matrix @ np.asarray(embedding, dtype=np.float32).reshape(-1)
        best: dict[int, float] = {}
        for label, sim in zip(self._labels.tolist(), sims.tolist()):
            if sim > best.get(label, -2.0):
                best[label] = sim
        ranked = sorted(best.items(), key=lambda item: item[1], reverse=True)
        return [Match(emp, self._names[emp], score) for emp, score in ranked]

    def identify(self, embedding: np.ndarray, threshold: float) -> Match | None:
        """Funcionário mais parecido, se passar do limiar; senão None (desconhecido)."""
        ranked = self.scores(embedding)
        if ranked and ranked[0].score >= threshold:
            return ranked[0]
        return None
