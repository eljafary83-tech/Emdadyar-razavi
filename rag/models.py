"""Domain models shared by ingestion, retrieval and the assistant service."""
from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class Chunk:
    id: str
    text: str
    source_file: str
    page_number: int
    chunk_index: int

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> "Chunk":
        return cls(**value)  # type: ignore[arg-type]


@dataclass(frozen=True)
class SearchResult:
    chunk: Chunk
    score: float
