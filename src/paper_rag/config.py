"""Load and validate `config.yaml` into typed, immutable dataclasses.

Plain dataclasses instead of a validation library: the config is small, and unknown
or missing keys still fail early with a clear error message.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class PathsConfig:
    papers_csv: Path
    pdf_dir: Path
    index_dir: Path
    questions: Path
    results_dir: Path


@dataclass(frozen=True)
class EmbeddingConfig:
    model: str
    use_prefixes: bool
    document_prefix: str
    query_prefix: str


@dataclass(frozen=True)
class LLMConfig:
    model: str
    num_ctx: int
    temperature: float
    max_tokens: int


@dataclass(frozen=True)
class ChunkingConfig:
    chunk_size: int
    chunk_overlap: int


@dataclass(frozen=True)
class RetrievalConfig:
    top_k: int


@dataclass(frozen=True)
class Config:
    paths: PathsConfig
    embedding: EmbeddingConfig
    llm: LLMConfig
    chunking: ChunkingConfig
    retrieval: RetrievalConfig
    raw: dict[str, Any]  # original YAML content, stored alongside evaluation results


def _build(cls: type, section: str, values: Any) -> Any:
    """Instantiate a section dataclass, reporting unknown or missing keys by name."""
    if not isinstance(values, dict):
        raise ValueError(f"Config section '{section}' must be a mapping")
    expected = {f.name for f in fields(cls)}
    unknown = set(values) - expected
    missing = expected - set(values)
    if unknown:
        raise ValueError(f"Unknown key(s) in '{section}': {sorted(unknown)}")
    if missing:
        raise ValueError(f"Missing key(s) in '{section}': {sorted(missing)}")
    return cls(**values)


def load_config(path: str | Path) -> Config:
    """Read a YAML config file; relative paths are resolved against the file's directory."""
    path = Path(path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    base = path.resolve().parent

    sections = {
        "paths": PathsConfig,
        "embedding": EmbeddingConfig,
        "llm": LLMConfig,
        "chunking": ChunkingConfig,
        "retrieval": RetrievalConfig,
    }
    unknown = set(raw) - set(sections)
    if unknown:
        raise ValueError(f"Unknown config section(s): {sorted(unknown)}")

    built = {name: _build(cls, name, raw.get(name)) for name, cls in sections.items()}
    built["paths"] = PathsConfig(
        **{f.name: base / getattr(built["paths"], f.name) for f in fields(PathsConfig)}
    )

    chunking: ChunkingConfig = built["chunking"]
    if chunking.chunk_size <= 0:
        raise ValueError("chunking.chunk_size must be positive")
    if not 0 <= chunking.chunk_overlap < chunking.chunk_size:
        raise ValueError("chunking.chunk_overlap must be >= 0 and smaller than chunk_size")
    if built["retrieval"].top_k <= 0:
        raise ValueError("retrieval.top_k must be positive")

    return Config(**built, raw=raw)
