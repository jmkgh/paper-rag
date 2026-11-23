from pathlib import Path

import pytest
import yaml

from paper_rag.config import load_config

REPO_ROOT = Path(__file__).resolve().parents[1]


def _write(tmp_path: Path, data: dict) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


@pytest.fixture
def raw_config() -> dict:
    return yaml.safe_load((REPO_ROOT / "config.yaml").read_text(encoding="utf-8"))


def test_repo_config_loads():
    cfg = load_config(REPO_ROOT / "config.yaml")
    assert cfg.llm.model.startswith("ollama")
    assert cfg.chunking.chunk_overlap < cfg.chunking.chunk_size
    assert cfg.paths.pdf_dir == REPO_ROOT / "data" / "pdfs"


def test_relative_paths_resolve_against_config_dir(tmp_path, raw_config):
    cfg = load_config(_write(tmp_path, raw_config))
    assert cfg.paths.index_dir == tmp_path / "data" / "index"


def test_unknown_key_is_rejected(tmp_path, raw_config):
    raw_config["retrieval"]["topk"] = 3
    with pytest.raises(ValueError, match="Unknown key"):
        load_config(_write(tmp_path, raw_config))


def test_missing_section_is_rejected(tmp_path, raw_config):
    del raw_config["llm"]
    with pytest.raises(ValueError, match="'llm'"):
        load_config(_write(tmp_path, raw_config))


def test_overlap_must_be_smaller_than_chunk_size(tmp_path, raw_config):
    raw_config["chunking"]["chunk_overlap"] = raw_config["chunking"]["chunk_size"]
    with pytest.raises(ValueError, match="chunk_overlap"):
        load_config(_write(tmp_path, raw_config))
