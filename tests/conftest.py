from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from jev_graph_builder.registry.loader import Registry

ROOT = Path(__file__).resolve().parents[1]
SEED_REGISTRY = ROOT / "registry"


@pytest.fixture
def reg() -> Registry:
    return Registry(SEED_REGISTRY)


@pytest.fixture
def reg_copy(tmp_path: Path) -> Registry:
    """A writable copy of the seed Registry (versioning / calibration tests)."""
    dst = tmp_path / "registry"
    shutil.copytree(SEED_REGISTRY, dst, ignore=shutil.ignore_patterns(".proposals"))
    return Registry(dst)
