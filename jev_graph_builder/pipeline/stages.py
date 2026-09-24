"""Stage order for `run --all` and `plan` (S1..S8; S0 bootstrap runs on its own)."""

from __future__ import annotations

from jev_graph_builder.pipeline.common import Stage
from jev_graph_builder.pipeline.s1_ingest import IngestStage
from jev_graph_builder.pipeline.s2_segment import SegmentStage
from jev_graph_builder.pipeline.s3_enrich import EnrichStage
from jev_graph_builder.pipeline.s4_embed import EmbedStage
from jev_graph_builder.pipeline.s5_entities import EntityStage
from jev_graph_builder.pipeline.s6_links import LinkStage
from jev_graph_builder.pipeline.s7_audit import AuditStage
from jev_graph_builder.pipeline.s8_training import DerivedTrainingStage, TrainingStage

GRAPH_ORDER: tuple[type[Stage], ...] = (IngestStage, SegmentStage, EnrichStage, EmbedStage, EntityStage, LinkStage, AuditStage)
ORDER: tuple[type[Stage], ...] = (*GRAPH_ORDER, TrainingStage, DerivedTrainingStage)


def graph_stages() -> list[Stage]:
    """S1..S7: everything that builds and verifies the graph (S8 is the Q&A spec's)."""
    return [cls() for cls in GRAPH_ORDER]


def all_stages() -> list[Stage]:
    return [cls() for cls in ORDER]


def graph_stage_by_name(name: str) -> Stage:
    for cls in GRAPH_ORDER:
        if cls.name == name:
            return cls()
    raise KeyError(f"unknown stage {name}; graph stages: {', '.join(c.name for c in GRAPH_ORDER)}")


def stage_by_name(name: str) -> Stage:
    for cls in ORDER:
        if cls.name == name:
            return cls()
    raise KeyError(f"unknown stage `{name}`; one of {[c.name for c in ORDER]}")
