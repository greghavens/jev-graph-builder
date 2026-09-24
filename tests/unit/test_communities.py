import math

from jev_graph_builder.pipeline.communities import partition, usable_edges

POL = {"resolution": 1.0, "seed": 7, "iterations": 10, "min_size": 3, "weight_decimals": 3, "timeout_s": 60}


def _clique(prefix: str, n: int) -> list[dict]:
    names = [f"{prefix}{i}" for i in range(n)]
    return [{"src_id": a, "dst_id": b, "weight": 0.9} for i, a in enumerate(names) for b in names[i + 1:]]


def test_usable_edges_drops_degenerate_weights_and_quantises():
    edges = [
        {"src_id": "a", "dst_id": "b", "weight": None},
        {"src_id": "a", "dst_id": "c", "weight": math.nan},
        {"src_id": "a", "dst_id": "d", "weight": 0.0},
        {"src_id": "a", "dst_id": "e", "weight": 0.0001},  # rounds to zero at 3 decimals
        {"src_id": "a", "dst_id": "f", "weight": 0.12345},
    ]
    assert usable_edges(edges, 3) == [("a", "f", 0.123)]


async def test_partition_finds_two_communities_deterministically():
    edges = _clique("x", 4) + _clique("y", 4) + [{"src_id": "x0", "dst_id": "y0", "weight": 0.1}]
    first = await partition(edges, POL)
    assert first == [["x0", "x1", "x2", "x3"], ["y0", "y1", "y2", "y3"]]
    assert await partition(list(reversed(edges)), POL) == first


async def test_partition_times_out_instead_of_hanging():
    edges = _clique("x", 4)
    assert await partition(edges, {**POL, "timeout_s": 0}) is None


async def test_partition_of_empty_graph():
    assert await partition([{"src_id": "a", "dst_id": "b", "weight": 0}], POL) == []
