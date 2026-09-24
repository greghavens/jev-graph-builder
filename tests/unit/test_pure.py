"""Pure helpers: tokens, limiter, grounding, segmentation DP, union-find guardrail, fusion, splits."""
from __future__ import annotations

import asyncio
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest

from jev_graph_builder.jev.gating import ACCEPT, REJECT
from jev_graph_builder.jev.limiter import DualLimiter, LimiterSettings
from jev_graph_builder.jev.tokens import TokenEstimator
from jev_graph_builder.jev.service import Decision
from jev_graph_builder.parse.parsers import parse_pdf
from jev_graph_builder.pipeline.evolution import evolution_due, jev_accepted, significantly_above
from jev_graph_builder.pipeline.ontology_checks import CheckReport, check_definitions
from jev_graph_builder.pipeline.grounding import EXACT, FUZZY, NORMALIZED, UNGROUNDED, ground
from jev_graph_builder.pipeline.segment_spans import runs, split_gaps, weakest_gap
from jev_graph_builder.pipeline.unionfind import JEV_SAID_DIFFERENT, merge
from jev_graph_builder.pipeline.s2_segment import SegmentStage, jev_gaps, keep_strength, named_values, section_groups, starts_section
from jev_graph_builder.pipeline.s5_entities import identical_pairs, name_key_pairs
from jev_graph_builder.pipeline.s8_training import assign_split
from jev_graph_builder.query.search import composite, rrf_fuse

FOLDS = {"’": "'"}


def test_truncation_rules():
    est = TokenEstimator(1)
    s = "abcdefghij"
    assert est.truncate(s, 4, "head") == "abcd"
    assert est.truncate(s, 4, "tail") == "ghij"
    assert est.truncate(s, 4, "head_tail").startswith("ab") and est.truncate(s, 4, "head_tail").endswith("ij")
    assert est.truncate(s, 100, "head") == s


def test_limiter_backoff_floor_and_recovery():
    now = [0.0]
    slept = []

    async def sleep(d):
        slept.append(d)
        now[0] += d

    lim = DualLimiter(LimiterSettings(60, 100, 0.5, 0.2, 10, 2), clock=lambda: now[0], sleep=sleep)
    for _ in range(5):
        lim.throttled()
    assert lim.scale == 0.2                      # floor
    now[0] += 11
    asyncio.run(lim.acquire(10))
    assert abs(lim.scale - 0.4) < 1e-9             # one recovery step (× recover_factor)
    asyncio.run(lim.acquire(10))
    assert slept, "second request within the reduced rate must wait"


def test_grounding_levels():
    text = "The gateway retries twice. It then gives up.\nThe gateway’s cache expires hourly."
    assert ground("retries twice", text, 90, FOLDS).kind == EXACT
    g = ground("the  GATEWAY's cache", text, 90, FOLDS)
    assert g.kind == NORMALIZED and text[g.start:g.end].lower().startswith("the gateway")
    assert ground("gateway retries twise", text, 85, FOLDS).kind == FUZZY
    assert ground("database migrations", text, 90, FOLDS).kind == UNGROUNDED
    amb = ground("The gateway", text, 90, FOLDS)
    assert amb.ambiguous and len(amb.candidates) == 2


def test_chunks_are_exactly_jevs_boundaries():
    assert runs(5, [False, True, False, False]) == [(0, 2), (2, 5)]
    assert runs(3, [False, False]) == [(0, 3)]
    assert runs(1, []) == [(0, 1)]
    assert runs(0, []) == []


def test_split_gaps_only_offers_cuts_whose_left_side_fits():
    tokens = [10] * 12
    assert split_gaps(tokens, 0, 12, 35) == [0, 1, 2]          # left sides of 10, 20, 30 tokens
    assert split_gaps(tokens, 4, 12, 20) == [4, 5]
    assert split_gaps([50, 10, 10], 0, 3, 20) == [0]           # one unit over the cap: still progress


def test_union_find_guardrail():
    nodes = ["a", "b", "c", "d"]
    known = {frozenset(("a", "b")): True, frozenset(("b", "c")): True, frozenset(("a", "c")): False}
    out = merge(nodes, [("a", "b", 0.99), ("b", "c", 0.95)], known)
    assert ("b", "c", JEV_SAID_DIFFERENT) in out.refused      # Jev said a≠c: the transitive merge is refused
    assert out.find("b") == "a" and out.find("c") == "c"
    out2 = merge(nodes, [("a", "b", 0.99), ("c", "d", 0.98), ("b", "c", 0.9)], {})
    assert not out2.refused and len({out2.find(n) for n in nodes}) == 1   # Jev never said any two differ: one entity


def test_identical_names_merge_by_code_first():
    rules = {"casefold": True, "strip_punctuation": False}
    ents = {"e1": {"surface": "vCenter", "entity_type": "product"}, "e2": {"surface": "VCENTER", "entity_type": "product"},
            "e3": {"surface": "vCenter", "entity_type": "feature"}, "e4": {"surface": "vSAN", "entity_type": "product"}}
    same = identical_pairs(ents, rules)
    assert same == [("e1", "e2")]                     # same type and normalized name; e3 differs in type
    out = merge(sorted(ents), [("e2", "e4", 0.9)], {}, same)
    assert out.find("e1") == out.find("e2") == out.find("e4") and out.find("e3") == "e3"


def _u(kind, text="t"):
    return {"kind": kind, "text": text, "heading_path": []}


def test_heading_after_content_is_a_code_boundary():
    units = [_u("heading"), _u("heading"), _u("paragraph"), _u("paragraph"), _u("heading"), _u("paragraph")]
    forced = [g for g in range(len(units) - 1) if starts_section(units, g)]
    assert forced == [3]                               # heading after a paragraph; heading under heading is Jev's
    asked = [g for g in range(len(units) - 1) if g not in forced]
    assert section_groups(asked, 2) == [[0, 1], [2], [4]]   # never spans the code cut, at most 2 per call


def test_jev_decides_gaps_only_inside_sections_over_the_cap():
    units = [{**_u(k), "tokens": t} for k, t in
             [("heading", 5), ("paragraph", 10), ("heading", 5), ("paragraph", 30), ("paragraph", 30)]]
    assert jev_gaps(units, 40) == [2, 3]              # section 1 (15 tokens) is one chunk; section 2 (65) goes to Jev
    assert jev_gaps(units, 100) == []                 # every section fits: Jev is not asked


def test_split_at_the_gap_jev_least_keeps_together():
    keep = [0.9, 0.55, 0.8, 0.55]
    assert weakest_gap([0, 1, 2], keep) == 1
    assert weakest_gap([0, 1, 2, 3], keep) == 3        # a tie goes to the later gap: the larger left chunk
    d = Decision("d", "gap", "g", "segment_fanout@1", "k", ACCEPT,
                 {"same_topic_continues": {"type": "noul", "p": 0.3}, "after_depends_on_before": {"type": "noul", "p": 0.7}},
                 [], "fake", None, 0.5, 0)
    assert keep_strength(d) == 0.7                     # either yes keeps the sides together


def test_overlong_run_splits_where_jev_is_least_sure_until_every_piece_fits():
    tokens = [10] * 6
    keep = [0.9, 0.6, 0.95, 0.7, 0.8]                  # Jev's keep-together strength per gap
    pieces = SegmentStage._fit(tokens, keep, 0, 6, 30)
    assert pieces == [(0, 2), (2, 4), (4, 6)]         # gap 1 (0.6) first, then gap 3 (0.7), weakest of the fitting gaps 2-4
    assert all(sum(tokens[s:e]) <= 30 for s, e in pieces)
    assert [i for s, e in pieces for i in range(s, e)] == list(range(6))
    assert SegmentStage._fit(tokens, keep, 0, 3, 30) == [(0, 3)]   # a run that fits is left whole


def test_named_values_are_whole_word_matches():
    values = ["9.0", "9.0.1", "9.1", "19.0"]
    assert named_values("Applies to VCF 9.0. See 19.0 too", values) == ["9.0", "19.0"]
    assert named_values("Upgrade to 9.0.1 only", values) == ["9.0.1"]   # 9.0 is part of a longer version
    assert named_values("Works on all releases", values) == []
    assert named_values("Deploy VMware  Cloud\nFoundation now", ["Cloud Foundation", "Cloud"]) == ["Cloud Foundation", "Cloud"]


def test_rrf_and_composite_deterministic():
    fused = rrf_fuse([["x", "y"], ["y", "z"]], 60)
    assert max(fused, key=fused.get) == "y"
    assert composite({"relevant": {"p": 1.0}, "direct_evidence": {"p": 0.5}}, {"relevant": 0.6, "direct_evidence": 0.4}) == 0.8


def test_split_assignment_is_deterministic():
    fr = {"train": 0.8, "validation": 0.1, "test": 0.1}
    assert assign_split("doc-1", fr) == assign_split("doc-1", fr)
    counts = {k: 0 for k in fr}
    for i in range(2000):
        counts[assign_split(f"k{i}", fr)] += 1
    assert 1500 < counts["train"] < 1700


@pytest.mark.skipif(shutil.which("tesseract") is None, reason="Tesseract not installed")
def test_scanned_pdf_page_is_ocred(tmp_path: Path) -> None:
    """§8.2: a PDF page with no text layer is OCRed with Tesseract."""
    import pymupdf

    src = pymupdf.open()
    page = src.new_page()
    page.insert_text((72, 144), "Ledgerd replicates every journal entry.", fontsize=20)
    image = page.get_pixmap(dpi=200).tobytes("png")  # rasterize: the scan has pixels only
    scan = pymupdf.open()
    scan.new_page().insert_image(pymupdf.Rect(0, 0, 595, 842), stream=image)
    path = tmp_path / "scan.pdf"
    scan.save(str(path))

    doc = parse_pdf(path, ocr_dpi=300, ocr_language="eng")
    assert doc.meta.get("ocr_pages") == [1] and "ocr_missing_pages" not in doc.meta
    assert "journal entry" in " ".join(u.text for u in doc.units)


def _ingest_policy() -> dict:
    from jev_graph_builder.registry.loader import Registry

    return Registry(Path(__file__).resolve().parents[2] / "registry").policy("ingest")


def test_html_keeps_block_structure_and_is_sniffed_without_extension(tmp_path) -> None:
    """Captured pages often lack an .html name; the parser detects HTML from content and keeps
    headings and paragraph boundaries (no site-specific rules)."""
    from jev_graph_builder.parse.parsers import parse

    body = "".join(f"<p>Paragraph {i} explains one step of the procedure in enough words to be kept.</p>" for i in range(3))
    page = (f"<!DOCTYPE html><html><head><title>Guide</title><meta name=\"author\" content=\"Ann Writer\">"
            f"<meta property=\"article:published_time\" content=\"2025-06-01\"></head><body><nav>Home | About</nav><article>"
            f"<h1>Guide</h1><h2>Install</h2>{body}<h2>Upgrade</h2>{body}</article></body></html>")
    path = tmp_path / "capture.raw"
    path.write_text(page, encoding="utf-8")

    doc = parse(path, _ingest_policy())

    assert doc.mime == "text/html" and doc.title == "Guide"
    paragraphs = [u for u in doc.units if u.kind == "paragraph"]
    assert len(paragraphs) == 6, [u.text for u in doc.units]
    assert {u.heading_path[-1] for u in paragraphs} == {"Install", "Upgrade"}
    # the page's own metadata is kept for S0 to choose from; the extraction date is not (determinism)
    assert doc.meta["author"] == "Ann Writer" and doc.meta["date"] == "2025-06-01"
    assert "filedate" not in doc.meta


def test_html_without_main_element_keeps_section_headings(tmp_path) -> None:
    """Pages whose sections are headings inside plain <div>s (no <main>/<article>): every section heading and
    code block survives; only generic chrome (scripts, navigation, footers) is removed."""
    from jev_graph_builder.parse.parsers import parse

    sections = "".join(f"<div><h4>{name}</h4><div><p>{name} text for the article, long enough to keep.</p></div></div>"
                       for name in ("Issue", "Cause", "Resolution"))
    page = (f"<html><head><title>Article</title><script>var x = 1;</script></head><body><nav>Menu</nav>"
            f"<div class=\"wrap\"><h3>Article</h3>{sections}<pre>run --fix now</pre></div><footer>Legal</footer></body></html>")
    path = tmp_path / "article.html"
    path.write_text(page, encoding="utf-8")

    doc = parse(path, _ingest_policy())

    headings = [u.text for u in doc.units if u.kind == "heading"]
    assert headings == ["Article", "Issue", "Cause", "Resolution"], headings
    assert any(u.kind == "code" and "run --fix now" in u.text for u in doc.units)
    text = " ".join(u.text for u in doc.units)
    assert "Menu" not in text and "Legal" not in text and "var x" not in text


def test_markdown_front_matter_is_metadata_not_content(tmp_path) -> None:
    from jev_graph_builder.parse.parsers import parse

    path = tmp_path / "page.md"
    path.write_text('---\nversion: "2.1"\nsource_url: "https://example.org/p"\n---\n# Real Title\n\nBody text.\n', encoding="utf-8")

    doc = parse(path, _ingest_policy())

    assert doc.title == "Real Title"
    assert doc.meta == {"version": "2.1", "source_url": "https://example.org/p"}
    assert all("source_url" not in u.text for u in doc.units)
    rule = tmp_path / "rule.md"
    rule.write_text("---\nnot: [valid\n---\n# T\n", encoding="utf-8")  # not YAML: stays in the text
    assert parse(rule, _ingest_policy()).meta == {}


def test_bootstrap_sample_represents_every_input(tmp_path) -> None:
    """S0's sample covers every input path given to the command: a symlinked file belongs to the input that
    links it, same-named folders in two inputs are separate strata, and the cap is shared round-robin."""
    from jev_graph_builder.pipeline.s0_bootstrap import sample_files
    from jev_graph_builder.pipeline.s1_ingest import discover_origins

    elsewhere = tmp_path / "store"
    elsewhere.mkdir()
    for version in ("v1", "v2"):
        for guide in ("install", "upgrade", "admin"):
            d = tmp_path / version / guide
            d.mkdir(parents=True)
            for n in range(4):
                (d / f"p{n}.md").write_text(f"# {version} {guide} {n}\n", encoding="utf-8")
    linked = tmp_path / "linked"
    linked.mkdir()
    for n in range(3):
        target = elsewhere / f"a{n}.html"
        target.write_text(f"<html><body><p>{n}</p></body></html>", encoding="utf-8")
        (linked / f"a{n}.html").symlink_to(target)
    inputs = [str(tmp_path / "v1"), str(tmp_path / "v2"), str(linked)]

    origins = discover_origins(inputs)
    assert {origins[(linked / "a0.html").resolve()]} == {(2, "")}
    picked = sample_files(origins, per_stratum=2, max_total=5, size_buckets=[1000])

    assert len(picked) == 5
    assert {origins[f][0] for f in picked} == {0, 1, 2}
    assert picked == sample_files(origins, per_stratum=2, max_total=5, size_buckets=[1000])  # deterministic


def test_chrome_candidates_need_no_css_and_strip_only_approved(tmp_path) -> None:
    """Candidates come from text that recurs across pages, with selectors taken from the pages (classes,
    ids or tag paths); a page with no classes or ids still yields them. Only approved selectors strip."""
    from jev_graph_builder.parse.parsers import parse_html, repeated_elements

    pages = [f"<html><body><div><span>thumb_up</span><p>Body text number {i} about topic {i}.</p></div></body></html>"
             for i in range(4)]
    found = repeated_elements(pages, _ingest_policy()["html"], 200)
    icon = found[":scope > div > span"]
    assert len(icon["texts"]["thumb_up"]) == 4
    assert all(len(p) == 1 for t, p in found[":scope > div > p"]["texts"].items())

    html = {**_ingest_policy()["html"], "strip_selectors": [":scope > div > span"]}
    doc = parse_html(pages[0], html)
    text = "\n".join(u.text for u in doc.units)
    assert "thumb_up" not in text and "Body text number 0" in text
    assert "thumb_up" in "\n".join(u.text for u in parse_html(pages[0], _ingest_policy()["html"]).units)


def test_chrome_candidate_texts_are_what_the_selector_removes() -> None:
    """A class selector also matches elements with further classes: the article held in
    `div.h5.body` is shown under `div.h5`, so Jev judges everything stripping `div.h5` would remove."""
    from jev_graph_builder.parse.parsers import repeated_elements

    pages = [f"<html><body><div class='h5'>Rate this</div><div class='h5 body'>Article {i} text.</div></body></html>"
             for i in range(3)]
    texts = repeated_elements(pages, _ingest_policy()["html"], 200)["div.h5"]["texts"]
    assert len(texts["Rate this"]) == 3
    assert {"Article 0 text.", "Article 1 text.", "Article 2 text."} <= set(texts)


def _decision(outcome: str) -> Decision:
    return Decision("d", "k", "s", "qs@1", None, outcome, {}, [], "fake", None, 0.5, 0)


def test_new_relation_kept_only_on_significant_jev_support() -> None:
    passages = {f"p{i}": (f"source {i}", f"target {i}") for i in range(1, 9)}
    items = [{"name": "fixes", "definition": "d", "evidence": list(passages)}]

    def unsupported(n_confirmed: int) -> list[str]:
        confirmed = {f"source {i}" for i in range(1, n_confirmed + 1)}   # Jev says yes on these pairs only

        class FakeJev:
            reg = SimpleNamespace(policy=lambda _key: 100)

            async def ask(self, _qs, _state, _kind, _sid, fanout_items=None):
                return SimpleNamespace(decisions=[_decision(ACCEPT if p["source"] in confirmed else REJECT)
                                                  for p in fanout_items.values()],
                                       single=_decision(REJECT))   # no overlap between relations

        rule = lambda yes, total: significantly_above(yes, total, 0.5, 0.05)  # noqa: E731
        return [u["name"] for u in asyncio.run(check_definitions(FakeJev(), "relation_types", items, passages, 4,
                                                                 supported=rule)).unsupported]
    assert unsupported(7) == []            # 7 of 8 confirmed: significantly above half (p = 0.035)
    assert unsupported(6) == ["fixes"]     # 6 of 8: not significant (p = 0.145)
    assert unsupported(3) == ["fixes"]


def test_overlap_recheck_asks_every_pair_at_the_backoff_repeat() -> None:
    """§11.5 back-off in S0 disambiguation: after k rounds of flagged overlaps, the re-check asks every
    pair (renamed or new ones included) with repeat k, like jev-no-bullshit's per-type back-off."""
    items = [{"name": n, "definition": n, "evidence": []} for n in ("a", "b", "c")]

    def repeats(overlap_repeat: int) -> list[int]:
        seen: list[int] = []

        class FakeJev:
            reg = SimpleNamespace(policy=lambda _key: 100)

            async def ask(self, qs, state, _kind, _sid, repeat=0, **_kw):
                if "relation_a" in state:
                    seen.append(repeat)
                return SimpleNamespace(decisions=[], single=_decision(REJECT))

        asyncio.run(check_definitions(FakeJev(), "relation_types", items, {}, 4, overlap_repeat=overlap_repeat))
        return seen

    assert repeats(0) == [0, 0, 0]
    assert repeats(2) == [2, 2, 2]


def test_other_excess_is_a_significance_test() -> None:
    assert not significantly_above(0, 0, 0.15, 0.05)       # no picks, no evidence
    assert not significantly_above(3, 10, 0.15, 0.05)      # 30% of 10 picks: not significant
    assert significantly_above(30, 100, 0.15, 0.05)        # 30% of 100 picks: significant
    assert not significantly_above(160, 1000, 0.15, 0.05)  # 16% of 1000: within noise of 15%
    assert significantly_above(200, 1000, 0.15, 0.05)


def test_none_picks_do_not_dilute_the_other_excess() -> None:
    pol = {"none_label": "none", "other_label": "other", "other_baseline_share": 0.15,
           "support_baseline_share": 0.5, "significance": 0.05}
    rels = ["other"] * 30 + ["supersedes"] * 70 + ["none"] * 900   # 30% of real picks; 3% if `none` counted
    assert evolution_due(rels, pol, final=False) == (True, 30, 100)
    assert evolution_due(["none"] * 50, pol, final=True) == (False, 0, 0)          # nothing `other` to examine
    assert evolution_due(["other"] * 5 + ["none"], pol, final=True) == (True, 5, 5)  # final pass: what is left


def test_evolution_waits_for_enough_other_pairs_to_support_a_type() -> None:
    """2 of 2 picks `other` is a significant excess over 15%, but no type cited on 2 pairs can pass
    the support test (2/2 yes is not significant over 50%): no harness job until 5 (0.5**5 < 0.05)."""
    pol = {"none_label": "none", "other_label": "other", "other_baseline_share": 0.15,
           "support_baseline_share": 0.5, "significance": 0.05}
    assert evolution_due(["other"] * 2, pol, final=False) == (False, 2, 2)
    assert evolution_due(["other"] * 4, pol, final=True) == (False, 4, 4)
    assert evolution_due(["other"] * 5, pol, final=False) == (True, 5, 5)


def test_jev_accepted_drops_unsupported_and_overlapping_types() -> None:
    proposals = [{"name": n} for n in ("b_new", "a_new", "c_new", "d_new")]
    check = CheckReport(unsupported=[{"name": "d_new"}],
                        overlaps=[{"a": "a_new", "b": "b_new"},      # two proposals: the first by name is kept
                                  {"a": "c_new", "b": "supersedes"}])  # overlaps an existing relation: dropped
    assert [p["name"] for p in jev_accepted(proposals, check)] == ["a_new"]


def test_name_key_cap_used_only_on_pairs_for_jev():
    # Three same-type "vcenter" entities (merged by code) and one of another type: with a cap of 1,
    # the one cross-type pair must still reach Jev.
    keys = {"a": ("product", "vcenter"), "b": ("product", "vcenter"), "c": ("product", "vcenter"), "d": ("service", "vcenter")}
    got = name_key_pairs(keys, 1)
    assert len(got) == 1 and keys[got[0][0]] != keys[got[0][1]]


def test_route_sample_is_clipped_spread_and_within_token_limit() -> None:
    from jev_graph_builder.app import route_sample
    from jev_graph_builder.jev.tokens import TokenEstimator

    est = TokenEstimator(4)
    records = [f"r{i:03d}" + "x" * 2000 for i in range(300)]
    picks = route_sample(records, est, 1200, 1500)
    assert 1 <= len(picks) < len(records)
    assert all(len(p) <= 1200 for p in picks)
    assert est.value(picks) <= 1500
    assert picks[0].startswith("r000") and not picks[-1].startswith("r000")  # spread across the batch
    assert route_sample([], est, 1200, 1500) == []
    assert route_sample(["short"], est, 1200, 1500) == ["short"]
    shrunk = route_sample(records, est, 100, 250)          # 10 clips fill 250 tokens; JSON overhead drops one
    assert len(shrunk) == 9 and est.value(shrunk) <= 250
    assert shrunk[-1].startswith("r266")                   # still spread to the end of the batch, not truncated
    one = route_sample(records, est, 100_000, 50)          # a clip longer than the token limit is cut to it
    assert len(one) == 1 and len(one[0]) == est.chars(50)


def test_pending_in_keeps_whole_batches_and_runs_only_selected() -> None:
    from jev_graph_builder.ledger.ledger import WorkItem
    from jev_graph_builder.pipeline.common import pending_in

    items = [WorkItem("s", f"i{n}", f"h{n}", None) for n in range(5)]
    batches = [items[0:2], items[2:4], items[4:5]]
    out = pending_in(batches, [items[1], items[4]])            # a resumed run: i0, i2, i3 are done
    assert [(b, t) for b, t in out] == [(items[0:2], [items[1]]), (items[4:5], [items[4]])]
    assert pending_in(batches, []) == []


def test_gating_policy_invalidates_jev_stages_only():
    from jev_graph_builder.pipeline.common import Context, Deps

    def ctx(gating):
        qs = SimpleNamespace(ref="qs@1", content_hash="h")
        reg = SimpleNamespace(question_set=lambda n: qs, policy=lambda n: gating)
        return SimpleNamespace(reg=reg, jev=SimpleNamespace(provider=SimpleNamespace(model="m")))

    jev, code = Deps(question_sets=("qs@1",)), Deps()
    base = ctx({"threshold": 0.5, "backoff": 0.5})
    assert Context.deps_hash(base, jev) != Context.deps_hash(ctx({"threshold": 0.6, "backoff": 0.5}), jev)
    assert Context.deps_hash(base, jev) != Context.deps_hash(ctx({"threshold": 0.5, "backoff": 0.75}), jev)
    assert Context.deps_hash(base, code) == Context.deps_hash(ctx({"threshold": 0.6, "backoff": 0.5}), code)
