"""S5: entity resolution (§8.6).

Entities of one type with the same normalized name key (normalization is Registry
data) are one entity by code. Candidate pairs for Jev (code): same name key across
types, ANN neighbours on entity embeddings, co-occurrence in a document — each
source capped by policy. `QS.entity_align` decides each pair. `same` pairs are
merged by union-find; a merge is refused only when Jev said "not the same" for a
pair across the two clusters, and refused merges stay unmerged. S5 decides
sameness only: how distinct entities connect is a typed relation, found through
the chunks that mention them (S6). Each merged cluster's canonical name is a Jev Choice among its surface
forms (`QS.canonical_name`). Merges are logged with their decision IDs and are
recomputed from decisions on every run, so they are reversible.
"""

from __future__ import annotations

import itertools
import unicodedata
from typing import Any

from rapidfuzz import fuzz

from jev_graph_builder.ids import pairs, sha256_hex, short_key
from jev_graph_builder.jev.gating import ACCEPT
from jev_graph_builder.ledger.ledger import DONE, WorkItem
from jev_graph_builder.pipeline.common import (
    Context, Deps, RunOptions, RunReport, Stage, Writer, run_single_item, write_decisions,
)
from jev_graph_builder.pipeline.unionfind import merge
from jev_graph_builder.store import knn, repo

QS_ALIGN, QS_CANONICAL = "entity_align", "canonical_name"
PAIR_KIND, CLUSTER_KIND = "entity_pair", "entity_cluster"
PAIR_SEP = ":"


def name_key(name: str, rules: dict[str, Any]) -> str:
    s = unicodedata.normalize("NFKC", name)
    if rules.get("casefold"):
        s = s.casefold()
    if rules.get("strip_punctuation"):
        s = "".join(" " if unicodedata.category(c).startswith("P") else c for c in s)
    drop = set(rules.get("drop_tokens") or [])
    tokens = [t for t in s.split() if t not in drop]
    s = " ".join(tokens)
    return (rules.get("aliases") or {}).get(s, s)


def code_keys(ents: dict[str, dict[str, Any]], rules: dict[str, Any]) -> dict[str, tuple[str, str]]:
    """(type, normalized name) per entity, computed once: two entities with the same key are one entity by code."""
    return {eid: (e["entity_type"], name_key(e["surface"], rules)) for eid, e in ents.items()}


def name_key_pairs(keys: dict[str, tuple[str, str]], cap: int) -> list[tuple[str, str]]:
    """Candidates for Jev: entities sharing a normalized name but of different types (same-type
    ones are merged by code), at most `cap` per name."""
    groups: dict[str, list[str]] = {}
    for eid in sorted(keys):
        groups.setdefault(keys[eid][1], []).append(eid)
    out: list[tuple[str, str]] = []
    for members in groups.values():
        cross_type = ((a, b) for a, b in pairs(members) if keys[a] != keys[b])
        out.extend(itertools.islice(cross_type, cap))
    return out


def identical_pairs(ents: dict[str, dict[str, Any]], rules: dict[str, Any]) -> list[tuple[str, str]]:
    """Entities of one type with the same normalized name are one entity: merged by code, never
    sent to Jev. Each member is paired with its group's first (sorted) id."""
    keys = code_keys(ents, rules)
    groups: dict[tuple[str, str], list[str]] = {}
    for eid in sorted(ents):
        groups.setdefault(keys[eid], []).append(eid)
    return [(m[0], x) for m in groups.values() for x in m[1:]]


def pair_id(a: str, b: str) -> str:
    x, y = sorted((a, b))
    return f"{x}{PAIR_SEP}{y}"


class EntityStage(Stage):
    name = "entities"
    deps = Deps(question_sets=(QS_ALIGN, QS_CANONICAL), policies=("entity",), ontology=("entity_types",))

    # -------------------------------------------------------------- candidates

    async def _entities(self, ctx: Context) -> dict[str, dict[str, Any]]:
        rows = await ctx.db.fetch(
            "SELECT entity_id, coalesce(surface, canonical_name) AS surface, entity_type, description FROM entities "
            "WHERE corpus_id = %s AND status = 'accepted' ORDER BY entity_id", (ctx.corpus_id,))
        return {r["entity_id"]: r for r in rows}

    async def candidate_pairs(self, ctx: Context, ents: dict[str, dict[str, Any]]) -> dict[str, set[str]]:
        reg = ctx.reg
        caps = reg.policy("entity.candidates")
        found: dict[str, set[str]] = {}
        rules = reg.policy("entity.normalization")

        keys = code_keys(ents, rules)

        def add(a: str, b: str, source: str) -> None:
            # Pairs code already merges (same type and name) never go to Jev.
            if a != b and a in keys and b in keys and keys[a] != keys[b]:
                found.setdefault(pair_id(a, b), set()).add(source)

        for a, b in name_key_pairs(keys, caps["name_key"]):
            add(a, b, "name_key")

        knn_policy = reg.policy("store.candidate_knn")
        order = knn.order_by(knn_policy, "ORDER BY e2.embedding <=> e.embedding")
        rows = await knn.fetch(ctx.db, knn_policy,
            "SELECT e.entity_id AS a, n.entity_id AS b FROM entities e CROSS JOIN LATERAL ("
            " SELECT s.entity_id FROM (SELECT e2.entity_id, e2.embedding <=> e.embedding AS dist FROM entities e2"
            "  WHERE e2.corpus_id = e.corpus_id AND e2.entity_id <> e.entity_id AND e2.status = 'accepted' AND e2.embedding IS NOT NULL"
            f"  {order} LIMIT %(fetch)s) s ORDER BY s.dist, s.entity_id LIMIT %(k)s) n "
            "WHERE e.corpus_id = %(c)s AND e.status = 'accepted' AND e.embedding IS NOT NULL",
            {"k": caps["ann_k"], "fetch": caps["ann_k"] * reg.policy("store.ann_overfetch"), "c": ctx.corpus_id})
        for r in rows:
            add(r["a"], r["b"], "ann")

        rows = await ctx.db.fetch(
            "SELECT c.doc_id, array_agg(DISTINCT m.entity_id) AS ids FROM mentions m JOIN chunks c ON c.chunk_id = m.chunk_id "
            "WHERE c.corpus_id = %s AND m.status = 'accepted' GROUP BY c.doc_id", (ctx.corpus_id,))
        min_sim = caps["cooccurrence_min_name_similarity"]
        for r in rows:
            ids = sorted(i for i in r["ids"] if i in ents)
            close = [
                (a, b) for a, b in pairs(ids)
                if fuzz.token_set_ratio(ents[a]["surface"].casefold(), ents[b]["surface"].casefold()) >= min_sim
            ]
            for a, b in close[: caps["cooccurrence_per_doc"]]:
                add(a, b, "cooccurrence")
        return found

    async def _contexts(self, ctx: Context, ids: list[str]) -> dict[str, list[str]]:
        """Snippets around each accepted mention. No document metadata (§8.2): the same thing
        documented for two releases is one entity."""
        n, width = ctx.reg.policy("entity.context_snippets"), ctx.reg.policy("entity.context_chars")
        rows = await ctx.db.fetch(
            "SELECT m.entity_id, c.text, m.char_start, m.char_end FROM mentions m JOIN chunks c ON c.chunk_id = m.chunk_id "
            "WHERE m.entity_id = ANY(%s) AND m.status = 'accepted' ORDER BY m.entity_id, c.doc_id, c.ord, m.char_start", (ids,))
        out: dict[str, list[str]] = {i: [] for i in ids}
        for r in rows:
            if len(out[r["entity_id"]]) >= n or r["char_start"] is None:
                continue
            s, e = max(r["char_start"] - width, 0), r["char_end"] + width
            out[r["entity_id"]].append(r["text"][s:e])
        return out

    async def enumerate(self, ctx: Context) -> list[WorkItem]:
        ents = await self._entities(ctx)
        found = await self.candidate_pairs(ctx, ents)
        contexts = await self._contexts(ctx, list(ents))
        ctx.cache["s5_entities"], ctx.cache["s5_pairs"] = ents, found
        items = []
        for pid in sorted(found):
            a, b = pid.split(PAIR_SEP)
            view = {i: {"name": ents[i]["surface"], "type": ents[i]["entity_type"],
                        "description": ents[i]["description"] or "", "contexts": contexts[i]} for i in (a, b)}
            items.append(self.item(ctx, pid, view[a], view[b], payload={"a": view[a], "b": view[b], "sources": sorted(found[pid])}))
        return items

    # ---------------------------------------------------------------- decision

    async def process(self, ctx: Context, item: WorkItem) -> Writer:
        r = await ctx.jev.ask(QS_ALIGN, {"entity_a": item.payload["a"], "entity_b": item.payload["b"]}, PAIR_KIND, item.item_id)

        async def write(conn) -> str:
            await write_decisions(conn, r)
            return DONE

        return write

    # ------------------------------------------------------------------ merges

    async def _pair_evidence(self, ctx: Context, ents: dict[str, Any]) -> tuple[list[tuple[str, str, float]], dict[frozenset[str], bool], dict[str, str]]:
        qs = ctx.reg.question_set(QS_ALIGN)
        rows = await ctx.db.fetch(
            "SELECT DISTINCT ON (d.subject_id) d.subject_id, d.decision_id, d.outcome, d.answers "
            "FROM decisions d WHERE d.subject_kind = %s AND d.question_set = %s ORDER BY d.subject_id, d.created_at DESC",
            (PAIR_KIND, qs.ref))
        accepted, known, dec_of = [], {}, {}
        q_same = ctx.reg.policy("entity.same_question")
        for r in rows:
            a, b = r["subject_id"].split(PAIR_SEP)
            if a not in ents or b not in ents:
                continue
            p = float(r["answers"]["answers"][q_same]["p"])
            known[frozenset((a, b))] = r["outcome"] == ACCEPT
            if r["outcome"] == ACCEPT:
                accepted.append((a, b, p))
                dec_of[r["subject_id"]] = r["decision_id"]
        return accepted, known, dec_of

    async def finalize(self, ctx: Context, report: RunReport, opts: RunOptions) -> None:
        ents = ctx.cache.get("s5_entities") or await self._entities(ctx)
        accepted, known, dec_of = await self._pair_evidence(ctx, ents)
        identical = identical_pairs(ents, ctx.reg.policy("entity.normalization"))
        outcome = merge(sorted(ents), accepted, known, identical)
        clusters = {r: m for r, m in outcome.clusters().items() if len(m) > 1}
        merge_item = self.item(ctx, f"merge{PAIR_SEP}{ctx.corpus_id}", sorted(dec_of.values()), outcome.refused, identical)

        async def merges_writer():
            async def write(conn) -> str:
                rows = []
                for root, members in clusters.items():
                    for m in members:
                        if m == root:
                            continue
                        ids = sorted(v for k, v in dec_of.items() if m in k.split(PAIR_SEP))
                        rows.append({"merge_id": sha256_hex(m, root, ids), "corpus_id": ctx.corpus_id, "entity_id": m,
                                     "merged_into": root, "decision_ids": ids, "status": "accepted",
                                     "registry_version": ctx.registry_version, "created_run_id": ctx.run_id})
                keep = [r["merge_id"] for r in rows]
                await conn.execute("UPDATE entity_merges SET status = 'superseded' WHERE corpus_id = %s AND status = 'accepted' "
                                   "AND NOT (merge_id = ANY(%s))", (ctx.corpus_id, keep))
                await repo.upsert_many(conn, "entity_merges", rows, key=("merge_id",))
                await conn.execute("UPDATE entities SET merged_into = NULL WHERE corpus_id = %s AND merged_into IS NOT NULL", (ctx.corpus_id,))
                for r in rows:
                    await conn.execute("UPDATE entities SET merged_into = %s WHERE entity_id = %s", (r["merged_into"], r["entity_id"]))
                # Merges Jev contradicted (§8.6 step 4) stay unmerged; they are listed in the item payload.
                return DONE
            return write

        await run_single_item(ctx, self, merge_item, merges_writer)
        for root, members in sorted(clusters.items()):
            await self._canonical(ctx, ents, root, members)

    async def _canonical(self, ctx: Context, ents: dict[str, Any], root: str, members: list[str]) -> None:
        forms = sorted({ents[m]["surface"] for m in members})
        item = self.item(ctx, f"{CLUSTER_KIND}{PAIR_SEP}{root}", members, forms)

        async def factory():
            decision = None
            name = forms[0]
            if len(forms) > 1:
                options = {short_key(f): f for f in forms}
                r = await ctx.jev.ask(QS_CANONICAL, {"entity_type": ents[root]["entity_type"], "forms": forms},
                                      CLUSTER_KIND, root, dynamic={"forms": options})
                decision = r.single
                if decision.outcome == ACCEPT:
                    name = options[decision.answers["canonical"]["choice"]]

            async def write(conn) -> str:
                await write_decisions(conn, decision)
                aliases = forms
                await conn.execute("UPDATE entities SET canonical_name = %s, aliases = %s WHERE entity_id = %s", (name, aliases, root))
                # The cluster keeps the union of its members' verified attributes; the first value
                # in member order wins, so the result does not depend on merge order.
                await conn.execute(
                    "UPDATE entities SET attributes = (SELECT coalesce(jsonb_object_agg(k, v), '{}'::jsonb) FROM ("
                    "  SELECT DISTINCT ON (a.k) a.k, a.v FROM unnest(%s::text[]) WITH ORDINALITY m(id, n) "
                    "  JOIN entities e ON e.entity_id = m.id, jsonb_each(coalesce(e.attributes, '{}'::jsonb)) a(k, v) "
                    "  ORDER BY a.k, m.n) x) WHERE entity_id = %s",
                    (sorted(members), root))
                return DONE
            return write

        await run_single_item(ctx, self, item, factory)
