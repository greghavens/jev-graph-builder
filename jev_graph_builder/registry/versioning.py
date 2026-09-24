"""Proposed Registry versions, diffs and activation (§7.3 step 6, §8.1, R-080).

A proposal is a full Registry tree staged under `registry/.proposals/<version>/`.
Nothing here ever changes the active tree except `activate`, which a human
triggers after `approve`.
"""

from __future__ import annotations

import difflib
import shutil
from dataclasses import dataclass
from pathlib import Path

from jev_graph_builder.registry.loader import PROPOSALS_DIR, Registry, RegistryError, tree_files, tree_hash


@dataclass
class Proposal:
    version: str
    path: Path


def proposals_root(active_root: Path) -> Path:
    return Path(active_root) / PROPOSALS_DIR


def stage_proposal(active_root: Path, draft_root: Path) -> Proposal:
    """Copy a draft tree into the proposal area, keyed by its content hash."""
    version = tree_hash(draft_root)
    dest = proposals_root(active_root) / version
    if not dest.exists():
        shutil.copytree(draft_root, dest, ignore=shutil.ignore_patterns(PROPOSALS_DIR, ".git"))
    return Proposal(version=version, path=dest)


def start_draft(active_root: Path, scratch: Path) -> Path:
    """A writable copy of the active tree for the bootstrap / evolution jobs to edit."""
    if scratch.exists():
        shutil.rmtree(scratch)
    shutil.copytree(active_root, scratch, ignore=shutil.ignore_patterns(PROPOSALS_DIR, ".git"))
    return scratch


def get_proposal(active_root: Path, version: str) -> Proposal:
    root = proposals_root(active_root)
    matches = [p for p in root.glob(f"{version}*") if p.is_dir()] if root.is_dir() else []
    if len(matches) != 1:
        raise RegistryError(f"no unique proposal matching `{version}`")
    return Proposal(version=matches[0].name, path=matches[0])


def list_proposals(active_root: Path) -> list[Proposal]:
    root = proposals_root(active_root)
    if not root.is_dir():
        return []
    return [Proposal(version=p.name, path=p) for p in sorted(root.iterdir()) if p.is_dir()]


def diff_trees(old_root: Path, new_root: Path) -> str:
    old = {p.relative_to(old_root).as_posix(): p for p in tree_files(old_root)}
    new = {p.relative_to(new_root).as_posix(): p for p in tree_files(new_root)}
    out: list[str] = []
    for rel in sorted(set(old) | set(new)):
        a = old[rel].read_text(encoding="utf-8").splitlines(keepends=True) if rel in old else []
        b = new[rel].read_text(encoding="utf-8").splitlines(keepends=True) if rel in new else []
        if a != b:
            out.extend(difflib.unified_diff(a, b, fromfile=f"active/{rel}", tofile=f"proposed/{rel}"))
    return "".join(out)


def changed_artifacts(old_root: Path, new_root: Path) -> set[str]:
    old = {p.relative_to(old_root).as_posix(): p.read_bytes() for p in tree_files(old_root)}
    new = {p.relative_to(new_root).as_posix(): p.read_bytes() for p in tree_files(new_root)}
    return {rel for rel in set(old) | set(new) if old.get(rel) != new.get(rel)}


def activate(active_root: Path, proposal: Proposal) -> Registry:
    """Replace the active tree with the proposal's (keeping the proposal area)."""
    active_root = Path(active_root)
    for p in tree_files(active_root):
        p.unlink()
    for p in tree_files(proposal.path):
        dest = active_root / p.relative_to(proposal.path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, dest)
    reg = Registry(active_root)
    if reg.version != proposal.version:
        raise RegistryError("activated tree hash does not match proposal version")
    return reg
