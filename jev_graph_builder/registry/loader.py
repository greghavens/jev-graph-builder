"""Load, validate and version the Registry (§7).

The Registry is data, not code: every domain word, prompt, question, threshold
and size lives here. The version is the content hash of the whole tree
(excluding the proposal staging area), and each artifact has its own hash so
that the ledger can compute precise invalidation (§14.3).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import Any

import jinja2
import jsonschema
import yaml

from jev_graph_builder.ids import sha256_hex
from jev_graph_builder.registry import schemas

PROPOSALS_DIR = ".proposals"
_EXCLUDED_TOP = {PROPOSALS_DIR, ".git"}
_QS_FILE = re.compile(r"^(?P<name>[a-z0-9_]+)@(?P<version>\d+)\.ya?ml$")
_PROMPT_FILE = re.compile(r"^(?P<name>[a-z0-9_]+)@(?P<version>\d+)\.md$")


ONTOLOGY_KINDS = ("entity_types", "relation_types", "claim_types", "doc_types", "chunk_roles", "topics")

class RegistryError(Exception):
    """The Registry is missing an artifact or an artifact is malformed."""


@dataclass(frozen=True)
class QuestionSet:
    name: str
    version: int
    jev_model: str
    state_template: dict[str, Any]
    questions: dict[str, Any]
    gating: dict[str, Any]
    fanout: dict[str, Any] | None
    content_hash: str
    raw: dict[str, Any] = field(repr=False)

    @property
    def ref(self) -> str:
        return f"{self.name}@{self.version}"


@dataclass(frozen=True)
class Prompt:
    name: str
    version: int
    meta: dict[str, Any]
    body: str
    content_hash: str

    @property
    def ref(self) -> str:
        return f"{self.name}@{self.version}"

    @property
    def task(self) -> str:
        """The body as a reader of the task sees it: each template variable shows as its own name."""
        env = jinja2.Environment(undefined=_NamedUndefined, autoescape=False)
        env.filters["tojson"] = lambda value, *_a, **_k: str(value)
        return env.from_string(self.body).render().strip()


class _NamedUndefined(jinja2.Undefined):
    def __str__(self) -> str:
        return self._undefined_name or ""

    def __getattr__(self, attr: str) -> Any:
        if attr.startswith("__"):
            raise AttributeError(attr)
        return _NamedUndefined(name=f"{self._undefined_name}.{attr}")

    __getitem__ = __getattr__


def _load_yaml(path: Path) -> Any:
    try:
        return yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise RegistryError(f"{path}: invalid YAML: {exc}") from exc


def _validate(doc: Any, schema: dict, where: Path | str) -> None:
    try:
        jsonschema.validate(doc, schema)
    except jsonschema.ValidationError as exc:
        raise RegistryError(f"{where}: {exc.message} at {list(exc.absolute_path)}") from exc


def split_frontmatter(text: str) -> tuple[dict[str, Any], str]:
    if not text.startswith("---"):
        return {}, text
    _, fm, body = text.split("---", 2)
    return yaml.safe_load(fm) or {}, body.lstrip("\n")


def tree_files(root: Path) -> list[Path]:
    files = []
    for p in sorted(root.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(root)
        if rel.parts[0] in _EXCLUDED_TOP or p.name.startswith("."):
            continue
        files.append(p)
    return files


def tree_hash(root: Path, prefix: str | None = None) -> str:
    parts: list[str] = []
    for p in tree_files(root):
        rel = p.relative_to(root).as_posix()
        if prefix is not None and not (rel == prefix or rel.startswith(prefix.rstrip("/") + "/")):
            continue
        parts.append(rel)
        parts.append(sha256_hex(p.read_bytes().decode("utf-8", errors="replace")))
    return sha256_hex(*parts)


class Registry:
    """Read-only view over one Registry tree."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root).resolve()
        if not self.root.is_dir():
            raise RegistryError(f"registry not found: {self.root}")
        # The tree is treated as immutable for the lifetime of this object.
        self._ontology_cache: dict[str, list[dict[str, Any]]] = {}
        self._qs_cache: dict[Path, QuestionSet] = {}

    # ---------- versioning ----------
    @cached_property
    def version(self) -> str:
        return tree_hash(self.root)

    def artifact_hash(self, *relpaths: str) -> str:
        """Hash of a subset of the tree (file or directory prefixes), for input_hash."""
        return sha256_hex(*[tree_hash(self.root, rp) for rp in relpaths])

    # ---------- top-level documents ----------
    @cached_property
    def corpus(self) -> dict[str, Any]:
        doc = _load_yaml(self.root / "corpus.yaml") or {}
        _validate(doc, schemas.CORPUS, "corpus.yaml")
        return doc

    @cached_property
    def policies(self) -> dict[str, Any]:
        doc = _load_yaml(self.root / "policies.yaml") or {}
        _validate(doc, schemas.POLICIES, "policies.yaml")
        return doc

    @cached_property
    def profiles(self) -> dict[str, Any]:
        doc = _load_yaml(self.root / "profiles.yaml") or {}
        _validate(doc, schemas.PROFILES, "profiles.yaml")
        return doc

    def policy(self, dotted: str) -> Any:
        """Look up a policy value. There are no code defaults (P1): a missing key is an error."""
        node: Any = self.policies
        path = dotted.removeprefix("policies.")
        for part in path.split("."):
            if not isinstance(node, dict) or part not in node:
                raise RegistryError(f"policies.yaml has no `{path}`")
            node = node[part]
        return node

    def policy_or(self, dotted: str, fallback: Any) -> Any:
        try:
            return self.policy(dotted)
        except RegistryError:
            return fallback

    def profile(self, kind: str, name: str) -> dict[str, Any]:
        try:
            return dict(self.profiles[kind][name], name=name)
        except KeyError as exc:
            raise RegistryError(f"profiles.yaml has no {kind} profile `{name}`") from exc

    # ---------- ontology ----------
    def ontology(self, kind: str) -> list[dict[str, Any]]:
        if kind not in self._ontology_cache:
            path = self.root / "ontology" / f"{kind}.yaml"
            if not path.is_file():
                raise RegistryError(f"ontology/{kind}.yaml missing")
            doc = _load_yaml(path) or {}
            _validate(doc, schemas.ONTOLOGY_FILE, path)
            self._ontology_cache[kind] = list(doc["items"])
        return list(self._ontology_cache[kind])

    def ontology_names(self, kind: str) -> list[str]:
        return [item["name"] for item in self.ontology(kind)]

    def ontology_item(self, kind: str, name: str) -> dict[str, Any]:
        for item in self.ontology(kind):
            if item["name"] == name:
                return item
        raise RegistryError(f"ontology/{kind}: unknown `{name}`")

    @cached_property
    def structural_labels(self) -> dict[str, str]:
        doc = _load_yaml(self.root / "ontology" / "structural_edges.yaml") or {}
        _validate(doc, schemas.STRUCTURAL_LABELS, "ontology/structural_edges.yaml")
        return doc

    def structural_label(self, role: str) -> str:
        try:
            return self.structural_labels[role]
        except KeyError as exc:
            raise RegistryError(f"ontology/structural_edges.yaml has no `{role}`") from exc

    def validate_label(self, kind: str, label: str) -> None:
        """§10.1: labels are text that must exist in the active Registry."""
        if label not in self.ontology_names(kind):
            raise RegistryError(f"label `{label}` is not a registered {kind}")

    # ---------- question sets ----------
    @cached_property
    def _question_set_files(self) -> dict[tuple[str, int], Path]:
        out: dict[tuple[str, int], Path] = {}
        for p in sorted((self.root / "question_sets").glob("*.y*ml")):
            m = _QS_FILE.match(p.name)
            if not m:
                raise RegistryError(f"question set file name must be <name>@<ver>.yaml: {p.name}")
            out[(m["name"], int(m["version"]))] = p
        return out

    def question_set_names(self) -> list[str]:
        return sorted({n for n, _ in self._question_set_files})

    def active_qs_version(self, name: str) -> int:
        active = self.policy("question_sets.active")
        if name not in active:
            raise RegistryError(f"policies.question_sets.active has no `{name}`")
        return int(active[name])

    def question_set(self, name: str, version: int | None = None) -> QuestionSet:
        version = self.active_qs_version(name) if version is None else version
        path = self._question_set_files.get((name, version))
        if path is None:
            raise RegistryError(f"question set {name}@{version} not found")
        return self._load_qs(path)

    def all_question_sets(self) -> list[QuestionSet]:
        return [self._load_qs(p) for p in self._question_set_files.values()]

    def _load_qs(self, path: Path) -> QuestionSet:
        if path in self._qs_cache:
            return self._qs_cache[path]
        doc = _load_yaml(path)
        _validate(doc, schemas.QUESTION_SET, path)
        m = _QS_FILE.match(path.name)
        assert m is not None
        if doc["name"] != m["name"] or int(doc["version"]) != int(m["version"]):
            raise RegistryError(f"{path.name}: name/version do not match file name")
        qs = self._qs_cache[path] = QuestionSet(
            name=doc["name"],
            version=int(doc["version"]),
            jev_model=doc["jev_model"],
            state_template=doc["state_template"],
            questions=doc["questions"],
            gating=doc.get("gating") or {},
            fanout=doc.get("fanout"),
            content_hash=sha256_hex(doc),
            raw=doc,
        )
        return qs

    # ---------- prompts & schemas ----------
    def prompt(self, ref: str) -> Prompt:
        name, _, ver = ref.partition("@")
        version = int(ver) if ver else int(self.policy(f"prompts.active.{name}"))
        path = self.root / "prompts" / f"{name}@{version}.md"
        if not path.is_file():
            raise RegistryError(f"prompt {name}@{version} not found")
        text = path.read_text(encoding="utf-8")
        meta, body = split_frontmatter(text)
        _validate(meta, schemas.PROMPT_FRONTMATTER, path)
        return Prompt(name=name, version=version, meta=meta, body=body, content_hash=sha256_hex(text))

    def all_prompts(self) -> list[Prompt]:
        out = []
        for p in sorted((self.root / "prompts").glob("*.md")):
            m = _PROMPT_FILE.match(p.name)
            if not m:
                raise RegistryError(f"prompt file name must be <name>@<ver>.md: {p.name}")
            out.append(self.prompt(f"{m['name']}@{m['version']}"))
        return out

    def schema(self, ref: str) -> dict[str, Any]:
        path = self.root / "schemas" / f"{ref}.json"
        if not path.is_file():
            raise RegistryError(f"schema {ref} not found")
        doc = json.loads(path.read_text(encoding="utf-8"))
        jsonschema.Draft202012Validator.check_schema(doc)
        return doc

    def schema_path(self, ref: str) -> Path:
        self.schema(ref)
        return self.root / "schemas" / f"{ref}.json"

    # ---------- training ----------
    def training_template(self, name: str) -> dict[str, Any]:
        path = self.root / "training" / "templates" / f"{name}.yaml"
        if not path.is_file():
            raise RegistryError(f"training template {name} not found")
        doc = _load_yaml(path)
        _validate(doc, schemas.TRAINING_TEMPLATE, path)
        return doc

    def training_template_names(self) -> list[str]:
        return sorted(p.stem for p in (self.root / "training" / "templates").glob("*.yaml"))
