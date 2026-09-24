"""R-002 hard-coding lint (§2 P1, §19).

An AST check over the package source:

* string literals of `max_string_length` characters or more fail anywhere in
  the package, unless the literal sits in an allow-listed context (SQL, log
  event keys, exception messages, CLI help, docstrings), an allow-listed module,
  or is listed verbatim;
* numeric literals fail in the decision paths (`numeric_paths`) unless the value
  is allow-listed as a structural identity (e.g. `0`, `1`).

Everything configurable lives in `lint_allowlist.yaml` at the project root.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

DEFAULT_ALLOWLIST = "lint_allowlist.yaml"
_EXCEPTION_SUFFIXES = ("Error", "Exception", "Exceeded", "Missing")


@dataclass(frozen=True)
class Violation:
    path: str
    line: int
    kind: str
    literal: str

    def __str__(self) -> str:
        return f"{self.path}:{self.line}: {self.kind} literal {self.literal}"


@dataclass
class Allowlist:
    max_string_length: int
    numeric_paths: list[str]
    allowed_numbers: set[Any]
    modules: list[str]
    contexts: set[str]
    sql_pattern: re.Pattern[str]
    log_methods: set[str]
    identifier_pattern: re.Pattern[str]
    message_calls: set[str] = field(default_factory=set)
    message_sinks: set[str] = field(default_factory=set)
    literals: set[str] = field(default_factory=set)

    @classmethod
    def load(cls, path: Path) -> Allowlist:
        doc = yaml.safe_load(path.read_text(encoding="utf-8"))
        return cls(
            max_string_length=doc["max_string_length"],
            numeric_paths=list(doc["numeric_paths"]),
            allowed_numbers=set(doc["allowed_numbers"]),
            modules=list(doc.get("modules") or []),
            contexts=set(doc.get("contexts") or []),
            sql_pattern=re.compile(doc["sql_pattern"], re.IGNORECASE | re.DOTALL),
            log_methods=set(doc.get("log_methods") or []),
            identifier_pattern=re.compile(doc["identifier_pattern"]),
            message_calls=set(doc.get("message_calls") or []),
            message_sinks=set(doc.get("message_sinks") or []),
            literals=set(doc.get("literals") or []),
        )


def _matches(rel: str, prefixes: list[str]) -> bool:
    return any(rel == p or rel.startswith(p) for p in prefixes)


class _Visitor(ast.NodeVisitor):
    def __init__(self, rel: str, allow: Allowlist) -> None:
        self.rel = rel
        self.allow = allow
        self.numeric = _matches(rel, allow.numeric_paths)
        self.strings = not _matches(rel, allow.modules)
        self.exempt: set[int] = set()  # ids of Constant nodes in allowed contexts
        self.violations: list[Violation] = []

    # -- contexts ---------------------------------------------------------

    def _exempt_tree(self, node: ast.AST | None) -> None:
        if node is None:
            return
        for sub in ast.walk(node):
            if isinstance(sub, (ast.Constant, ast.JoinedStr)):
                self.exempt.add(id(sub))

    def _docstring(self, node: ast.AST) -> None:
        body = getattr(node, "body", None)
        if "docstring" in self.allow.contexts and body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
            self.exempt.add(id(body[0].value))

    def visit_Module(self, node: ast.Module) -> None:
        self._docstring(node)
        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._docstring(node)
        self.generic_visit(node)

    visit_AsyncFunctionDef = visit_FunctionDef  # type: ignore[assignment]

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self._docstring(node)
        self.generic_visit(node)

    def visit_Raise(self, node: ast.Raise) -> None:
        if "raise" in self.allow.contexts:
            self._exempt_tree(node.exc)
        self.generic_visit(node)

    def _sink(self, node: ast.AST) -> bool:
        """`errors.append`, `rep.warnings.append`, `(a if c else b).append` over message lists."""
        if isinstance(node, ast.IfExp):
            return self._sink(node.body) and self._sink(node.orelse)
        name = node.attr if isinstance(node, ast.Attribute) else node.id if isinstance(node, ast.Name) else None
        return name in self.allow.message_sinks

    def visit_Call(self, node: ast.Call) -> None:
        func = node.func
        name = func.attr if isinstance(func, ast.Attribute) else func.id if isinstance(func, ast.Name) else None
        if "log" in self.allow.contexts and name in self.allow.log_methods:
            self._exempt_tree(node)
        if "raise" in self.allow.contexts and name and (name in self.allow.message_calls or name.endswith(_EXCEPTION_SUFFIXES)):
            self._exempt_tree(node)
        if "diagnostic" in self.allow.contexts and name == "append" and isinstance(func, ast.Attribute) and self._sink(func.value):
            self._exempt_tree(node)
        if "regex" in self.allow.contexts and name == "compile" and isinstance(func, ast.Attribute) \
                and isinstance(func.value, ast.Name) and func.value.id == "re":
            self._exempt_tree(node)
        if "help" in self.allow.contexts and name in {"Option", "Argument", "Typer"}:
            for kw in node.keywords:
                if kw.arg == "help":
                    self._exempt_tree(kw.value)
        self.generic_visit(node)

    # -- literals ---------------------------------------------------------

    def _flag(self, text: str) -> bool:
        a = self.allow
        if not self.strings or len(text) < a.max_string_length or text in a.literals:
            return False
        if "sql" in a.contexts and a.sql_pattern.search(text):
            return False
        return not ("identifier" in a.contexts and a.identifier_pattern.fullmatch(text))

    def visit_Constant(self, node: ast.Constant) -> None:
        value = node.value
        if id(node) in self.exempt or isinstance(value, bool) or value is None:
            return
        if isinstance(value, str):
            if not self._flag(value):
                return
            self.violations.append(Violation(self.rel, node.lineno, "string", repr(value[:60])))
        elif isinstance(value, (int, float)) and self.numeric and value not in self.allow.allowed_numbers:
            self.violations.append(Violation(self.rel, node.lineno, "numeric", repr(value)))

    def visit_JoinedStr(self, node: ast.JoinedStr) -> None:
        """f-strings: the literal parts together are the hard-coded text."""
        if id(node) in self.exempt:
            return
        text = "".join(v.value for v in node.values if isinstance(v, ast.Constant) and isinstance(v.value, str))
        for v in node.values:
            if isinstance(v, ast.Constant):
                self.exempt.add(id(v))
        if self._flag(text):
            self.violations.append(Violation(self.rel, node.lineno, "string", repr(text[:60])))
        self.generic_visit(node)


def lint_source(source: str, rel: str, allow: Allowlist) -> list[Violation]:
    visitor = _Visitor(rel, allow)
    visitor.visit(ast.parse(source, filename=rel))
    return visitor.violations


def lint_package(package_root: Path, allow: Allowlist) -> list[Violation]:
    out: list[Violation] = []
    for path in sorted(package_root.rglob("*.py")):
        rel = path.relative_to(package_root).as_posix()
        out.extend(lint_source(path.read_text(encoding="utf-8"), rel, allow))
    return out


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(prog="jgb-lint-hardcoding", description="R-002 hard-coding lint")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--allowlist", type=Path, default=Path(DEFAULT_ALLOWLIST))
    args = parser.parse_args(argv)
    violations = lint_package(args.root, Allowlist.load(args.allowlist))
    for v in violations:
        print(v)
    print(f"{len(violations)} violation(s)")
    return 1 if violations else 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
