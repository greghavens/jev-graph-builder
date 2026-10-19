"""Pluggable parsers (§8.2): text plus structure, never a semantic judgment.

Each parser returns structural units (heading, paragraph, list, table, code)
with the heading path in force and the page number where known. Tables are
rendered as Markdown. Scanned PDF pages without a text layer are OCRed with
Tesseract when it is installed; otherwise the page is recorded in
`meta.ocr_missing_pages` so the gap is visible rather than silent.
"""

from __future__ import annotations

import mimetypes
import re
import shutil
import unicodedata
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

HEADING, PARAGRAPH, LIST, TABLE, CODE = "heading", "paragraph", "list", "table", "code"
_WS = re.compile(r"[ \t\r\f\v]+")
_BLANKS = re.compile(r"\n{3,}")
_HEADING_STYLE = re.compile(r"^heading\s*(\d+)$", re.IGNORECASE)
_TITLE_STYLE = "title"


@dataclass
class ParsedUnit:
    kind: str
    text: str
    heading_path: list[str]
    page: int | None = None


@dataclass
class ParsedDoc:
    mime: str
    title: str | None
    units: list[ParsedUnit]
    meta: dict[str, Any] = field(default_factory=dict)


class ParseError(Exception):
    pass


def normalize(text: str) -> str:
    """Unicode NFC and whitespace normalization (§8.2); keeps line structure."""
    text = unicodedata.normalize("NFC", text).replace(" ", " ")
    lines = [_WS.sub(" ", ln).strip() for ln in text.split("\n")]
    return _BLANKS.sub("\n\n", "\n".join(lines)).strip()


class _Headings:
    def __init__(self) -> None:
        self.stack: list[tuple[int, str]] = []

    def push(self, level: int, text: str) -> None:
        while self.stack and self.stack[-1][0] >= level:
            self.stack.pop()
        self.stack.append((level, text))

    @property
    def path(self) -> list[str]:
        return [t for _, t in self.stack]


def _table_md(rows: list[list[str]]) -> str:
    if not rows:
        return ""
    width = max(len(r) for r in rows)
    norm = [[(c or "").replace("\n", " ").replace("|", "\\|").strip() for c in r] + [""] * (width - len(r)) for r in rows]
    out = ["| " + " | ".join(norm[0]) + " |", "|" + "|".join(["---"] * width) + "|"]
    out.extend("| " + " | ".join(r) + " |" for r in norm[1:])
    return "\n".join(out)


# ------------------------------------------------------------------ markdown


_FRONT_MATTER = re.compile(r"\A---[ \t]*\n(.*?)\n---[ \t]*(?:\n|\Z)", re.S)


def split_front_matter(text: str) -> tuple[dict[str, Any], str]:
    """A leading YAML front-matter block is document metadata, not content: return it as a dict
    and the text without it. A block that is not a YAML mapping is left in the text. Scalars stay
    as written (`version: 9.10` is "9.10", not the float 9.1)."""
    import yaml

    m = _FRONT_MATTER.match(text)
    if not m:
        return {}, text
    try:
        data = yaml.load(m.group(1), Loader=yaml.BaseLoader)  # noqa: S506 - BaseLoader builds only str/list/dict
    except yaml.YAMLError:
        return {}, text
    if not isinstance(data, dict):
        return {}, text
    return data, text[m.end():]


def parse_markdown(text: str) -> ParsedDoc:
    from markdown_it import MarkdownIt

    front, text = split_front_matter(text)
    md = MarkdownIt("commonmark").enable("table")
    tokens = md.parse(text)
    lines = text.split("\n")
    heads = _Headings()
    units: list[ParsedUnit] = []
    title: str | None = None
    depth = 0
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if tok.type == "heading_open":
            level = int(tok.tag[1:])
            content = tokens[i + 1].content.strip()
            heads.push(level, content)
            if title is None:
                title = content
            units.append(ParsedUnit(HEADING, content, heads.path[:-1]))
            i += 3
            continue
        if tok.nesting == 1 and depth == 0 and tok.type in {"paragraph_open", "bullet_list_open", "ordered_list_open", "table_open", "blockquote_open"}:
            kind = {"paragraph_open": PARAGRAPH, "table_open": TABLE, "blockquote_open": PARAGRAPH}.get(tok.type, LIST)
            start, end = tok.map or (None, None)
            if start is not None:
                units.append(ParsedUnit(kind, "\n".join(lines[start:end]).strip(), heads.path))
            depth += 1
            i += 1
            continue
        if tok.nesting == 1:
            depth += 1
        elif tok.nesting == -1:
            depth -= 1
        elif depth == 0 and tok.type in {"fence", "code_block"}:
            units.append(ParsedUnit(CODE, tok.content.rstrip("\n"), heads.path))
        elif depth == 0 and tok.type == "html_block":
            units.append(ParsedUnit(PARAGRAPH, tok.content.strip(), heads.path))
        i += 1
    return ParsedDoc("text/markdown", title, [u for u in units if u.text], front)  # keys at the top level, where `metadata_fields` names them


# ---------------------------------------------------------------------- text


def parse_text(text: str) -> ParsedDoc:
    blocks = [b.strip() for b in re.split(r"\n\s*\n", text) if b.strip()]
    title = blocks[0].split("\n")[0] if blocks else None
    return ParsedDoc("text/plain", title, [ParsedUnit(PARAGRAPH, b, []) for b in blocks])


# ---------------------------------------------------------------------- html


def parse_html(raw: str, html: dict[str, Any]) -> ParsedDoc:
    """Faithful HTML -> Markdown: headings, lists, tables and code survive, so S2 sees the page's structure.

    Only generic page chrome (`html.strip_tags`, e.g. scripts, navigation, footers) is removed, and the first
    element matching `html.main_selectors` is kept when there is one. Nothing decides here what is content:
    leftover boilerplate becomes chunks that Jev classifies as boilerplate in S2 (P2).
    """
    import trafilatura
    from bs4 import BeautifulSoup
    from markdownify import markdownify

    soup = BeautifulSoup(raw, "lxml")
    structured = _json_ld(soup)  # read before `script` tags are stripped
    root = _page_body(soup, html)
    # Corpus-specific page chrome: selectors S0 found repeated across the sample, Jev verified, the human approved.
    for sel in html["strip_selectors"]:
        for el in root.select(sel):
            el.decompose()
    _code_blocks(soup)
    # Markdown is only read back for block structure; escaping `_` would change identifiers (`a_b` -> `a\\_b`).
    md = markdownify(str(root), heading_style="ATX", strip=html["drop_tags"], escape_underscores=False)
    doc = parse_markdown(md)
    doc.mime = "text/html"
    meta = trafilatura.extract_metadata(raw)
    found = _html_meta(meta.as_dict()) if meta is not None else {}
    if structured.keys() & _STRUCTURED_DATES:
        found.pop("date", None)  # trafilatura's guess at the dates the page states itself
    if meta is not None:
        doc.title = meta.title or doc.title
    doc.meta = {**found, **structured, **(doc.meta or {})}
    return doc


def _page_body(soup: Any, html: dict[str, Any]) -> Any:
    """The page without generic chrome (`html.strip_tags`): the first `html.main_selectors` match, else <body>."""
    for tag in soup(html["strip_tags"]):
        tag.decompose()
    root = next((el for sel in html["main_selectors"] if (el := soup.select_one(sel)) is not None), None)
    return root or soup.body or soup


# A class or id usable in a CSS selector without escaping (CSS syntax, not a site rule).
_CSS_NAME = re.compile(r"-?[_a-zA-Z][_a-zA-Z0-9-]*")


def _selector(el: Any, root: Any) -> str:
    """A CSS selector for `el` taken from the page itself: its classes, else its id, else its tag
    path from the page body. No attribute is required."""
    classes = sorted(c for c in el.get("class") or [] if _CSS_NAME.fullmatch(c))
    if classes:
        return ".".join([el.name, *classes])
    if isinstance(el.get("id"), str) and _CSS_NAME.fullmatch(el["id"]):
        return f"{el.name}#{el['id']}"
    parent = el.parent
    return f":scope > {el.name}" if parent is None or parent is root else f"{_selector(parent, root)} > {el.name}"


def repeated_elements(pages: list[str], html: dict[str, Any], text_chars: int) -> dict[str, dict[str, Any]]:
    """Inside each page body, the selectors of elements whose exact text recurs across pages, with the
    pages each occurs on and, per text, the pages it occurs on: the candidates for corpus-specific
    chrome (S0). Document text differs between pages; interface text repeats. Code only counts; Jev
    decides what is chrome."""
    from bs4 import BeautifulSoup

    roots = [_page_body(BeautifulSoup(raw, "lxml"), html) for raw in pages]
    selectors = {_selector(el, root) for root in roots for el in root.find_all(True)}
    found: dict[str, dict[str, Any]] = {}
    # Texts come from running each selector, so they are exactly what stripping it would remove: a class
    # selector also matches elements carrying further classes (e.g. the element holding the article).
    for sel in selectors:
        for i, root in enumerate(roots):
            for el in root.select(sel):
                full = " ".join(el.get_text(" ").split())
                if not full:
                    continue
                text = full if len(full) <= text_chars else full[:text_chars] + " …"  # Jev sees that it goes on
                entry = found.setdefault(sel, {"pages": set(), "texts": {}})
                entry["pages"].add(i)
                entry["texts"].setdefault(text, set()).add(i)
    return found


def _code_blocks(soup: Any) -> None:
    """Inline `<code>` that spans lines is a code block: as inline code its lines would be read as
    Markdown (a `# comment` line becomes a heading and the rest of the page is filed under it)."""
    for code in soup.find_all("code"):
        if code.find_parent("pre") is not None or not (code.find("br") or "\n" in code.get_text().strip()):
            continue
        for br in code.find_all("br"):
            br.replace_with("\n")
        code.wrap(soup.new_tag("pre"))


# schema.org date properties; when the page states them, they are its dates.
_STRUCTURED_DATES = frozenset({"datePublished", "dateModified", "dateCreated"})


def _json_ld(soup: Any) -> dict[str, Any]:
    """Text properties of the page's schema.org JSON-LD objects (a W3C/schema.org standard, not a site rule)."""
    import json

    out: dict[str, Any] = {}
    for script in soup.find_all("script", type="application/ld+json"):
        try:
            data = json.loads(script.get_text())
        except ValueError:
            continue
        objects = data if isinstance(data, list) else [data]
        objects = [o for x in objects if isinstance(x, dict) for o in (x.get("@graph") or [x]) if isinstance(o, dict)]
        for obj in objects:
            for k, v in _html_meta({k: v for k, v in obj.items() if not k.startswith("@")}).items():
                out.setdefault(k, v)
    return out


# trafilatura's `filedate` is the day of extraction, not a property of the page: keeping it would change
# the document on every run.
_EXTRACTION_ONLY = frozenset({"filedate"})


def _html_meta(fields: dict[str, Any]) -> dict[str, Any]:
    """The page's own metadata (url, date, author, site, tags, ...): text values and lists of text only."""
    out: dict[str, Any] = {}
    for k, v in fields.items():
        text = isinstance(v, str) or (isinstance(v, list) and all(isinstance(x, str) for x in v))
        if text and v and k not in _EXTRACTION_ONLY:
            out[k] = v
    return out


# ---------------------------------------------------------------------- docx


def parse_docx(path: Path) -> ParsedDoc:
    import docx
    from docx.table import Table
    from docx.text.paragraph import Paragraph

    d = docx.Document(str(path))
    heads = _Headings()
    units: list[ParsedUnit] = []
    title = (d.core_properties.title or None) if d.core_properties else None
    list_buf: list[str] = []

    def flush_list() -> None:
        if list_buf:
            units.append(ParsedUnit(LIST, "\n".join(list_buf), heads.path))
            list_buf.clear()

    for block in d.element.body.iterchildren():
        tag = block.tag.rsplit("}", 1)[-1]
        if tag == "p":
            p = Paragraph(block, d)
            text = p.text.strip()
            if not text:
                continue
            style = (p.style.name if p.style is not None else "") or ""
            m = _HEADING_STYLE.match(style)
            if m or style.lower() == _TITLE_STYLE:
                flush_list()
                level = int(m.group(1)) if m else 0
                heads.push(level, text)
                title = title or text
                units.append(ParsedUnit(HEADING, text, heads.path[:-1]))
            elif "list" in style.lower():
                list_buf.append(f"- {text}")
            else:
                flush_list()
                units.append(ParsedUnit(PARAGRAPH, text, heads.path))
        elif tag == "tbl":
            flush_list()
            t = Table(block, d)
            rows = [[cell.text for cell in row.cells] for row in t.rows]
            units.append(ParsedUnit(TABLE, _table_md(rows), heads.path))
    flush_list()
    return ParsedDoc("application/vnd.openxmlformats-officedocument.wordprocessingml.document", title, units)


# ----------------------------------------------------------------------- pdf


def _ocr_page(page: Any, dpi: int, language: str) -> str | None:
    if shutil.which("tesseract") is None:
        return None
    import io

    import pytesseract
    from PIL import Image

    pix = page.get_pixmap(dpi=dpi)
    img = Image.open(io.BytesIO(pix.tobytes("png")))
    return pytesseract.image_to_string(img, lang=language)


def parse_pdf(path: Path, ocr_dpi: int, ocr_language: str) -> ParsedDoc:
    import pymupdf

    doc = pymupdf.open(str(path))
    toc = doc.get_toc(simple=True)  # [level, title, page]
    toc_by_page: dict[int, list[tuple[int, str]]] = {}
    for level, heading, page_no in toc:
        toc_by_page.setdefault(page_no, []).append((level, heading.strip()))
    heads = _Headings()
    units: list[ParsedUnit] = []
    meta: dict[str, Any] = {"pages": doc.page_count}
    ocr_missing: list[int] = []
    for index, page in enumerate(doc):
        page_no = index + 1
        for level, heading in toc_by_page.get(page_no, []):
            heads.push(level, heading)
            units.append(ParsedUnit(HEADING, heading, heads.path[:-1], page_no))
        table_rects = []
        try:
            for table in page.find_tables().tables:
                table_rects.append(pymupdf.Rect(table.bbox))
                units.append(ParsedUnit(TABLE, _table_md(table.extract()), heads.path, page_no))
        except Exception:  # table detection is best-effort structure, never content loss
            table_rects = []
        blocks = page.get_text("blocks", sort=True)
        text_blocks = [b for b in blocks if b[6] == 0]
        if not text_blocks:
            text = _ocr_page(page, ocr_dpi, ocr_language)
            if text is None:
                ocr_missing.append(page_no)
                continue
            meta.setdefault("ocr_pages", []).append(page_no)
            for para in re.split(r"\n\s*\n", text):
                if para.strip():
                    units.append(ParsedUnit(PARAGRAPH, para.strip(), heads.path, page_no))
            continue
        for b in text_blocks:
            rect = pymupdf.Rect(b[:4])
            if any(rect.intersects(t) for t in table_rects):
                continue
            text = b[4].strip()
            if text:
                units.append(ParsedUnit(PARAGRAPH, text, heads.path, page_no))
    if ocr_missing:
        meta["ocr_missing_pages"] = ocr_missing
    title = (doc.metadata or {}).get("title") or (toc[0][1] if toc else None)
    return ParsedDoc("application/pdf", title, units, meta)


# ------------------------------------------------------------------ openapi


OPENAPI_MIME = "application/vnd.oai.openapi"
_HTTP_METHODS = ("get", "put", "post", "delete", "patch", "options", "head", "trace")
_MAX_REF_HOPS = 10


def _yaml12_loader() -> type:
    """A safe YAML loader whose booleans are YAML 1.2's (`true`/`false` only), as OpenAPI specifies:
    an enum of `ON, OFF` stays two strings, not YAML 1.1's True and False."""
    import yaml

    base = getattr(yaml, "CSafeLoader", yaml.SafeLoader)
    loader = type("Yaml12Loader", (base,), {})
    loader.yaml_implicit_resolvers = {
        first: [(tag, rx) for tag, rx in resolvers if tag != "tag:yaml.org,2002:bool"]
        for first, resolvers in base.yaml_implicit_resolvers.items()}
    loader.add_implicit_resolver("tag:yaml.org,2002:bool", re.compile(r"^(?:true|True|TRUE|false|False|FALSE)$"), list("tTfF"))
    return loader


def _load_structured(path: Path) -> Any:
    """A `.json`, `.yaml` or `.yml` file's data, or None when it does not parse."""
    import json

    import yaml

    text = _read_text(path)
    try:
        if path.suffix.lower() == ".json":
            return json.loads(text)
        return yaml.load(text, Loader=_yaml12_loader())  # noqa: S506 - a safe loader
    except (ValueError, yaml.YAMLError):
        return None


def load_openapi(path: Path) -> dict[str, Any] | None:
    """The spec in a file whose top level names an OpenAPI or Swagger version; None for any other file."""
    spec = _load_structured(path)
    return spec if isinstance(spec, dict) and ("openapi" in spec or "swagger" in spec) else None


class _Refs:
    """Follows `$ref`s inside the spec and into sibling files (`defs.yaml#/Name`), each file read once."""

    def __init__(self, path: Path, spec: dict[str, Any]) -> None:
        self.base = path.parent
        self.files: dict[str, Any] = {"": spec}

    def file(self, name: str) -> Any:
        if name not in self.files:
            target = (self.base / name).resolve()
            self.files[name] = _load_structured(target) if target.is_file() else None
        return self.files[name]

    def resolve(self, obj: Any) -> Any:
        for _ in range(_MAX_REF_HOPS):
            if not (isinstance(obj, dict) and isinstance(obj.get("$ref"), str)):
                break
            name, _, pointer = obj["$ref"].partition("#")
            node = self.file(name)
            for part in [p for p in pointer.split("/") if p]:
                node = node.get(part.replace("~1", "/").replace("~0", "~")) if isinstance(node, dict) else None
            if node is None:
                break
            obj = node
        return obj


def _ref_name(ref: str) -> str:
    return ref.rsplit("/", 1)[-1].rsplit("#", 1)[-1]


def _schema_type(schema: Any) -> str:
    """A schema as a short type: a referenced schema by name, arrays and compositions spelled out."""
    if not isinstance(schema, dict):
        return ""
    if isinstance(schema.get("$ref"), str):
        return _ref_name(schema["$ref"])
    if schema.get("type") == "array":
        return f"array of {_schema_type(schema.get('items')) or 'any'}"
    for key, sep in (("allOf", " and "), ("oneOf", " or "), ("anyOf", " or ")):
        if isinstance(schema.get(key), list):
            return sep.join(t for t in (_schema_type(s) for s in schema[key]) if t)
    kind = schema.get("type", "")
    kind = " or ".join(map(str, kind)) if isinstance(kind, list) else str(kind)
    if schema.get("format"):
        kind = f"{kind} ({schema['format']})"
    if isinstance(schema.get("enum"), list):
        kind = f"{kind}, one of: {', '.join(map(str, schema['enum']))}"
    return kind


def _text(value: Any) -> str:
    return "" if value is None else str(value).strip()


def _names(value: Any) -> list[str]:
    """An object's `required` list; a property's own `required: true` (an old Swagger habit) is not one."""
    return [str(v) for v in value] if isinstance(value, list) else []


def _properties(schema: dict[str, Any]) -> tuple[dict[str, Any], set[str], list[str]]:
    """An object schema's properties and required names, inline `allOf` parts included, and the
    schemas it extends (its `allOf` references)."""
    props = dict(schema.get("properties") or {})
    required = set(_names(schema.get("required")))
    bases: list[str] = []
    for part in schema.get("allOf") or []:
        if isinstance(part, dict) and isinstance(part.get("$ref"), str):
            bases.append(_ref_name(part["$ref"]))
        elif isinstance(part, dict):
            props.update(part.get("properties") or {})
            required |= set(_names(part.get("required")))
    return props, required, bases


class _OpenApiWriter:
    def __init__(self, refs: _Refs) -> None:
        self.refs = refs
        self.heads = _Headings()
        self.units: list[ParsedUnit] = []

    def heading(self, level: int, text: str) -> None:
        self.heads.push(level, text)
        self.units.append(ParsedUnit(HEADING, text, self.heads.path[:-1]))

    def add(self, kind: str, text: str) -> None:
        if text:
            self.units.append(ParsedUnit(kind, text, self.heads.path))

    def paragraphs(self, text: Any) -> None:
        for para in re.split(r"\n\s*\n", _text(text)):
            self.add(PARAGRAPH, para)

    def table(self, rows: list[list[str]]) -> None:
        if len(rows) > 1:
            self.add(TABLE, _table_md(rows))

    def operation(self, level: int, method: str, path: str, op: dict[str, Any], shared: list[Any]) -> None:
        self.heading(level, f"{method.upper()} {path}")
        self.paragraphs(op.get("summary"))
        if _text(op.get("description")) != _text(op.get("summary")):
            self.paragraphs(op.get("description"))
        if op.get("operationId"):
            self.add(PARAGRAPH, f"Operation ID: {op['operationId']}")
        if op.get("deprecated"):
            self.add(PARAGRAPH, "Deprecated.")
        params = {}
        for raw in [*shared, *(op.get("parameters") or [])]:  # an operation's own parameter overrides a path one
            p = self.refs.resolve(raw)
            if isinstance(p, dict) and p.get("name"):
                params[(p["name"], p.get("in"))] = p
        if params:
            self.add(PARAGRAPH, "Parameters:")
            self.table([["Name", "In", "Type", "Required", "Description"]] + [
                [str(p["name"]), _text(p.get("in")), _schema_type(p.get("schema") or p),
                 "yes" if p.get("required") else "no", _text(p.get("description"))] for p in params.values()])
        body = self.refs.resolve(op.get("requestBody"))
        if isinstance(body, dict):
            self.add(PARAGRAPH, "Request body:")
            self.paragraphs(body.get("description"))
            self.table([["Media type", "Schema", "Required"]] + [
                [media, _schema_type((c or {}).get("schema")), "yes" if body.get("required") else "no"]
                for media, c in (body.get("content") or {}).items()])
        responses = op.get("responses") or {}
        if responses:
            self.add(PARAGRAPH, "Responses:")
            rows = [["Status", "Description", "Schema"]]
            for code, raw in responses.items():
                r = self.refs.resolve(raw) or {}
                schemas = [_schema_type((c or {}).get("schema")) for c in (r.get("content") or {}).values()]
                schemas.append(_schema_type(r.get("schema")))
                rows.append([str(code), _text(r.get("description")), ", ".join(dict.fromkeys(s for s in schemas if s))])
            self.table(rows)

    def schema(self, level: int, name: str, schema: Any) -> None:
        schema = self.refs.resolve(schema)
        if not isinstance(schema, dict):
            return
        self.heading(level, name)
        self.paragraphs(schema.get("title"))
        self.paragraphs(schema.get("description"))
        props, required, bases = _properties(schema)
        if bases:
            self.add(PARAGRAPH, f"Extends: {', '.join(bases)}")
        if props:
            self.table([["Property", "Type", "Required", "Description"]] + [
                [str(k), _schema_type(v), "yes" if k in required else "no", _text((v or {}).get("description"))]
                for k, v in props.items()])
        elif not bases:
            self.add(PARAGRAPH, f"Type: {_schema_type(schema)}" if _schema_type(schema) else "")


def _external_ref_files(node: Any, out: set[str]) -> set[str]:
    if isinstance(node, dict):
        ref = node.get("$ref")
        if isinstance(ref, str) and not ref.startswith("#"):
            out.add(ref.partition("#")[0])
        for v in node.values():
            _external_ref_files(v, out)
    elif isinstance(node, list):
        for v in node:
            _external_ref_files(v, out)
    return out


def parse_openapi(path: Path, spec: dict[str, Any]) -> ParsedDoc:
    """An OpenAPI 3 or Swagger 2.0 spec as a document: one section per tag, one per operation (method
    and path, then its parameters, request body and responses as tables), then one per schema
    (its properties as a table). Referenced schemas are named, not expanded."""
    refs = _Refs(path, spec)
    w = _OpenApiWriter(refs)
    info = spec.get("info") or {}
    title = " ".join(t for t in (_text(info.get("title")) or path.stem, _text(info.get("version"))) if t)
    w.heading(1, title)
    w.paragraphs(info.get("description"))
    servers = [_text(s.get("url")) for s in spec.get("servers") or [] if isinstance(s, dict)]
    if spec.get("host"):
        servers.append(f"{spec['host']}{spec.get('basePath', '')}")
    if servers:
        w.add(PARAGRAPH, "Servers: " + ", ".join(servers))

    tag_docs = {t["name"]: t.get("description") for t in spec.get("tags") or [] if isinstance(t, dict) and t.get("name")}
    by_tag: dict[str, list[tuple[str, str, dict[str, Any], list[Any]]]] = {name: [] for name in tag_docs}
    for path_key, item in (spec.get("paths") or {}).items():
        item = refs.resolve(item)
        if not isinstance(item, dict):
            continue
        for method in _HTTP_METHODS:
            op = item.get(method)
            if isinstance(op, dict):
                tag = (op.get("tags") or ["Other operations"])[0]
                by_tag.setdefault(str(tag), []).append((method, str(path_key), op, item.get("parameters") or []))
    for tag, ops in by_tag.items():
        if not ops:
            continue
        w.heading(2, tag)
        w.paragraphs(tag_docs.get(tag))
        for method, path_key, op, shared in ops:
            w.operation(3, method, path_key, op, shared)

    schemas = dict((spec.get("components") or {}).get("schemas") or spec.get("definitions") or {})
    for name in sorted(_external_ref_files(spec, set())):
        data = refs.file(name)
        if isinstance(data, dict):
            schemas.update({k: v for k, v in data.items() if k not in schemas})
    if schemas:
        w.heading(2, "Schemas")
        for name, schema in schemas.items():
            w.schema(3, str(name), schema)
    meta = {"openapi_version": _text(spec.get("openapi") or spec.get("swagger")), "api_version": _text(info.get("version"))}
    return ParsedDoc(OPENAPI_MIME, title, w.units, meta)


# ------------------------------------------------------------------ dispatch


def _read_text(path: Path) -> str:
    return path.read_bytes().decode("utf-8", errors="replace")


_HTML_MARKERS = (b"<!doctype html", b"<html")
_SNIFF_BYTES = 1024


def _looks_like_html(path: Path) -> bool:
    """Format from content when the file name does not say (e.g. captured pages saved as `.raw`)."""
    with path.open("rb") as f:
        head = f.read(_SNIFF_BYTES).lstrip().lower()
    return head.startswith(_HTML_MARKERS)


def parser_for(path: Path, opts: dict[str, Any]) -> Callable[[], ParsedDoc]:
    """`opts` is `policies.ingest`: OCR settings for PDFs, conversion settings for HTML."""
    ocr, html = opts["ocr"], opts["html"]
    suffix = path.suffix.lower()
    if suffix in {".md", ".markdown"}:
        return lambda: parse_markdown(_read_text(path))
    if suffix in {".html", ".htm"}:
        return lambda: parse_html(_read_text(path), html)
    if suffix == ".docx":
        return lambda: parse_docx(path)
    if suffix == ".pdf":
        return lambda: parse_pdf(path, ocr["dpi"], ocr["language"])
    if suffix in {".json", ".yaml", ".yml"}:
        spec = load_openapi(path)
        if spec is not None:
            return lambda: parse_openapi(path, spec)
    if _looks_like_html(path):
        return lambda: parse_html(_read_text(path), html)
    mime, _ = mimetypes.guess_type(path.name)
    if suffix in {".txt", ".text", ".rst"} or (mime or "").startswith("text/"):
        return lambda: parse_text(_read_text(path))
    raise ParseError(f"no parser for {path.name}")


def parse(path: Path, opts: dict[str, Any]) -> ParsedDoc:
    doc = parser_for(path, opts)()
    for u in doc.units:
        u.text = normalize(u.text)
        u.heading_path = [normalize(h) for h in u.heading_path]
    doc.units = [u for u in doc.units if u.text]
    if doc.title:
        doc.title = normalize(doc.title)
    return doc
