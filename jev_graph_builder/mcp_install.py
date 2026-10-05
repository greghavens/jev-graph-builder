"""Register a build directory's graph as an MCP server with agent clients: Claude Code, pi, OpenCode, Triggerfish.

Every client runs the same command (`jev-graph-builder mcp --name <name>`) with the build directory's
config and registry in its environment; the TypeSafe key comes from the key file (`config.secret`).
With a repo, the entry goes in that repo's own config (Claude Code `.mcp.json`, OpenCode `opencode.json`),
so only sessions in that repo get it. Without one (`--user`), it goes in the client's user config:
Claude Code through its CLI, the others by editing their config file. Triggerfish and pi have only a
user config. An existing entry of the same name is replaced; everything else is kept.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from jev_graph_builder.config import CONFIG_FILE_ENV

CLIENTS = ("claude", "pi", "opencode", "triggerfish")
TRIGGERFISH_LEVELS = ("PUBLIC", "INTERNAL", "CONFIDENTIAL", "RESTRICTED")


class InstallError(Exception):
    pass


@dataclass(frozen=True)
class Server:
    name: str
    exe: Path
    config: Path
    registry: Path

    @property
    def args(self) -> list[str]:
        return ["mcp", "--name", self.name]

    @property
    def env(self) -> dict[str, str]:
        return {CONFIG_FILE_ENV: str(self.config), "JGB_REGISTRY_PATH": str(self.registry)}


def claude_entry(server: Server) -> dict[str, Any]:
    """`mcpServers` entry for a Claude Code project `.mcp.json`."""
    return {"type": "stdio", "command": str(server.exe), "args": server.args, "env": server.env}


def claude_add_args(server: Server, scope: str) -> list[str]:
    """`claude mcp add` arguments for `server` (`scope`: user | project | local)."""
    env = [a for k, v in server.env.items() for a in ("-e", f"{k}={v}")]
    return ["claude", "mcp", "add", "--scope", scope, server.name, *env, "--", str(server.exe), *server.args]


def pi_entry(server: Server) -> dict[str, Any]:
    """`mcpServers` entry for pi's built-in MCP support (pi 0.99+)."""
    return {"command": str(server.exe), "args": server.args, "env": server.env}


def opencode_entry(server: Server) -> dict[str, Any]:
    """`mcp` entry for OpenCode."""
    return {"type": "local", "command": [str(server.exe), *server.args], "environment": server.env, "enabled": True}


def triggerfish_entry(server: Server, classification: str) -> dict[str, Any]:
    """`mcp.servers` entry for Triggerfish. Without a classification its gateway leaves the server UNTRUSTED,
    and the agent can't call it."""
    return {"command": str(server.exe), "args": server.args, "env": server.env,
            "classification": classification, "enabled": True}


def _config_home() -> Path:
    return Path(os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config")


def pi_path() -> Path:
    return Path(os.environ.get("PI_CODING_AGENT_DIR") or Path.home() / ".pi" / "agent") / "mcp.json"


def opencode_path(base: Path | None = None) -> Path:
    """The OpenCode config in `base` (default: the user config dir) that exists, `opencode.jsonc` first,
    else `opencode.json`."""
    base = base or _config_home() / "opencode"
    return next((p for p in (base / "opencode.jsonc", base / "opencode.json") if p.is_file()), base / "opencode.json")


def triggerfish_path() -> Path:
    return Path(os.environ.get("TRIGGERFISH_DATA_DIR") or Path.home() / ".triggerfish") / "triggerfish.yaml"


def _set_json(path: Path, keys: tuple[str, ...], name: str, entry: dict[str, Any]) -> bool:
    """Put `entry` at `keys` + `name` in a JSON file; True when it replaced one. A file that isn't plain JSON
    (for example JSONC with comments) is left alone."""
    data: dict[str, Any] = {}
    if path.is_file() and path.read_text().strip():
        try:
            data = json.loads(path.read_text())
        except json.JSONDecodeError as exc:
            raise InstallError(f"{path} isn't plain JSON ({exc}); add this under {'.'.join(keys)}.{name} by hand: "
                               f"{json.dumps(entry)}") from exc
    node = data
    for key in keys:
        node = node.setdefault(key, {})
    replaced = name in node
    node[name] = entry
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n")
    return replaced


def _set_triggerfish(path: Path, name: str, entry: dict[str, Any]) -> bool:
    """Put `entry` at `mcp.servers.<name>`, after copying the file to `<file>.bak`. YAML is rewritten,
    so the file's leading comment lines are kept and comments elsewhere are lost."""
    if not path.is_file():
        raise InstallError(f"no {path}: run `triggerfish dive` first")
    text = path.read_text()
    data = yaml.safe_load(text) or {}
    servers = data.setdefault("mcp", {}).setdefault("servers", {})
    replaced = name in servers or name in (data.get("mcp_servers") or {})
    (data.get("mcp_servers") or {}).pop(name, None)
    servers[name] = entry
    header = "".join(line for line in text.splitlines(keepends=True)[:_leading_comments(text)])
    shutil.copy2(path, path.with_name(path.name + ".bak"))
    path.write_text(header + yaml.safe_dump(data, sort_keys=False, allow_unicode=True))
    return replaced


def _leading_comments(text: str) -> int:
    lines = text.splitlines()
    n = 0
    while n < len(lines) and (lines[n].startswith("#") or not lines[n].strip()):
        n += 1
    return n


def install(client: str, server: Server, repo: Path | None = None, classification: str = "INTERNAL") -> dict[str, Any]:
    """Register `server` with `client` in `repo`'s config, or in the user config when `repo` is None;
    returns where it went and whether it replaced an entry."""
    if client not in CLIENTS:
        raise InstallError(f"unknown client {client!r}: one of {', '.join(CLIENTS)}")
    if repo is not None:
        if not repo.is_dir():
            raise InstallError(f"no directory {repo}")
        if client == "claude":
            path = repo / ".mcp.json"
            return {"client": client, "path": str(path), "replaced": _set_json(path, ("mcpServers",), server.name, claude_entry(server))}
        if client == "opencode":
            path = opencode_path(repo)
            return {"client": client, "path": str(path), "replaced": _set_json(path, ("mcp",), server.name, opencode_entry(server))}
        raise InstallError(f"{client} has no per-repo MCP config; use --user")
    if client == "claude":
        if not shutil.which("claude"):
            raise InstallError("the claude command is not on PATH")
        replaced = subprocess.run(["claude", "mcp", "remove", server.name, "-s", "user"], capture_output=True).returncode == 0
        added = subprocess.run(claude_add_args(server, "user"), capture_output=True, text=True)
        if added.returncode:
            raise InstallError(added.stderr or added.stdout)
        return {"client": client, "scope": "user", "replaced": replaced}
    if client == "pi":
        path = pi_path()
        return {"client": client, "path": str(path), "replaced": _set_json(path, ("mcpServers",), server.name, pi_entry(server))}
    if client == "opencode":
        path = opencode_path()
        return {"client": client, "path": str(path), "replaced": _set_json(path, ("mcp",), server.name, opencode_entry(server))}
    if classification not in TRIGGERFISH_LEVELS:
        raise InstallError(f"--classification must be one of {', '.join(TRIGGERFISH_LEVELS)}")
    path = triggerfish_path()
    replaced = _set_triggerfish(path, server.name, triggerfish_entry(server, classification))
    return {"client": client, "path": str(path), "classification": classification, "replaced": replaced}
