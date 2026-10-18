// Deterministic hashing, as jev_graph_builder/ids.py: decision IDs here match the Python query path's.

/** Stable JSON: sorted keys, no whitespace (Python `canonical_json`). */
export function canonicalJson(value: unknown): string {
  if (value === null || value === undefined) return "null";
  if (Array.isArray(value)) return "[" + value.map(canonicalJson).join(",") + "]";
  if (typeof value === "object") {
    const obj = value as Record<string, unknown>;
    return "{" + Object.keys(obj).sort().map((k) => JSON.stringify(k) + ":" + canonicalJson(obj[k])).join(",") + "}";
  }
  return JSON.stringify(value);
}

const encoder = new TextEncoder();
const SEPARATOR = new Uint8Array([0x1f]);

/** sha256 over the parts, each followed by the unit separator (Python `sha256_hex`). */
export async function sha256Hex(...parts: unknown[]): Promise<string> {
  const chunks: Uint8Array[] = [];
  for (const part of parts) {
    chunks.push(encoder.encode(typeof part === "string" ? part : canonicalJson(part)), SEPARATOR);
  }
  const all = new Uint8Array(chunks.reduce((n, c) => n + c.length, 0));
  let offset = 0;
  for (const c of chunks) {
    all.set(c, offset);
    offset += c.length;
  }
  const digest = new Uint8Array(await crypto.subtle.digest("SHA-256", all));
  return Array.from(digest, (b) => b.toString(16).padStart(2, "0")).join("");
}
