// A web-search query (Postgres `websearch_to_tsquery` syntax) as an SQLite FTS5 MATCH expression:
// words are ANDed, "quoted text" is a phrase, `or` between terms is OR, and a leading `-` excludes a term.

const WORD = /[\p{L}\p{N}_]+/gu;

function phrase(text: string): string | null {
  const words = text.toLowerCase().match(WORD);
  return words ? `"${words.join(" ")}"` : null;
}

export function ftsQuery(query: string): string | null {
  const tokens = query.match(/-?"[^"]*"?|\S+/g) ?? [];
  const groups: string[][] = [[]]; // OR of ANDed groups
  const excluded: string[] = [];
  let pendingOr = false;
  for (const raw of tokens) {
    if (raw.toLowerCase() === "or") {
      pendingOr = groups[groups.length - 1].length > 0;
      continue;
    }
    const negated = raw.startsWith("-");
    const p = phrase((negated ? raw.slice(1) : raw).replace(/"/g, " "));
    if (!p) continue;
    if (negated) {
      excluded.push(p);
    } else if (pendingOr) {
      groups.push([p]);
    } else {
      groups[groups.length - 1].push(p);
    }
    pendingOr = false;
  }
  const positive = groups.filter((g) => g.length).map((g) => (g.length > 1 ? `(${g.join(" AND ")})` : g[0]));
  if (!positive.length) return null;
  const match = positive.length > 1 ? `(${positive.join(" OR ")})` : positive[0];
  return excluded.length ? `${match} NOT (${excluded.join(" OR ")})` : match;
}
