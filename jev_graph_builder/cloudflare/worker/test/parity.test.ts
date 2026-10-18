// Expected values come from the Python query path (tests/integration/test_cloudflare_parity.py).
import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";

import { sha256Hex } from "../src/hash";
import { buildState, Estimator, gate, topLevel } from "../src/jev";

const path = process.env.JGB_PARITY_FIXTURES;

describe.skipIf(!path)("parity with the Python query path", () => {
  const f = path ? JSON.parse(readFileSync(path, "utf8")) : {};
  const est = new Estimator(f.bundle?.chars_per_token ?? 1);

  it("builds the same state and decision ID", async () => {
    for (const c of f.states) {
      const qs = f.bundle.question_sets[c.qs];
      const state = buildState(qs, c.inputs, est, f.bundle.untrusted_field);
      expect(state, c.qs).toEqual(c.state);
      const id = await sha256Hex("query", "subject", qs.ref, await sha256Hex(state), qs.questions_hash, "");
      expect(id, c.qs).toBe(c.decision_id);
    }
  });

  it("gates the same way", () => {
    for (const c of f.gates) expect(gate(f.bundle.question_sets[c.qs], c.answers), JSON.stringify(c)).toBe(c.outcome);
  });

  it("reads the same score level", () => {
    for (const c of f.levels) expect(topLevel(c.answer)).toBe(c.level);
  });
});
