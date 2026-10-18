// Asking a rendered question set (bundle.json): state (jev/state.py), System One, answers, gating (jev/gating.py).

import { canonicalJson, sha256Hex } from "./hash";

export interface StateField {
  source: string;
  fields: string[] | null;
  optional: boolean;
  untrusted: boolean;
  truncate: string;
  max_tokens: number | null;
}

export interface QuestionSet {
  ref: string;
  questions_hash: string;
  wire: Record<string, Record<string, unknown>>;
  keys: Record<string, { logical: string; type: string; levels: number | null }>;
  state_template: Record<string, StateField>;
  accept_when: Condition | null;
  threshold: number;
  leaf_thresholds: Record<string, number>;
}

export interface SystemOne {
  url: string;
  model: string;
  field_map: { model: string; state: string; questions: string; answers: string };
  response_root: string | null;
  auth_header: string;
  auth_scheme: string;
  max_questions: number | null;
  context_tokens: number;
}

export interface Retry {
  max_retries: number;
  backoff_initial_s: number;
  backoff_max_s: number;
  http_statuses: number[];
}

type Condition = { all?: Condition[]; any?: Condition[]; not?: Condition; q?: string; is?: string; threshold?: number;
  eq?: string; in?: string[]; not_in?: string[]; level_in?: number[]; if_asked?: boolean };

export type Answer = Record<string, any>;

export interface Decision {
  decision_id: string;
  subject_kind: string;
  subject_id: string;
  question_set: string;
  outcome: "accept" | "reject";
  answers: Record<string, Answer>;
  threshold: number;
}

export const ACCEPT = "accept";
export const REJECT = "reject";
const CREDITS_EXHAUSTED = 402;

/** Characters per token, counted in code points as Python counts string length. */
export class Estimator {
  constructor(readonly charsPerToken: number) {}
  chars(tokens: number): number {
    return Math.floor(tokens * this.charsPerToken);
  }
  truncate(s: string, maxTokens: number, rule: string): string {
    const cps = Array.from(s);
    const max = this.chars(maxTokens);
    if (cps.length <= max) return s;
    if (rule === "head") return cps.slice(0, max).join("");
    if (rule === "tail") return cps.slice(cps.length - max).join("");
    const half = Math.floor(max / 2);
    return cps.slice(0, half).join("") + "\n…\n" + cps.slice(cps.length - half).join("");
  }
}

const MISSING = Symbol("missing");

function resolve(inputs: Record<string, unknown>, dotted: string): unknown {
  let node: unknown = inputs;
  for (const part of dotted.split(".")) {
    if (node !== null && typeof node === "object" && !Array.isArray(node) && part in (node as object)) {
      node = (node as Record<string, unknown>)[part];
    } else {
      return MISSING;
    }
  }
  return node;
}

function empty(v: unknown): boolean {
  return v === null || v === undefined || v === "" || (Array.isArray(v) && v.length === 0);
}

function pick(value: unknown, fields: string[] | null): unknown {
  if (!fields) return value;
  if (Array.isArray(value)) return value.map((v) => pick(v, fields));
  if (value !== null && typeof value === "object") {
    const obj = value as Record<string, unknown>;
    return Object.fromEntries(fields.filter((f) => f in obj && !empty(obj[f])).map((f) => [f, obj[f]]));
  }
  throw new Error(`cannot pick fields ${fields} from ${typeof value}`);
}

function truncateValue(est: Estimator, value: unknown, maxTokens: number | null, rule: string): unknown {
  if (maxTokens === null) return value;
  if (typeof value === "string") return est.truncate(value, maxTokens, rule);
  if (Array.isArray(value)) return value.map((v) => truncateValue(est, v, maxTokens, rule));
  if (value !== null && typeof value === "object") {
    return Object.fromEntries(Object.entries(value).map(([k, v]) => [k, truncateValue(est, v, maxTokens, rule)]));
  }
  return value;
}

/** Only the fields the template names, truncated by its rule; untrusted text wrapped in the data field. */
export function buildState(qs: QuestionSet, inputs: Record<string, unknown>, est: Estimator, dataField: string): Record<string, unknown> {
  const state: Record<string, unknown> = {};
  for (const [name, spec] of Object.entries(qs.state_template)) {
    let value = resolve(inputs, spec.source);
    if (value === MISSING || value === null || value === undefined) {
      if (spec.optional) continue;
      throw new Error(`${qs.ref}: state input \`${spec.source}\` missing`);
    }
    value = truncateValue(est, pick(value, spec.fields), spec.max_tokens, spec.truncate);
    state[name] = spec.untrusted ? { [dataField]: value } : value;
  }
  return state;
}

/** One stable shape per type (jev/questions.py `normalize_answer`). */
function normalize(raw: Answer, type: string, levels: number | null): Answer {
  if (type === "noul") return { type, p: Number(raw.noul) };
  if (type === "choice") return { type, choice: raw.choice, probabilities: raw.probabilities, confidence: raw.confidence ?? null };
  return { type, score: Number(raw.score), probabilities: raw.probabilities, confidence: raw.confidence ?? null, levels };
}

export function topLevel(answer: Answer): number {
  const probs: Record<string, number> = answer.probabilities ?? {};
  const levels = Object.keys(probs);
  if (levels.length === 0) return Math.round(Number(answer.score));
  return Number(levels.sort((a, b) => Number(a) - Number(b)).reduce((best, k) => (probs[k] > probs[best] ? k : best)));
}

const SKIPPED = Symbol("skipped");
type Truth = boolean | typeof SKIPPED;

function leaf(cond: Condition, answers: Record<string, Answer>, qsThreshold: number): Truth {
  const a = answers[cond.q!];
  if (a === undefined && cond.if_asked) return SKIPPED;
  if (a === undefined) throw new Error(`gating refers to \`${cond.q}\`, which has no answer`);
  if (a.type === "noul") {
    const t = cond.threshold ?? qsThreshold;
    return cond.is === "yes" ? a.p > t : 1 - a.p > t;
  }
  if (a.type === "choice") {
    if (cond.eq !== undefined) return a.choice === cond.eq;
    if (cond.in !== undefined) return cond.in.includes(a.choice);
    return !cond.not_in!.includes(a.choice);
  }
  return cond.level_in!.includes(topLevel(a));
}

function evaluate(cond: Condition, answers: Record<string, Answer>, qsThreshold: number): Truth {
  if (cond.all) {
    const vs = cond.all.map((c) => evaluate(c, answers, qsThreshold)).filter((v) => v !== SKIPPED);
    return vs.length ? vs.every(Boolean) : SKIPPED;
  }
  if (cond.any) {
    const vs = cond.any.map((c) => evaluate(c, answers, qsThreshold)).filter((v) => v !== SKIPPED);
    return vs.length ? vs.some(Boolean) : SKIPPED;
  }
  if (cond.not) {
    const r = evaluate(cond.not, answers, qsThreshold);
    return r === SKIPPED ? r : !r;
  }
  return leaf(cond, answers, qsThreshold);
}

/** Accept unless the question set's `accept_when` says otherwise (leaf thresholds are already in the bundle). */
export function gate(qs: QuestionSet, answers: Record<string, Answer>): "accept" | "reject" {
  if (!qs.accept_when) return ACCEPT;
  const withLeaves = (c: Condition): Condition =>
    c.q !== undefined ? { ...c, threshold: qs.leaf_thresholds[c.q] ?? c.threshold } :
    { ...c, all: c.all?.map(withLeaves), any: c.any?.map(withLeaves), not: c.not && withLeaves(c.not) };
  const r = evaluate(withLeaves(qs.accept_when), answers, qs.threshold);
  return r === true || r === SKIPPED ? ACCEPT : REJECT;
}

export class CreditsExhausted extends Error {}

export interface Caller {
  (state: unknown, questions: Record<string, unknown>): Promise<Record<string, Answer>>;
}

/** A System One call over HTTPS (any endpoint and key), or through the Workers AI binding. */
export function systemOneCaller(s1: SystemOne, retry: Retry, key: string | undefined, ai: Ai | null): Caller {
  const f = s1.field_map;
  const body = (state: unknown, questions: Record<string, unknown>) => ({ [f.model]: s1.model, [f.state]: state, [f.questions]: questions });
  if (ai) {
    const model = s1.url.replace(/^.*\/ai\/run\//, "");
    return async (state, questions) => {
      const out = (await ai.run(model as any, body(state, questions) as any)) as Record<string, any>;
      return out[f.answers];
    };
  }
  return async (state, questions) => {
    for (let attempt = 0; ; attempt++) {
      const resp = await fetch(s1.url, {
        method: "POST",
        headers: { "content-type": "application/json", ...(key ? { [s1.auth_header]: `${s1.auth_scheme} ${key}` } : {}) },
        body: JSON.stringify(body(state, questions)),
      });
      if (resp.status === CREDITS_EXHAUSTED) throw new CreditsExhausted(`System One ${resp.status}: ${await resp.text()}`);
      if (retry.http_statuses.includes(resp.status) && attempt < retry.max_retries) {
        await new Promise((r) => setTimeout(r, 1000 * Math.min(retry.backoff_max_s, retry.backoff_initial_s * 2 ** attempt)));
        continue;
      }
      if (!resp.ok) throw new Error(`System One ${resp.status}: ${(await resp.text()).slice(0, 500)}`);
      const data = (await resp.json()) as Record<string, any>;
      return (s1.response_root ? data[s1.response_root] : data)[f.answers];
    }
  };
}

/** Questions packed into as few calls as the context window and question cap allow (JevService._batches). */
function batches(s1: SystemOne, est: Estimator, state: unknown, wire: Record<string, unknown>): Record<string, unknown>[] {
  const size = (v: unknown) => Math.ceil(Array.from(typeof v === "string" ? v : canonicalJson(v)).length / est.charsPerToken);
  const room = s1.context_tokens - size(state);
  const out: Record<string, unknown>[] = [];
  let current: Record<string, unknown> = {};
  let used = 0;
  for (const [key, q] of Object.entries(wire)) {
    const n = size({ [key]: q });
    if (Object.keys(current).length && (used + n > room || Object.keys(current).length === (s1.max_questions ?? Infinity))) {
      out.push(current);
      current = {};
      used = 0;
    }
    current[key] = q;
    used += n;
  }
  if (Object.keys(current).length) out.push(current);
  return out;
}

export class Jev {
  constructor(private readonly call: Caller, private readonly s1: SystemOne, private readonly est: Estimator,
              private readonly dataField: string) {}

  async ask(qs: QuestionSet, inputs: Record<string, unknown>, subjectKind: string, subjectId: string): Promise<Decision> {
    const state = buildState(qs, inputs, this.est, this.dataField);
    const raw: Record<string, Answer> = {};
    for (const batch of batches(this.s1, this.est, state, qs.wire)) Object.assign(raw, await this.call(state, batch));
    const answers: Record<string, Answer> = {};
    for (const [key, info] of Object.entries(qs.keys)) {
      if (!(key in raw)) throw new Error(`${qs.ref}: no answer for ${key}`);
      answers[info.logical] = normalize(raw[key], info.type, info.levels);
    }
    const stateHash = await sha256Hex(state);
    return {
      decision_id: await sha256Hex(subjectKind, subjectId, qs.ref, stateHash, qs.questions_hash, ""),
      subject_kind: subjectKind, subject_id: subjectId, question_set: qs.ref,
      outcome: gate(qs, answers), answers, threshold: qs.threshold,
    };
  }
}
