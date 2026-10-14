/**
 * Compare golden outputs from asr-golden-ids.ts or tts-golden.ts. Run with oxnode. Exit 1 on any mismatch.
 *
 *   oxnode scripts/server/compare-golden.ts asr <baseline.jsonl> <candidate.jsonl> [--expect N]
 *   oxnode scripts/server/compare-golden.ts tts <baseline-dir> <candidate-dir> [--expect N] [--min-snr dB]
 *
 * asr: counts must match each other, each file's `.meta.json` count (when present) and --expect. Rows are matched
 *      by id; token_ids must be equal. Every differing clip is listed with both texts.
 * tts: items are matched by id. Each row needs a 64-hex pcmSha256 that equals sha256 of its .f32 file on disk;
 *      missing or stale evidence is a mismatch. PCM sha256 must be equal. A differing item prints its PCM SNR in dB
 *      (baseline as signal). With --min-snr, an item with equal length and SNR >= the bar counts as a pass.
 * Comparing a path with itself is refused (exit 2).
 */
import { createHash } from 'node:crypto';
import { existsSync, readFileSync, realpathSync, statSync } from 'node:fs';
import { basename, join } from 'node:path';
import { parseArgs } from 'node:util';

const { values, positionals } = parseArgs({
  allowPositionals: true,
  options: { expect: { type: 'string' }, 'min-snr': { type: 'string' } },
});
const [mode, left, right] = positionals;
if (!['asr', 'tts'].includes(mode) || !left || !right || positionals.length !== 3) {
  console.error('Usage: compare-golden.ts <asr|tts> <baseline> <candidate> [--expect N] [--min-snr dB]');
  process.exit(2);
}
if (!existsSync(left) || !existsSync(right)) {
  console.error(`Missing input: ${existsSync(left) ? right : left}`);
  process.exit(2);
}
const same = (x: string, y: string) => {
  const [sx, sy] = [statSync(x), statSync(y)];
  return realpathSync(x) === realpathSync(y) || (sx.dev === sy.dev && sx.ino === sy.ino);
};
if (same(left, right) || (mode === 'tts' && same(join(left, 'tts.jsonl'), join(right, 'tts.jsonl')))) {
  console.error(`Refusing to compare ${left} with itself`);
  process.exit(2);
}
const expect = values.expect === undefined ? undefined : Number(values.expect);
if (expect !== undefined && (!Number.isSafeInteger(expect) || expect < 1)) {
  console.error('--expect must be a positive integer');
  process.exit(2);
}
const minSnr = values['min-snr'] === undefined ? undefined : Number(values['min-snr']);
if (minSnr !== undefined && !Number.isFinite(minSnr)) {
  console.error('--min-snr must be a number');
  process.exit(2);
}

const readJsonl = (path: string) =>
  readFileSync(path, 'utf8')
    .split('\n')
    .filter((line) => line.trim())
    .map((line) => JSON.parse(line) as Record<string, unknown>);
const failures: string[] = [];
const fail = (message: string) => failures.push(message);
function byId(rows: Record<string, unknown>[], label: string): Map<string, Record<string, unknown>> {
  const map = new Map<string, Record<string, unknown>>();
  for (const row of rows) {
    const id = row.id;
    if (typeof id !== 'string') fail(`${label}: row without a string id`);
    else if (map.has(id)) fail(`${label}: duplicate id ${id}`);
    else map.set(id, row);
  }
  return map;
}
function checkCounts(a: number, b: number, metaCounts: [string, number | undefined][]) {
  if (a !== b) fail(`count differs: baseline ${a}, candidate ${b}`);
  for (const [label, count] of metaCounts)
    if (count !== undefined && count !== (label.startsWith('baseline') ? a : b))
      fail(`${label} says ${count} items, file has ${label.startsWith('baseline') ? a : b}`);
  if (expect !== undefined && (a !== expect || b !== expect)) fail(`expected ${expect} items, got ${a} and ${b}`);
}
const metaCount = (path: string): number | undefined =>
  existsSync(path) ? (JSON.parse(readFileSync(path, 'utf8')) as { count?: number }).count : undefined;

let compared = 0;
let equal = 0;
let snrPass = 0;
if (mode === 'asr') {
  const a = readJsonl(left);
  const b = readJsonl(right);
  checkCounts(a.length, b.length, [
    ['baseline meta', metaCount(`${left}.meta.json`)],
    ['candidate meta', metaCount(`${right}.meta.json`)],
  ]);
  const ma = byId(a, 'baseline');
  const mb = byId(b, 'candidate');
  for (const id of mb.keys()) if (!ma.has(id)) fail(`candidate-only id ${id}`);
  for (const [id, ra] of ma) {
    const rb = mb.get(id);
    if (!rb) {
      fail(`baseline-only id ${id}`);
      continue;
    }
    compared++;
    if (!Array.isArray(ra.token_ids) || !Array.isArray(rb.token_ids)) {
      fail(`${id}: token_ids missing`);
      continue;
    }
    if (JSON.stringify(ra.token_ids) === JSON.stringify(rb.token_ids)) {
      equal++;
      continue;
    }
    const ta = ra.token_ids as number[];
    const tb = rb.token_ids as number[];
    let at = 0;
    while (at < ta.length && at < tb.length && ta[at] === tb[at]) at++;
    fail(
      `${id}: token_ids differ at ${at} (len ${ta.length} vs ${tb.length})\n    baseline : ${JSON.stringify(ra.text)}\n    candidate: ${JSON.stringify(rb.text)}`,
    );
  }
} else {
  const a = readJsonl(join(left, 'tts.jsonl'));
  const b = readJsonl(join(right, 'tts.jsonl'));
  checkCounts(a.length, b.length, [
    ['baseline meta', metaCount(join(left, 'meta.json'))],
    ['candidate meta', metaCount(join(right, 'meta.json'))],
  ]);
  const ma = byId(a, 'baseline');
  const mb = byId(b, 'candidate');
  for (const id of mb.keys()) if (!ma.has(id)) fail(`candidate-only id ${id}`);
  // The row sha is a claim; the gate is the bytes on disk. Re-hash every file and fail on missing or stale evidence.
  const pcm = (dir: string, label: string, id: string, row: Record<string, unknown>): Float32Array | undefined => {
    const sha = row.pcmSha256;
    if (typeof sha !== 'string' || !/^[0-9a-f]{64}$/.test(sha)) {
      fail(`${id}: ${label} row has no 64-hex pcmSha256`);
      return undefined;
    }
    if (typeof row.file !== 'string' || !row.file || basename(row.file) !== row.file) {
      fail(`${id}: ${label} row has no plain file name`);
      return undefined;
    }
    const path = join(dir, row.file);
    if (!existsSync(path)) {
      fail(`${id}: ${label} PCM file missing: ${path}`);
      return undefined;
    }
    const bytes = readFileSync(path);
    const actual = createHash('sha256').update(bytes).digest('hex');
    if (actual !== sha) {
      fail(`${id}: ${label} PCM file sha ${actual} does not match row pcmSha256 ${sha}`);
      return undefined;
    }
    if (bytes.byteLength % 4 !== 0 || (row.samples !== undefined && row.samples !== bytes.byteLength / 4)) {
      fail(`${id}: ${label} PCM file has ${bytes.byteLength} bytes, row says ${String(row.samples)} samples`);
      return undefined;
    }
    return new Float32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 4);
  };
  for (const [id, ra] of ma) {
    const rb = mb.get(id);
    if (!rb) {
      fail(`baseline-only id ${id}`);
      continue;
    }
    compared++;
    if (ra.text !== rb.text || ra.seed !== rb.seed || ra.voice !== rb.voice) {
      fail(`${id}: inputs differ (text, seed or voice)`);
      continue;
    }
    const xa = pcm(left, 'baseline', id, ra);
    const xb = pcm(right, 'candidate', id, rb);
    if (!xa || !xb) continue;
    if (ra.pcmSha256 === rb.pcmSha256) {
      equal++;
      continue;
    }
    const n = Math.min(xa.length, xb.length);
    let signal = 0;
    let noise = 0;
    for (let i = 0; i < n; i++) {
      signal += xa[i] * xa[i];
      noise += (xb[i] - xa[i]) ** 2;
    }
    const snr = noise === 0 ? Infinity : 10 * Math.log10(signal / noise);
    const line = `${id}: sha differs, samples ${xa.length} vs ${xb.length}, SNR ${snr.toFixed(2)} dB over ${n} samples`;
    if (minSnr !== undefined && xa.length === xb.length && snr >= minSnr) {
      snrPass++;
      console.log(`${line} (>= ${minSnr} dB, pass)`);
    } else fail(line);
  }
}
if (compared === 0) fail('nothing compared');
for (const message of failures) console.log(`MISMATCH ${message}`);
console.log(
  `${mode}: compared=${compared} equal=${equal}${minSnr === undefined ? '' : ` snr_pass=${snrPass}`} mismatches=${failures.length} baseline=${left} candidate=${right}`,
);
process.exit(failures.length ? 1 : 0);
