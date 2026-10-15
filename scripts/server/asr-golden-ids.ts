/**
 * Qwen3-ASR golden token ids over a fixed SeedTTS clip set, in process (no server). Run with oxnode.
 *
 *   oxnode scripts/server/asr-golden-ids.ts --model <asr-checkpoint> --clips <clip-dir> --set en200 --out <file.jsonl>
 *     [--concurrency N] [--max-tokens N] [--warmup N] [--force]
 *
 * <clip-dir> holds `sets.json` plus `<split>.jsonl` rows ({index, id, ref_text, target_text, ref_audio}) in the
 * bench client's order. Sets: en200 / zh200 = first 200 rows, en1088 / zh2020 = whole split. The script recomputes
 * the client's evaluation-input fingerprint and refuses a set that does not match `sets.json`.
 *
 * Requests mirror the server path: WAV read like audio-service.ts, `startTranscription`, `language` = split,
 * no `maxTokens` unless `--max-tokens` is given (the server's non-streaming default). Decoding is greedy (temperature 0).
 * `--concurrency N` keeps N transcriptions in flight against ONE loaded model.
 * Rows are written in set order. Run metadata goes to `<out>.meta.json`.
 */
import { execFileSync } from 'node:child_process';
import { createHash } from 'node:crypto';
import { existsSync, readFileSync, writeFileSync } from 'node:fs';
import { join, resolve } from 'node:path';
import { parseArgs } from 'node:util';

import { Qwen3AsrModel, type Qwen3AsrResult } from '@mlx-node/asr';
import { WavReader } from '@mlx-node/tts';

const { values } = parseArgs({
  options: {
    model: { type: 'string' },
    clips: { type: 'string' },
    set: { type: 'string' },
    out: { type: 'string' },
    concurrency: { type: 'string', default: '1' },
    'max-tokens': { type: 'string' },
    warmup: { type: 'string', default: '1' },
    force: { type: 'boolean', default: false },
  },
});
const usage =
  'Usage: oxnode scripts/server/asr-golden-ids.ts --model <dir> --clips <dir> --set <en200|zh200|en1088|zh2020> --out <file.jsonl> [--concurrency N] [--max-tokens N] [--warmup N] [--force]';
if (!values.model || !values.clips || !values.set || !values.out) throw new Error(usage);
const positiveInt = (raw: string, name: string, min: number): number => {
  const value = Number(raw);
  if (!Number.isSafeInteger(value) || value < min) throw new Error(`--${name} must be an integer >= ${min}`);
  return value;
};
const concurrency = positiveInt(values.concurrency, 'concurrency', 1);
const maxTokens = values['max-tokens'] === undefined ? undefined : positiveInt(values['max-tokens'], 'max-tokens', 1);
const warmup = positiveInt(values.warmup, 'warmup', 0);
const clipsDir = resolve(values.clips);
const outPath = resolve(values.out);
if (existsSync(outPath) && !values.force) throw new Error(`${outPath} exists; pass --force to overwrite`);

interface ClipRow {
  index: number;
  id: string;
  ref_text: string;
  target_text: string;
  ref_audio: string;
}
interface SetInfo {
  split: string;
  count: number;
  fingerprint: string;
}
const sets = (JSON.parse(readFileSync(join(clipsDir, 'sets.json'), 'utf8')) as { sets: Record<string, SetInfo> }).sets;
const info = sets[values.set];
if (!info) throw new Error(`Unknown set ${values.set}; known: ${Object.keys(sets).join(', ')}`);
const clips = readFileSync(join(clipsDir, `${info.split}.jsonl`), 'utf8')
  .split('\n')
  .filter((line) => line.trim())
  .map((line) => JSON.parse(line) as ClipRow)
  .slice(0, info.count);
if (clips.length !== info.count) throw new Error(`Set ${values.set} needs ${info.count} clips, found ${clips.length}`);

// Same digest as sglang-omni benchmarks/eval/benchmark_asr_seedtts.py::_evaluation_input_sha256.
const fingerprint = (() => {
  const digest = createHash('sha256').update('seedtts-evaluation-input-v1\0');
  const field = (bytes: Buffer) => {
    const size = Buffer.alloc(8);
    size.writeBigUInt64BE(BigInt(bytes.length));
    digest.update(size).update(bytes);
  };
  for (const clip of clips) {
    for (const value of [clip.id, clip.ref_text, clip.target_text]) field(Buffer.from(value, 'utf8'));
    field(readFileSync(join(clipsDir, clip.ref_audio)));
  }
  return digest.digest('hex');
})();
if (fingerprint !== info.fingerprint)
  throw new Error(`Set ${values.set} fingerprint ${fingerprint} != sets.json ${info.fingerprint}`);

// Mirrors audio-service.ts transcribe(): WavReader + mono chunks of 16384 frames.
async function readClip(path: string): Promise<{ samples: Float32Array; sampleRate: number }> {
  const reader = await WavReader.open(path);
  try {
    const samples = new Float32Array(reader.frames);
    let offset = 0;
    for await (const chunk of reader.chunks(16384, true)) {
      samples.set(chunk, offset);
      offset += chunk.length;
    }
    return { samples, sampleRate: reader.sampleRate };
  } finally {
    await reader.close();
  }
}

const git = (args: string[]) => {
  try {
    return execFileSync('git', args, { encoding: 'utf8' }).trim();
  } catch {
    return null;
  }
};
const mapped = () => {
  try {
    return execFileSync('lsof', ['-p', String(process.pid)], { encoding: 'utf8' })
      .split('\n')
      .filter((line) => /\.node$|\.metallib$/.test(line))
      .map((line) => line.split(/\s+/).slice(8).join(' '));
  } catch {
    return [];
  }
};

const language = info.split;
const options = { language, ...(maxTokens === undefined ? {} : { maxTokens }) };
const startedAt = new Date().toISOString();
const loadStart = performance.now();
const model = await Qwen3AsrModel.load(resolve(values.model));
const loadMs = performance.now() - loadStart;
const slots: (Record<string, unknown> | undefined)[] = Array.from({ length: clips.length }, () => undefined);
let wallMs = 0;
let loaded: string[] = [];
try {
  if (warmup > 0) {
    const first = await readClip(join(clipsDir, clips[0].ref_audio));
    for (let i = 0; i < warmup; i++)
      await model.startTranscription(first.samples, { ...options, sampleRate: first.sampleRate }).result();
  }
  let next = 0;
  let done = 0;
  const runStart = performance.now();
  const worker = async () => {
    for (;;) {
      const index = next++;
      if (index >= clips.length) return;
      const clip = clips[index];
      const audio = await readClip(join(clipsDir, clip.ref_audio));
      const callStart = performance.now();
      const result: Qwen3AsrResult = await model
        .startTranscription(audio.samples, { ...options, sampleRate: audio.sampleRate })
        .result();
      const callMs = performance.now() - callStart;
      slots[index] = {
        id: clip.id,
        index: clip.index,
        token_ids: result.tokenIds,
        text: result.text,
        language: result.language ?? null,
        reachedMaxTokens: result.reachedMaxTokens,
        audioSeconds: result.audioSeconds,
        totalMs: result.totalMs,
        featureMs: result.featureMs,
        prefillMs: result.prefillMs,
        decodeMs: result.decodeMs,
        callMs,
      };
      if (++done % 50 === 0) process.stderr.write(`[asr-golden] ${done}/${clips.length}\n`);
    }
  };
  await Promise.all(Array.from({ length: Math.min(concurrency, clips.length) }, worker));
  wallMs = performance.now() - runStart;
  loaded = mapped();
} finally {
  await model.dispose();
}

const rows = slots.filter((row): row is Record<string, unknown> => row !== undefined);
if (rows.length !== info.count) throw new Error(`Produced ${rows.length} rows, expected ${info.count}`);
writeFileSync(outPath, rows.map((row) => JSON.stringify(row)).join('\n') + '\n');
const totals = rows.map((row) => row.totalMs as number).sort((a, b) => a - b);
const sum = totals.reduce((a, b) => a + b, 0);
const pick = (p: number) => totals[Math.min(totals.length - 1, Math.floor((totals.length - 1) * p))];
const summary = {
  set: values.set,
  clips: rows.length,
  concurrency,
  wallS: Number((wallMs / 1000).toFixed(3)),
  clipsPerS: Number((rows.length / (wallMs / 1000)).toFixed(3)),
  totalMsMean: Number((sum / totals.length).toFixed(3)),
  totalMsP50: Number(pick(0.5).toFixed(3)),
  totalMsP95: Number(pick(0.95).toFixed(3)),
  reachedMaxTokens: rows.filter((row) => row.reachedMaxTokens).length,
};
writeFileSync(
  `${outPath}.meta.json`,
  JSON.stringify(
    {
      ...summary,
      count: info.count,
      fingerprint,
      clipsDir,
      split: info.split,
      model: resolve(values.model),
      options: { ...options, warmup },
      startedAt,
      loadMs,
      cwd: process.cwd(),
      gitHead: git(['rev-parse', 'HEAD']),
      gitDirtyFiles: git(['status', '--porcelain'])?.split('\n').filter(Boolean).length ?? null,
      addonDir: process.env.MLX_ADDON_DIR ?? null,
      mapped: loaded,
      node: process.version,
    },
    null,
    1,
  ) + '\n',
);
console.log(JSON.stringify(summary));
console.log(`clips=${rows.length} set=${values.set} size=${info.count}`);
