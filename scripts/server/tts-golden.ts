/**
 * Qwen3-TTS golden PCM over fixed SeedTTS EN texts, in process (no server). Run with oxnode.
 *
 *   oxnode scripts/server/tts-golden.ts --cv-model <customvoice-checkpoint> --base-model <base-checkpoint>
 *     --clips <clip-dir> --out <new-dir> [--seed 42] [--texts 20] [--refs 3] [--voice ryan]
 *
 * <clip-dir> is the same SeedTTS export asr-golden-ids.ts reads (`en.jsonl`, wavs).
 *   cv-NN    CustomVoice preset voice on `target_text` of EN rows 0..texts-1.
 *   base-N   Base voice clone: the first `refs` EN rows with distinct reference wavs; each row's reference
 *            (wav + ref_text) is prepared like POST /v1/audio/voices, then its `target_text` is synthesized.
 * Every item uses the same fixed seed. Each PCM is written as raw float32 LE `<id>.f32`; `tts.jsonl` holds one
 * row per item with its sha256. The TTS API does not expose codec codes (TtsNativeChunk carries PCM only),
 * so no codes are recorded.
 */
import { execFileSync } from 'node:child_process';
import { createHash } from 'node:crypto';
import { existsSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join, resolve } from 'node:path';
import { parseArgs } from 'node:util';

import { WavReader, loadTtsModel, type TtsAudio, type TtsModel, type TtsVoice } from '@mlx-node/tts';

const { values } = parseArgs({
  options: {
    'cv-model': { type: 'string' },
    'base-model': { type: 'string' },
    clips: { type: 'string' },
    out: { type: 'string' },
    seed: { type: 'string', default: '42' },
    texts: { type: 'string', default: '20' },
    refs: { type: 'string', default: '3' },
    voice: { type: 'string', default: 'ryan' },
  },
});
const usage =
  'Usage: oxnode scripts/server/tts-golden.ts --cv-model <dir> --base-model <dir> --clips <dir> --out <new-dir> [--seed 42] [--texts 20] [--refs 3] [--voice ryan]';
if (!values['cv-model'] || !values['base-model'] || !values.clips || !values.out) throw new Error(usage);
const int = (raw: string, name: string, min: number): number => {
  const value = Number(raw);
  if (!Number.isSafeInteger(value) || value < min) throw new Error(`--${name} must be an integer >= ${min}`);
  return value;
};
const seed = int(values.seed, 'seed', 0);
const textCount = int(values.texts, 'texts', 1);
const refCount = int(values.refs, 'refs', 0);
const clipsDir = resolve(values.clips);
const outDir = resolve(values.out);
if (existsSync(outDir)) throw new Error(`${outDir} exists; pick a new output directory`);

interface ClipRow {
  index: number;
  id: string;
  ref_text: string;
  target_text: string;
  ref_audio: string;
}
const en = readFileSync(join(clipsDir, 'en.jsonl'), 'utf8')
  .split('\n')
  .filter((line) => line.trim())
  .map((line) => JSON.parse(line) as ClipRow);
const cvRows = en.slice(0, textCount);
if (cvRows.length !== textCount) throw new Error(`Need ${textCount} EN rows, found ${cvRows.length}`);
const refRows: ClipRow[] = [];
for (const row of en) {
  if (refRows.length === refCount) break;
  if (!refRows.some((picked) => picked.ref_audio === row.ref_audio)) refRows.push(row);
}
if (refRows.length !== refCount) throw new Error(`Need ${refCount} distinct references, found ${refRows.length}`);

// Mirrors audio-service.ts createVoice(): WavReader + mono chunks of 16384 frames.
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
const git = (args: string[]) => {
  try {
    return execFileSync('git', args, { encoding: 'utf8' }).trim();
  } catch {
    return null;
  }
};

mkdirSync(outDir, { recursive: true });
const rows: Record<string, unknown>[] = [];
let loaded: string[] = [];
async function record(
  id: string,
  kind: 'cv' | 'base',
  modelPath: string,
  voice: string,
  row: ClipRow,
  run: () => Promise<TtsAudio>,
) {
  const start = performance.now();
  const audio = await run();
  const wallMs = performance.now() - start;
  const bytes = Buffer.from(audio.samples.buffer, audio.samples.byteOffset, audio.samples.byteLength);
  const file = `${id}.f32`;
  writeFileSync(join(outDir, file), bytes);
  const entry = {
    id,
    kind,
    model: modelPath,
    voice,
    clip: row.id,
    seed,
    text: row.target_text,
    file,
    sampleRate: audio.sampleRate,
    channels: audio.channels,
    samples: audio.samples.length,
    audioSeconds: audio.stats.audioSeconds,
    finishReason: audio.stats.finishReason,
    pcmSha256: createHash('sha256').update(bytes).digest('hex'),
    synthesisMs: audio.stats.synthesisMs,
    firstPcmMs: audio.stats.firstPcmMs,
    wallMs,
  };
  rows.push(entry);
  console.log(
    JSON.stringify({ id, samples: entry.samples, finish: entry.finishReason, sha: entry.pcmSha256.slice(0, 16) }),
  );
}
async function withModel(path: string, run: (model: TtsModel) => Promise<void>) {
  const model = await loadTtsModel(path);
  try {
    await run(model);
    loaded = [...new Set([...loaded, ...mapped()])];
  } finally {
    await model.dispose();
  }
}

const startedAt = new Date().toISOString();
const cvPath = resolve(values['cv-model']);
await withModel(cvPath, async (model) => {
  const voice = model.capabilities.voices.find((v) => v.toLowerCase() === values.voice.toLowerCase());
  if (!voice) throw new Error(`Voice ${values.voice} not in ${JSON.stringify(model.capabilities.voices)}`);
  // Unrecorded warmup, as the bench client sends before timing.
  await model.synthesize(cvRows[0].target_text, { voice, seed });
  for (const [i, row] of cvRows.entries())
    await record(`cv-${String(i).padStart(2, '0')}`, 'cv', cvPath, voice, row, () =>
      model.synthesize(row.target_text, { voice, seed }),
    );
});
const basePath = resolve(values['base-model']);
if (refRows.length)
  await withModel(basePath, async (model) => {
    if (!model.capabilities.voiceCloning) throw new Error(`${basePath} does not support voice cloning`);
    const clone = async (row: ClipRow, use: (voice: TtsVoice) => Promise<TtsAudio>) => {
      const ref = await readClip(join(clipsDir, row.ref_audio));
      const voice = await model.prepareVoice({
        audio: { samples: ref.samples, sampleRate: ref.sampleRate, channels: 1 },
        transcript: row.ref_text,
      });
      try {
        return await use(voice);
      } finally {
        await voice.dispose();
      }
    };
    await clone(refRows[0], (voice) => model.synthesize(refRows[0].target_text, { voice, seed }));
    for (const [i, row] of refRows.entries())
      await record(`base-${i}`, 'base', basePath, `clone:${row.ref_audio}`, row, () =>
        clone(row, (voice) => model.synthesize(row.target_text, { voice, seed })),
      );
  });

const expected = textCount + refCount;
if (rows.length !== expected) throw new Error(`Produced ${rows.length} items, expected ${expected}`);
writeFileSync(join(outDir, 'tts.jsonl'), rows.map((row) => JSON.stringify(row)).join('\n') + '\n');
writeFileSync(
  join(outDir, 'meta.json'),
  JSON.stringify(
    {
      count: rows.length,
      cv: textCount,
      base: refCount,
      seed,
      voice: values.voice,
      clipsDir,
      startedAt,
      codes: 'not recorded: the TTS API exposes PCM only (TtsNativeChunk has no codec codes)',
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
console.log(`items=${rows.length} expected=${expected} out=${outDir}`);
