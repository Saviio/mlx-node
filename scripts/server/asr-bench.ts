/** Qwen3-ASR one-shot stage timings (median of ROUNDS) on a real checkpoint. Run with oxnode. */
import { resolve } from 'node:path';

import type { Qwen3AsrResult } from '@mlx-node/core';
import { Qwen3AsrModel } from '@mlx-node/core';
import { readWav } from '@mlx-node/tts';

// Usage: oxnode scripts/server/asr-bench.ts <converted-asr-checkpoint> [fixture.wav ...]
const modelPath = process.argv[2];
if (!modelPath) throw new Error('Usage: oxnode scripts/server/asr-bench.ts <model-dir> [fixture.wav ...]');
const fixtures = process.argv.length > 3 ? process.argv.slice(3) : ['examples/audio-ask-16k.wav'];
const rounds = Number(process.env.ROUNDS ?? 3);
const model = await Qwen3AsrModel.load(modelPath);
const med = (values: number[]) => [...values].sort((a, b) => a - b)[Math.floor(values.length / 2)];
type Key = 'featureMs' | 'prefillMs' | 'decodeMs' | 'totalMs' | 'tokensPerSecond' | 'realTimeFactor';
const keys: Key[] = ['featureMs', 'prefillMs', 'decodeMs', 'totalMs', 'tokensPerSecond', 'realTimeFactor'];
try {
  for (const fixture of fixtures) {
    const wav = await readWav(resolve(fixture));
    const audio = wav.samples;
    const results: Qwen3AsrResult[] = [];
    await model.transcribe(audio, { sampleRate: wav.sampleRate });
    for (let i = 0; i < rounds; i++) results.push(await model.transcribe(audio, { sampleRate: wav.sampleRate }));
    const row: Record<string, string | number> = {
      fixture,
      audioS: results[0].audioSeconds.toFixed(2),
      tokens: results[0].tokenIds.length,
    };
    for (const key of keys) row[key] = Number(med(results.map((r) => r[key])).toFixed(1));
    console.log(JSON.stringify(row));
    console.log('  text:', results[0].text.slice(0, 100));
  }
} finally {
  await model.dispose();
}
