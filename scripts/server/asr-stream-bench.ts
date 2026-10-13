/** Qwen3-ASR rolling-transcript revision timings on a real checkpoint. Run with oxnode. */
import { resolve } from 'node:path';

import type { Qwen3AsrResult } from '@mlx-node/core';
import { Qwen3AsrModel } from '@mlx-node/core';
import { readWav } from '@mlx-node/tts';

// Usage: oxnode scripts/server/asr-stream-bench.ts <converted-asr-checkpoint> [fixture.wav]
const modelPath = process.argv[2];
if (!modelPath) throw new Error('Usage: oxnode scripts/server/asr-stream-bench.ts <model-dir> [fixture.wav]');
const fixture = resolve(process.argv[3] ?? 'examples/audio-ask-16k.wav');
const chunkSeconds = Number(process.env.CHUNK_SECONDS ?? 1);
const model = await Qwen3AsrModel.load(modelPath);
const med = (values: number[]) => [...values].sort((a, b) => a - b)[Math.floor(values.length / 2)];
try {
  const wav = await readWav(fixture);
  const step = Math.round(wav.sampleRate * chunkSeconds);
  for (let round = 0; round < 2; round++) {
    const stream = await model.createStream({ sampleRate: wav.sampleRate, chunkSeconds });
    const revisions: Qwen3AsrResult[] = [];
    const wall: number[] = [];
    for (let offset = 0; offset < wav.samples.length; offset += step) {
      const start = performance.now();
      const result = await stream.feed(wav.samples.subarray(offset, offset + step));
      wall.push(performance.now() - start);
      if (result) revisions.push(result);
    }
    const start = performance.now();
    const final = await stream.finish();
    const finishMs = performance.now() - start;
    if (round === 0) continue;
    const row = {
      fixture,
      chunkSeconds,
      revisions: revisions.length,
      feedWallMedMs: Number(med(wall).toFixed(1)),
      feedWallMaxMs: Number(Math.max(...wall).toFixed(1)),
      featureMedMs: Number(med(revisions.map((r) => r.featureMs)).toFixed(1)),
      encoderMedMs: Number(med(revisions.map((r) => r.encoderMs)).toFixed(1)),
      prefillMedMs: Number(med(revisions.map((r) => r.prefillMs)).toFixed(1)),
      decodeMedMs: Number(med(revisions.map((r) => r.decodeMs)).toFixed(1)),
      totalMedMs: Number(med(revisions.map((r) => r.totalMs)).toFixed(1)),
      tokPerSec: Number(med(revisions.map((r) => r.tokensPerSecond).filter((v) => v > 0)).toFixed(1)),
      finishMs: Number(finishMs.toFixed(1)),
    };
    console.log(JSON.stringify(row));
    console.log('  final:', final.text.slice(0, 160));
  }
} finally {
  await model.dispose();
}
