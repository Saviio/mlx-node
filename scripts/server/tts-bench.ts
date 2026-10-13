/** Qwen3-TTS synthesis timings (median of ROUNDS, fixed seed) on a real checkpoint. Run with oxnode. */
import { createHash } from 'node:crypto';

import { loadTtsModel } from '@mlx-node/tts';

// Usage: oxnode scripts/server/tts-bench.ts <checkpoint-dir> [voice] [text]
const modelPath = process.argv[2];
if (!modelPath) throw new Error('Usage: oxnode scripts/server/tts-bench.ts <model-dir> [voice] [text]');
const rounds = Number(process.env.ROUNDS ?? 3);
const model = await loadTtsModel(modelPath);
const voice = process.argv[3] ?? model.capabilities.voices[0];
const texts = process.argv[4]
  ? [process.argv[4]]
  : [
      'Hello. This is a local speech service.',
      'The quick brown fox jumps over the lazy dog, while the sleepy cat watches from the warm windowsill and the evening light fades slowly over the quiet town.',
    ];
const med = (values: number[]) => [...values].sort((a, b) => a - b)[Math.floor(values.length / 2)];
try {
  for (const text of texts) {
    const samples: Record<string, number[]> = { synthesisMs: [], wallMs: [], firstPcmMs: [], rtf: [] };
    let audioSeconds = 0;
    let hash = '';
    let finish = '';
    await model.synthesize(text, { voice, seed: 42 });
    for (let i = 0; i < rounds; i++) {
      const result = await model.synthesize(text, { voice, seed: 42 });
      samples.synthesisMs.push(result.stats.synthesisMs);
      samples.wallMs.push(result.stats.wallMs);
      samples.firstPcmMs.push(result.stats.firstPcmMs ?? 0);
      samples.rtf.push(result.stats.realTimeFactor ?? 0);
      audioSeconds = result.stats.audioSeconds;
      finish = result.stats.finishReason;
      hash = createHash('sha256').update(Buffer.from(result.samples.buffer)).digest('hex').slice(0, 12);
    }
    const row: Record<string, string | number> = {
      voice,
      chars: text.length,
      audioS: audioSeconds.toFixed(2),
      finish,
      hash,
    };
    for (const [key, values] of Object.entries(samples)) row[key] = Number(med(values).toFixed(key === 'rtf' ? 3 : 1));
    row.framesPerSec = Number(((audioSeconds * 12.5) / (med(samples.synthesisMs) / 1000)).toFixed(1));
    console.log(JSON.stringify(row));
  }
} finally {
  await model.dispose();
}
