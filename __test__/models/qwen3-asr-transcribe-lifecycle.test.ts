import { existsSync } from 'node:fs';
import { resolve } from 'node:path';

import { Qwen3AsrModel, type Qwen3AsrResult } from '@mlx-node/asr';
import { readWav } from '@mlx-node/tts';
import { describe, expect, it } from 'vite-plus/test';

/**
 * A one-shot Qwen3-ASR transcribe replies while the decode step queued past
 * EOS can still be running on the GPU. The next request on the same model
 * thread and a dispose sent right after the reply both land behind that step.
 *
 * Opt-in: set QWEN3_ASR_MODEL_PATH to a Qwen3-ASR checkpoint directory.
 *   QWEN3_ASR_MODEL_PATH=<dir> vp test __test__/models/qwen3-asr-transcribe-lifecycle.test.ts
 */
const modelPath = process.env.QWEN3_ASR_MODEL_PATH;
const enabled = modelPath !== undefined && existsSync(resolve(modelPath, 'config.json'));
const fixture = resolve(import.meta.dirname, '../../examples/audio-ask-16k.wav');

async function transcribeFixture(model: Qwen3AsrModel): Promise<Qwen3AsrResult> {
  const wav = await readWav(fixture);
  expect(wav.channels).toBe(1);
  return model.startTranscription(wav.samples, { language: 'en', sampleRate: wav.sampleRate }).result();
}

describe.runIf(enabled)('Qwen3-ASR transcribe lifecycle', () => {
  it('back-to-back requests return the same ids', async () => {
    const model = await Qwen3AsrModel.load(modelPath!);
    try {
      const first = await transcribeFixture(model);
      const second = await transcribeFixture(model);
      const third = await transcribeFixture(model);
      expect(first.reachedMaxTokens).toBe(false);
      expect(first.tokenIds.length).toBeGreaterThan(1);
      expect(first.text.length).toBeGreaterThan(0);
      expect(second.tokenIds).toEqual(first.tokenIds);
      expect(third.tokenIds).toEqual(first.tokenIds);
      expect(second.text).toBe(first.text);
    } finally {
      await model.dispose();
    }
  });

  it('dispose right after a reply resolves and later requests fail cleanly', async () => {
    const reference = await Qwen3AsrModel.load(modelPath!);
    let expected: number[];
    try {
      expected = (await transcribeFixture(reference)).tokenIds;
    } finally {
      await reference.dispose();
    }

    const model = await Qwen3AsrModel.load(modelPath!);
    const result = await transcribeFixture(model);
    await model.dispose();
    expect(result.tokenIds).toEqual(expected);
    await expect(transcribeFixture(model)).rejects.toThrow(/disposed/);
  });
});
