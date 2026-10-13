/** Opt-in real Base checkpoint / OpenAI Node SDK smoke. No model downloads. */
import assert from 'node:assert/strict';
import { mkdtemp, readFile, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';

import { audioWavHeader } from '@mlx-node/core';
import { readWav } from '@mlx-node/tts';
import OpenAI from 'openai';

import { createInferenceHost } from '../../packages/server/src/host/index.js';

const [models, reference, transcriptFile] = process.argv.slice(2);
assert.ok(
  models && reference && transcriptFile,
  'Usage: oxnode scripts/server/audio-voices-smoke.ts MODELS_DIR REFERENCE.wav TRANSCRIPT.txt',
);
const model = process.env.BASE_MODEL ?? 'Base-8bit';
const audio = await readFile(resolve(reference));
const transcript = await readFile(resolve(transcriptFile), 'utf8');
const logDir = await mkdtemp(join(tmpdir(), 'mlx-voices-smoke-'));
const host = await createInferenceHost({
  modelsDir: resolve(models),
  model,
  port: 0,
  disableStore: true,
  sweepOrphanTempRoots: false,
  logDir,
});
const client = new OpenAI({ baseURL: `${host.url}/v1`, apiKey: 'local', maxRetries: 0, timeout: 120_000 });
interface Voice {
  id: string;
  status: string;
  persistent: boolean;
}
const created: string[] = [];
async function prepare() {
  const form = new FormData();
  form.set('audio_sample', new Blob([audio]), 'reference.wav');
  form.set('model', model);
  form.set('transcript', transcript);
  form.set('name', 'SDK smoke');
  // This installed SDK has no voices resource yet; its generic methods use
  // the same transport, error mapping and multipart body support.
  const voice = await client.post<Voice>('/audio/voices', { body: form });
  assert.equal(voice.status, 'active');
  assert.equal(voice.persistent, false);
  created.push(voice.id);
  return voice;
}
try {
  await host.server.withSuspendedDrains(async () => {
    const tiny = Buffer.concat([audioWavHeader(96000, 1, 2), Buffer.alloc(2)]);
    await assert.rejects(
      client.post('/audio/voices', {
        body: tiny,
        headers: { 'Content-Type': 'audio/wav' },
        query: { model, name: 'Too short', transcript: 'Hello' },
      }),
      (error: unknown) =>
        error instanceof OpenAI.APIError &&
        error.status === 400 &&
        error.code === 'invalid_audio' &&
        error.param === 'audio_sample',
    );
    console.log('PASS checkpoint reference validation returns 400 via SDK');
    const voice = await prepare();
    const current = await client.get<Voice>(`/audio/voices/${voice.id}`);
    assert.deepEqual(current, voice);
    console.log('PASS prepare / query via SDK');
    const input = {
      model,
      voice: { id: voice.id },
      input: process.env.SPEECH_TEXT ?? '你好，这是一段声音测试。',
      sampling: { seed: 42, temperature: 0, max_duration_seconds: 8 },
    };
    const wav = await client.audio.speech.create({ ...input, response_format: 'wav' });
    const bytes = Buffer.from(await wav.arrayBuffer());
    assert.equal(bytes.toString('ascii', 0, 4), 'RIFF');
    assert.equal(bytes.readUInt32LE(4), bytes.length - 8);
    assert.ok(bytes.length > 44);
    assert.ok(Number(wav.headers.get('x-sample-rate')) > 0);
    const pcm = await client.audio.speech.create({ ...input, response_format: 'pcm' });
    assert.deepEqual(Buffer.from(await pcm.arrayBuffer()), bytes.subarray(44));
    console.log('PASS cloned WAV / PCM deterministic parity via SDK');
    const binding = host.server.registry.audio.get(model);
    assert.ok(binding?.kind === 'tts');
    const prepared = await binding.model.prepareVoice({ audio: await readWav(reference), transcript });
    const stream = binding.model.synthesizeStream(input.input.repeat(6), {
      voice: prepared,
      seed: 42,
      temperature: 0,
      maxDurationSeconds: 8,
      chunkDurationMs: 80,
      audioBufferSeconds: 0.08,
    });
    void stream.completed.catch(() => {});
    let timeout: ReturnType<typeof setTimeout> | undefined;
    try {
      const first = await stream[Symbol.asyncIterator]().next();
      assert.ok(!first.done && first.value.samples.length > 0);
      // Intentionally stop draining PCM. A status query queued behind Generate
      // would deadlock once its bounded output fills. Always cancel and await
      // actual completion in finally, even when the diagnostic timeout wins.
      const status = await Promise.race([
        prepared.status(),
        new Promise<never>((_, reject) => {
          timeout = setTimeout(() => reject(new Error('Voice status blocked behind PCM output')), 5000);
        }),
      ]);
      assert.equal(status, 'active');
      console.log('PASS native status while PCM consumption is paused');
    } finally {
      clearTimeout(timeout);
      stream.cancel();
      await stream.completed.catch(() => {});
      await prepared.dispose();
    }
    await client.delete(`/audio/voices/${voice.id}`);
    assert.equal((await client.get<Voice>(`/audio/voices/${voice.id}`)).status, 'released');
    await assert.rejects(
      client.audio.speech.create({ ...input, response_format: 'pcm' }),
      (error: unknown) =>
        error instanceof OpenAI.APIError && error.status === 410 && error.code === 'voice_unavailable',
    );
    console.log('PASS release / stale synthesis rejection via SDK');
    const first = await prepare();
    // Observe native policy, not a second implementation of its capacity.
    for (let attempt = 0; attempt < 32; attempt++) {
      await prepare();
      if ((await client.get<Voice>(`/audio/voices/${first.id}`)).status === 'evicted') break;
    }
    assert.equal((await client.get<Voice>(`/audio/voices/${first.id}`)).status, 'evicted');
    console.log('PASS authoritative native cache eviction via SDK');
    await host.server.registry.audio.disposeAll();
    assert.equal((await client.get<Voice>(`/audio/voices/${created.at(-1)}`)).status, 'evicted');
    console.log('PASS model disposal invalidates HTTP voice resources');
  });
} finally {
  await host.close();
  await rm(logDir, { recursive: true, force: true });
}
