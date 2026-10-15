/** Opt-in real-checkpoint comparison. Does not download models. Run with oxnode. */
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { mkdtemp, readFile, writeFile, rm } from 'node:fs/promises';
import { request } from 'node:http';
import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';

import type { Qwen3AsrResult } from '@mlx-node/core';
import { audioEncodePcm16 } from '@mlx-node/core';
import { readWav } from '@mlx-node/tts';

import type { AudioModel } from '../../packages/server/src/audio-registry.js';
import { loadAudioModel } from '../../packages/server/src/host/audio-models.js';
import { createInferenceHost } from '../../packages/server/src/host/index.js';

const modelsDir = resolve(process.argv[2] ?? '.cache/tts/models');
const fixture = resolve(process.argv[3] ?? 'examples/audio-ask-16k.wav');
const output = resolve(process.argv[4] ?? '.cache/serve-audio-benchmark.json');
const names = {
  tts: process.env.TTS_MODEL ?? 'CustomVoice-8bit',
  asr: process.env.ASR_MODEL ?? 'ASR-1.7B-hf-8bit',
  design: process.env.DESIGN_MODEL ?? 'VoiceDesign-1.7B-8bit',
};
const loaded = new Map<string, AudioModel>();
const logDir = await mkdtemp(join(tmpdir(), 'mlx-audio-benchmark-'));
const host = await createInferenceHost({
  modelsDir,
  model: names.tts,
  port: 0,
  disableStore: true,
  sweepOrphanTempRoots: false,
  logDir,
  loadAudioModel: async (entry) => {
    const model = await loadAudioModel(entry);
    loaded.set(entry.name, model);
    return model;
  },
});
const results: Record<string, unknown> = {
  node: process.version,
  arch: process.arch,
  modelsDir,
  names,
  rounds: 3,
  fixture,
  recordedAt: new Date().toISOString(),
};
const hash = (bytes: Buffer) => createHash('sha256').update(bytes).digest('hex');
const stats = (values: number[]) => {
  const sorted = [...values].sort((a, b) => a - b);
  return {
    p50: sorted[Math.ceil(sorted.length * 0.5) - 1],
    p95: sorted[Math.ceil(sorted.length * 0.95) - 1],
    samples: values,
  };
};
function pcm(body: Record<string, unknown>, incremental = false) {
  const start = performance.now();
  return new Promise<{
    bytes: Buffer;
    firstMs: number;
    wallMs: number;
    headers: Record<string, unknown>;
    trailers: Record<string, string | undefined>;
  }>((resolve, reject) => {
    const req = request(
      `${host.url}/v1/audio/speech${incremental ? '/stream' : ''}`,
      { method: 'POST', headers: { 'Content-Type': incremental ? 'application/x-ndjson' : 'application/json' } },
      (res) => {
        const chunks: Buffer[] = [];
        let firstMs = 0;
        res.on('data', (chunk) => {
          if (!chunks.length) {
            firstMs = performance.now() - start;
            if (incremental) {
              assert.equal(req.writableEnded, false);
              req.end();
            }
          }
          chunks.push(chunk);
        });
        res.on('error', reject);
        res.on('end', () => {
          const bytes = Buffer.concat(chunks);
          if (res.statusCode !== 200) return reject(new Error(`${res.statusCode}: ${bytes.toString()}`));
          resolve({ bytes, firstMs, wallMs: performance.now() - start, headers: res.headers, trailers: res.trailers });
        });
      },
    );
    req.on('error', reject);
    if (incremental) {
      const { input, ...config } = body;
      req.write(JSON.stringify({ type: 'config', ...config }) + '\n');
      req.write(JSON.stringify({ type: 'text', text: input }) + '\n');
      req.write(JSON.stringify({ type: 'flush' }) + '\n');
    } else req.end(JSON.stringify(body));
  });
}
const body = {
  model: names.tts,
  input: 'Hello. This is a local speech service.',
  voice: 'ryan',
  response_format: 'pcm',
  sampling: { seed: 42, temperature: 0, max_duration_seconds: 12 },
};
try {
  await host.server.withSuspendedDrains(async () => {
    const cold = await pcm(body);
    results.ttsCold = {
      firstMs: cold.firstMs,
      wallMs: cold.wallMs,
      headers: cold.headers,
      trailers: cold.trailers,
      bytes: cold.bytes.length,
    };
    const binding = loaded.get(names.tts)!;
    assert.equal(binding.kind, 'tts');
    if (binding.kind !== 'tts') throw new Error('TTS missing');
    const sdkFirst: number[] = [],
      sdkWall: number[] = [],
      httpFirst: number[] = [],
      httpWall: number[] = [];
    const sdkRtf: number[] = [],
      httpRtf: number[] = [];
    const parity: boolean[] = [];
    for (let i = 0; i < 3; i++) {
      const sdk = await host.server.modelWork.withInference(async () => {
        const start = performance.now();
        let first = 0;
        const chunks: Buffer[] = [];
        const source = binding.model.synthesizeStream(body.input, {
          voice: body.voice,
          seed: 42,
          temperature: 0,
          maxDurationSeconds: 12,
        });
        for await (const chunk of source) {
          if (!chunks.length) first = performance.now() - start;
          chunks.push(audioEncodePcm16(chunk.samples));
        }
        const s = await source.completed;
        return { first, wall: performance.now() - start, bytes: Buffer.concat(chunks), rtf: s.realTimeFactor! };
      });
      const http = await pcm(body);
      parity.push(hash(sdk.bytes) === hash(http.bytes));
      sdkFirst.push(sdk.first);
      sdkWall.push(sdk.wall);
      sdkRtf.push(sdk.rtf);
      httpFirst.push(http.firstMs);
      httpWall.push(http.wallMs);
      httpRtf.push(Number(http.trailers['x-realtime-factor']));
      console.log('TTS round', i + 1, { sdkMs: sdk.wall, httpMs: http.wallMs, exactPcm: parity[i] });
    }
    results.ttsWarm = {
      sdkFirst: stats(sdkFirst),
      sdkWall: stats(sdkWall),
      sdkRtf: stats(sdkRtf),
      httpFirst: stats(httpFirst),
      httpWall: stats(httpWall),
      httpRtf: stats(httpRtf),
      exactPcm: parity,
    };
    assert.ok(parity.every(Boolean), 'same-seed direct SDK / HTTP PCM differs');
    const duplex = await pcm(body, true);
    assert.equal(hash(duplex.bytes), hash(cold.bytes));
    results.duplex = { firstMs: duplex.firstMs, wallMs: duplex.wallMs, exactPcm: true, audioBeforeRequestEof: true };
    const faster = await pcm({ ...body, speed: 1.5 });
    const ratio = faster.bytes.length / cold.bytes.length;
    assert.ok(Math.abs(ratio - 1 / 1.5) < 0.03);
    results.speed = { speed: 1.5, durationRatio: ratio, trailers: faster.trailers };
    const wav = await readFile(fixture);
    const audio = await readWav(fixture);
    const transcribe = async (stream = false) => {
      const start = performance.now();
      const res = await fetch(`${host.url}/v1/audio/transcriptions?model=${names.asr}&stream=${stream}`, {
        method: 'POST',
        headers: { 'Content-Type': 'audio/wav' },
        body: wav,
      });
      const first = performance.now() - start,
        text = await res.text();
      assert.equal(res.status, 200, text);
      return { first, wall: performance.now() - start, text };
    };
    const asrCold = await transcribe();
    results.asrCold = asrCold;
    const asr = loaded.get(names.asr)!;
    if (asr.kind !== 'asr') throw new Error('ASR missing');
    const asrSdk: number[] = [],
      asrHttp: number[] = [];
    for (let i = 0; i < 3; i++) {
      const start = performance.now();
      const sdk: Qwen3AsrResult = await host.server.modelWork.withInference((): Promise<Qwen3AsrResult> =>
        asr.model.startTranscription(audio.samples, { sampleRate: audio.sampleRate }).result(),
      );
      asrSdk.push(performance.now() - start);
      const http = await transcribe();
      asrHttp.push(http.wall);
      assert.equal(JSON.parse(http.text).text, sdk.text);
      console.log('ASR round', i + 1, { sdkMs: asrSdk[i], httpMs: http.wall, text: sdk.text });
    }
    results.asrWarm = { sdk: stats(asrSdk), http: stats(asrHttp), duration: audio.samples.length / audio.sampleRate };
    const rolling = await transcribe(true);
    const revisions = rolling.text
      .split('\n')
      .filter((l) => l.startsWith('data: '))
      .map((l) => JSON.parse(l.slice(6)));
    assert.ok(revisions.length > 1);
    assert.ok(revisions.at(-1).is_final);
    assert.equal(revisions.at(-1).provisional, '');
    for (let i = 1; i < revisions.length; i++) assert.ok(revisions[i].revision > revisions[i - 1].revision);
    results.asrRolling = {
      firstMs: rolling.first,
      wallMs: rolling.wall,
      revisions: revisions.length,
      final: revisions.at(-1),
    };
    await host.server.modelWork.withInference(async () => {
      const task = asr.model.startTranscription(new Float32Array(16000 * 120), { sampleRate: 16000, maxTokens: 4096 });
      const done = task.result();
      void done.catch(() => {});
      await new Promise((resolve) => setTimeout(resolve, 10));
      const cancelAt = performance.now();
      task.cancel();
      await assert.rejects(done, /cancel/i);
      results.asrCancelMs = performance.now() - cancelAt;
    });
    await transcribe(); // cancelled model remains usable
    const [parallelTts, parallelAsr] = await Promise.all([pcm(body), transcribe()]);
    results.parallel = {
      ttsMs: parallelTts.wallMs,
      asrMs: parallelAsr.wall,
      residents: host.server.registry.audio.list().map((e) => e.id),
    };
    const design = await pcm({
      ...body,
      model: names.design,
      voice: { type: 'description' as const, description: 'A warm, clear adult female voice speaking English.' },
    });
    results.design = {
      firstMs: design.firstMs,
      wallMs: design.wallMs,
      bytes: design.bytes.length,
      headers: design.headers,
      trailers: design.trailers,
    };
    // Also exercise ordinary VoiceDesign EOS with its default sampler. The
    // greedy, per-segment 12s request above deliberately exposes truncation.
    const designBody = {
      ...body,
      model: names.design,
      input: 'Hello, welcome!',
      voice: { type: 'description' as const, description: 'A warm, clear adult female voice speaking English.' },
      sampling: { seed: 42, max_duration_seconds: 30 },
    };
    const designNormal = await pcm(designBody);
    assert.equal(designNormal.trailers['x-finish-reason'], 'eos');
    const designModel = loaded.get(names.design)!;
    if (designModel.kind !== 'tts') throw new Error('VoiceDesign missing');
    const designSdk = await host.server.modelWork.withInference(async () => {
      const source = designModel.model.synthesizeStream(designBody.input, {
        voice: designBody.voice,
        seed: 42,
        maxDurationSeconds: 30,
      });
      const chunks: Buffer[] = [];
      for await (const chunk of source) chunks.push(audioEncodePcm16(chunk.samples));
      await source.completed;
      return Buffer.concat(chunks);
    });
    assert.equal(hash(designNormal.bytes), hash(designSdk));
    results.designNormal = {
      firstMs: designNormal.firstMs,
      wallMs: designNormal.wallMs,
      exactPcm: true,
      trailers: designNormal.trailers,
    };
    assert.deepEqual(
      host.server.registry.audio
        .list()
        .map((e) => e.id)
        .sort(),
      [names.asr, names.design].sort(),
    );
    results.peakRssMiB = process.resourceUsage().maxRSS / 1024;
  });
} finally {
  await host.close();
  results.residentsAfterClose = host.server.registry.audio.list().length;
  const logs = (await readFile(join(logDir, 'requests.ndjson'), 'utf8'))
    .trim()
    .split('\n')
    .filter(Boolean)
    .map((line) => JSON.parse(line));
  results.requestMetrics = logs.map((entry) => entry.audioMetrics);
  results.peakRequestTempBytes = Math.max(0, ...logs.map((entry) => entry.audioMetrics?.tempBytes ?? 0));
  await rm(logDir, { recursive: true, force: true });
  await writeFile(output, JSON.stringify(results, null, 2) + '\n');
}
console.log('Saved', output);
