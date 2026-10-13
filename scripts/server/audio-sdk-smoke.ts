/** Real-checkpoint interoperability with unmodified OpenAI clients. No model downloads. */
import assert from 'node:assert/strict';
import { spawn } from 'node:child_process';
import { createHash, randomUUID } from 'node:crypto';
import { createReadStream } from 'node:fs';
import { mkdtemp, readFile, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';

import OpenAI from 'openai';
import { Stream } from 'openai/core/streaming';
import { VERSION } from 'openai/version';

import { createInferenceHost } from '../../packages/server/src/host/index.js';

const modelsDir = resolve(process.argv[2] ?? '.cache/tts/models');
const fixture = resolve(process.argv[3] ?? 'examples/audio-ask-16k.wav');
const output = resolve(process.argv[4] ?? '.cache/serve-audio-sdk.json');
const names = {
  asr: process.env.ASR_MODEL ?? 'ASR-1.7B-hf-8bit',
  tts: process.env.TTS_MODEL ?? 'CustomVoice-8bit',
  design: process.env.DESIGN_MODEL ?? 'VoiceDesign-1.7B-8bit',
};
const work = await mkdtemp(join(tmpdir(), 'mlx-audio-sdk-'));
const token = randomUUID();
const host = await createInferenceHost({
  modelsDir,
  model: names.tts,
  port: 0,
  authToken: token,
  disableStore: true,
  sweepOrphanTempRoots: false,
  logDir: join(work, 'logs'),
});
const client = new OpenAI({ baseURL: `${host.url}/v1`, apiKey: token, maxRetries: 0, timeout: 60_000 });
type RecordData = Record<string, unknown>;
const results: { name: string; status: 'passed' | 'failed'; ms: number; detail: unknown }[] = [];
const hash = (bytes: Uint8Array) => createHash('sha256').update(bytes).digest('hex');
const record = (value: unknown): RecordData => {
  assert.ok(value && typeof value === 'object');
  return value as RecordData;
};
async function check(name: string, action: () => Promise<unknown>) {
  const start = performance.now();
  try {
    const detail = await action();
    results.push({ name, status: 'passed', ms: performance.now() - start, detail });
    console.log('PASS', name);
  } catch (error) {
    results.push({ name, status: 'failed', ms: performance.now() - start, detail: String(error) });
    console.error('FAIL', name, error);
  }
}
async function run(command: string, args: string[], env = process.env) {
  await new Promise<void>((resolve, reject) => {
    const child = spawn(command, args, { env, stdio: 'inherit' });
    child.once('error', reject);
    child.once('exit', (code, signal) =>
      code === 0 ? resolve() : reject(new Error(`${command} exited ${code ?? signal}`)),
    );
  });
}
async function expectedError(action: () => PromiseLike<unknown>, status: number, code?: string, param?: string) {
  try {
    await action();
  } catch (error) {
    assert.ok(error instanceof OpenAI.APIError);
    assert.equal(error.status, status);
    if (code) assert.equal(error.code, code);
    if (param) assert.equal(error.param, param);
    return { status: error.status, code: error.code, param: error.param };
  }
  throw new Error(`Expected HTTP ${status}`);
}
async function idle() {
  const deadline = performance.now() + 10_000;
  while (host.server.registry.audio.pending) {
    assert.ok(performance.now() < deadline, 'Native task did not acknowledge cancellation');
    await new Promise((resolve) => setTimeout(resolve, 20));
  }
}
let voice = process.env.TTS_VOICE ?? 'ryan';
let supportsInstructions = false;
let rate = 0,
  channels = 0;
let ordinaryText = '';
let baseline = Buffer.alloc(0);
const sampling = { seed: 42, temperature: 0, max_duration_seconds: 12 };
const speech = () => ({ model: names.tts, voice, input: 'Hello. This is a local speech service.', sampling });
const audioHeaders = (response: Response) => {
  assert.equal(Number(response.headers.get('x-sample-rate')), rate);
  assert.equal(Number(response.headers.get('x-channels')), channels);
  const first = response.headers.get('x-first-pcm-ms');
  assert.notEqual(first, null);
  assert.ok(Number.isFinite(Number(first)) && Number(first) >= 0);
};
let python: unknown;
try {
  await host.server.withSuspendedDrains(async () => {
    await check('node/models-and-voices', async () => {
      const models = await client.models.list();
      for (const name of [names.asr, names.tts]) assert.ok(models.data.some((m) => m.id === name));
      const catalog = await client.get<{ data: RecordData[] }>('/audio/voices', { query: { model: names.tts } });
      const caps = catalog.data[0];
      assert.ok(Array.isArray(caps.voices));
      if (!caps.voices.includes(voice)) voice = String(caps.voices[0]);
      rate = Number(caps.sample_rate);
      channels = Number(caps.channels);
      assert.ok(Array.isArray(caps.conditioning));
      supportsInstructions = caps.conditioning.some((value) => record(value).instruct === 'supported');
      assert.ok(rate > 0 && channels > 0);
      return { models: models.data.map((m) => m.id), voice, sampleRate: rate, channels };
    });
    await check('node/bearer-auth-error', () =>
      expectedError(
        () =>
          new OpenAI({
            baseURL: `${host.url}/v1`,
            apiKey: 'invalid-local-token',
            maxRetries: 0,
          }).models.list(),
        401,
      ),
    );
    await check('node/x-api-key-auth', async () => {
      const models = await client.models.list({ headers: { Authorization: null, 'x-api-key': token } });
      assert.ok(models.data.length);
      return { models: models.data.length };
    });
    await check('node/asr-multipart-json', async () => {
      const result = await client.audio.transcriptions.create({ model: names.asr, file: createReadStream(fixture) });
      assert.ok(result.text.trim());
      ordinaryText = result.text;
      return result;
    });
    await check('node/asr-text', async () => {
      const result = await client.audio.transcriptions
        .create({
          model: names.asr,
          file: createReadStream(fixture),
          response_format: 'text',
        })
        .withResponse();
      assert.equal(result.data, ordinaryText);
      assert.equal(result.response.headers.get('x-finish-reason'), 'stop');
      return { text: result.data };
    });
    for (const sampleRate of [16000, 22050, 24000, 44100, 48000, 96000]) {
      await check(`node/asr-${sampleRate}Hz-stereo-pcm24`, async () => {
        const path = join(work, `input-${sampleRate}.wav`);
        await run(process.env.FFMPEG ?? 'ffmpeg', [
          '-v',
          'error',
          '-y',
          '-i',
          fixture,
          '-ar',
          String(sampleRate),
          '-ac',
          '2',
          '-c:a',
          'pcm_s24le',
          path,
        ]);
        const result = await client.audio.transcriptions.create({ model: names.asr, file: createReadStream(path) });
        assert.ok(result.text.trim());
        assert.equal(record(result).finish_reason, 'stop');
        return { sampleRate, text: result.text, duration: record(result).duration };
      });
    }
    await check('node/asr-raw-wav-sdk-post', async () => {
      const result = await client.post<RecordData>('/audio/transcriptions', {
        query: { model: names.asr },
        body: await readFile(fixture),
        headers: { 'Content-Type': 'audio/wav' },
      });
      assert.equal(result.text, ordinaryText);
      return result;
    });
    await check('node/asr-revision-sse', async () => {
      const params = {
        model: names.asr,
        file: createReadStream(fixture),
        stream: true as const,
        sampling: JSON.stringify({ max_tokens: 128 }),
        stream_options: JSON.stringify({ chunk_seconds: 1 }),
      };
      const events: RecordData[] = [];
      for await (const event of await client.audio.transcriptions.create(params)) {
        const snapshot = record(event);
        assert.equal(snapshot.type, 'transcript');
        assert.ok(Number(snapshot.revision) > Number(events.at(-1)?.revision ?? -1));
        assert.equal(typeof snapshot.stable, 'string');
        assert.equal(typeof snapshot.provisional, 'string');
        events.push(snapshot);
      }
      assert.ok(events.length > 1);
      assert.equal(events.at(-1)?.is_final, true);
      assert.equal(events.at(-1)?.provisional, '');
      return { events: events.length, final: events.at(-1), semantics: 'replace whole snapshot' };
    });
    await check('node/tts-pcm', async () => {
      const response = await client.audio.speech.create({ ...speech(), response_format: 'pcm' });
      audioHeaders(response);
      const pieces: Buffer[] = [];
      for await (const chunk of response.body!) pieces.push(Buffer.from(chunk));
      baseline = Buffer.concat(pieces);
      assert.ok(baseline.length > 0 && baseline.length % (channels * 2) === 0);
      assert.ok(response.headers.get('trailer')?.includes('X-Audio-Seconds'));
      await writeFile(join(work, 'node.pcm'), baseline);
      return {
        bytes: baseline.length,
        sha256: hash(baseline),
        chunks: pieces.length,
        trailerDeclaration: response.headers.get('trailer'),
        trailersExposedByFetch: false,
      };
    });
    await check('node/tts-default-wav', async () => {
      const response = await client.audio.speech.create(speech());
      audioHeaders(response);
      const bytes = Buffer.from(await response.arrayBuffer());
      assert.equal(bytes.toString('ascii', 0, 4), 'RIFF');
      assert.equal(bytes.readUInt32LE(4), bytes.length - 8);
      assert.equal(bytes.readUInt32LE(24), rate);
      assert.equal(bytes.readUInt16LE(22), channels);
      assert.equal(hash(bytes.subarray(44)), hash(baseline));
      assert.equal(Number(response.headers.get('x-audio-seconds')), baseline.length / (2 * rate * channels));
      for (const header of ['x-first-pcm-ms', 'x-realtime-factor']) {
        assert.notEqual(response.headers.get(header), null);
        assert.ok(Number.isFinite(Number(response.headers.get(header))));
      }
      return { bytes: bytes.length, exactPcm: true, headers: Object.fromEntries(response.headers) };
    });
    await check('node/tts-sse-sdk-decoder', async () => {
      const response = await client.audio.speech.create({ ...speech(), response_format: 'pcm', stream_format: 'sse' });
      audioHeaders(response);
      const pieces: Buffer[] = [];
      let done: RecordData | undefined;
      for await (const event of Stream.fromSSEResponse<RecordData>(response, new AbortController())) {
        if (event.type === 'speech.audio.delta') {
          assert.equal(done, undefined);
          assert.equal(event.sample_rate, rate);
          assert.equal(event.channels, channels);
          pieces.push(Buffer.from(String(event.audio), 'base64'));
        } else {
          assert.equal(event.type, 'speech.audio.done');
          assert.equal(done, undefined);
          done = event;
        }
      }
      assert.ok(done);
      assert.equal(hash(Buffer.concat(pieces)), hash(baseline));
      assert.equal(done.audio_seconds, baseline.length / (2 * rate * channels));
      return { chunks: pieces.length, exactPcm: true, done };
    });
    await check('node/tts-speed', async () => {
      const response = await client.audio.speech.create({ ...speech(), response_format: 'pcm', speed: 1.5 });
      const bytes = await response.arrayBuffer();
      const ratio = bytes.byteLength / baseline.length;
      assert.ok(Math.abs(ratio - 1 / 1.5) < 0.03);
      return { speed: 1.5, durationRatio: ratio };
    });
    await check('node/instructions-capability', async () => {
      const body = { ...speech(), instructions: 'Speak clearly and warmly.' };
      if (!supportsInstructions)
        return expectedError(() => client.audio.speech.create(body), 400, 'invalid_request', 'instructions');
      const response = await client.audio.speech.create(body);
      assert.ok((await response.arrayBuffer()).byteLength > 44);
      return { instructionsAccepted: true };
    });
    await check('node/tts-stream-cancel', async () => {
      const response = await client.audio.speech.create({ ...speech(), response_format: 'pcm' });
      const reader = response.body!.getReader();
      assert.ok((await reader.read()).value?.length);
      await reader.cancel();
      await idle();
      return { nativePendingAfterCancel: host.server.registry.audio.pending };
    });
    await check('node/ndjson-sdk-post-duplex', async () => {
      let requestEnded = false;
      let release!: () => void;
      const firstAudio = new Promise<void>((resolve) => {
        release = resolve;
      });
      async function* body() {
        const { input, ...config } = speech();
        for (const event of [
          { type: 'config', ...config, response_format: 'pcm' },
          { type: 'text', text: input },
          { type: 'flush' },
        ])
          yield Buffer.from(JSON.stringify(event) + '\n');
        await firstAudio;
        yield Buffer.from(JSON.stringify({ type: 'text', text: ' Goodbye.' }) + '\n');
        requestEnded = true;
      }
      try {
        const response = await client
          .post('/audio/speech/stream', {
            body: body(),
            headers: { 'Content-Type': 'application/x-ndjson' },
          })
          .asResponse();
        assert.equal(response.status, 200);
        audioHeaders(response);
        let bytes = 0,
          chunks = 0;
        for await (const chunk of response.body!) {
          if (!chunks) {
            assert.equal(requestEnded, false);
            release();
          }
          chunks++;
          bytes += chunk.length;
        }
        assert.ok(bytes > 0);
        assert.equal(requestEnded, true);
        return { audioBeforeRequestEof: true, bytes, chunks };
      } finally {
        release();
      }
    });
    for (const format of ['mp3', 'opus', 'aac', 'flac'] as const)
      await check(`node/reject-${format}`, () =>
        expectedError(
          () => client.audio.speech.create({ ...speech(), response_format: format }),
          400,
          'unsupported_response_format',
          'response_format',
        ),
      );
    await check('node/reject-oversized-input', () =>
      expectedError(
        () => client.audio.speech.create({ ...speech(), input: '😀'.repeat(4097) }),
        400,
        'invalid_request',
        'input',
      ),
    );
    await check('node/reject-unknown-model', () =>
      expectedError(
        () => client.audio.speech.create({ ...speech(), model: 'missing-sdk-test-model' }),
        404,
        'model_not_found',
        'model',
      ),
    );
    await check('node/reject-wrong-task', () =>
      expectedError(
        () => client.audio.speech.create({ ...speech(), model: names.asr }),
        400,
        'model_capability_mismatch',
        'model',
      ),
    );
    await check('node/voicedesign-extension', async () => {
      const response = await client
        .post('/audio/speech', {
          body: {
            model: names.design,
            input: 'Hello, welcome!',
            voice: { type: 'description', description: 'A warm adult female voice.' },
            instructions: 'Speak clearly and warmly.',
            response_format: 'wav',
            sampling: { seed: 42, max_duration_seconds: 12 },
          },
        })
        .asResponse();
      assert.equal(response.status, 200);
      const bytes = Buffer.from(await response.arrayBuffer());
      assert.equal(bytes.toString('ascii', 0, 4), 'RIFF');
      return { bytes: bytes.length, sampleRate: Number(response.headers.get('x-sample-rate')) };
    });
    if (process.env.SDK_PYTHON)
      await check('python/sdk-suite', async () => {
        const path = join(work, 'python.json');
        try {
          await run(process.env.SDK_PYTHON!, ['scripts/server/audio-sdk-smoke.py', fixture, path], {
            ...process.env,
            MLX_AUDIO_SDK_BASE_URL: `${host.url}/v1`,
            MLX_AUDIO_SDK_TOKEN: token,
            ASR_MODEL: names.asr,
            TTS_MODEL: names.tts,
            TTS_VOICE: voice,
            MLX_AUDIO_SDK_PCM: join(work, 'node.pcm'),
          });
        } finally {
          python = await readFile(path, 'utf8')
            .then(JSON.parse)
            .catch((error: unknown) => ({ reportError: String(error) }));
        }
        return python;
      });
  });
} finally {
  await host.close();
  await writeFile(
    output,
    JSON.stringify(
      {
        recordedAt: new Date().toISOString(),
        node: process.version,
        nodeSdk: VERSION,
        names,
        fixture,
        results,
        python,
        residentsAfterClose: host.server.registry.audio.list().length,
        pythonRequested: !!process.env.SDK_PYTHON,
      },
      null,
      2,
    ) + '\n',
  );
  await rm(work, { recursive: true, force: true });
}
console.log(`Report: ${output}`);
if (results.some((r) => r.status === 'failed')) process.exitCode = 1;
