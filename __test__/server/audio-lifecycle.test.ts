import { mkdtemp, readFile, rm } from 'node:fs/promises';
import { request as httpRequest } from 'node:http';
import type { AddressInfo } from 'node:net';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

import { audioEncodePcm16, audioWavHeader } from '@mlx-node/core';
import type { Qwen3AsrResult } from '@mlx-node/core';
import OpenAI from 'openai';
import { afterEach, expect, it, vi } from 'vite-plus/test';

import { AudioRegistry, type AsrModel } from '../../packages/server/src/audio-registry.js';
import { attachLogger } from '../../packages/server/src/host/logger.js';
import { createServer, type ServerInstance } from '../../packages/server/src/server.js';
import { createTtsModel, type BackendChunk, type TtsBackend } from '../../packages/tts/src/model.js';

const servers: ServerInstance[] = [];
afterEach(async () => {
  for (const s of servers.splice(0)) await s.close({ timeoutMs: 100 });
});
function deferred<T = void>() {
  let resolve!: (value: T) => void;
  let reject!: (error: unknown) => void;
  const promise = new Promise<T>((a, b) => {
    resolve = a;
    reject = b;
  });
  return { resolve, reject, promise };
}
async function start(options: Parameters<typeof createServer>[0] = {}) {
  const server = await createServer({
    host: '127.0.0.1',
    port: 0,
    disableStore: true,
    idleClearCacheMs: 0,
    ...options,
  });
  servers.push(server);
  return { server, url: `http://127.0.0.1:${(server.server.address() as AddressInfo).port}` };
}
function backend(): TtsBackend {
  return {
    capabilities: {
      family: 'test',
      variant: 'custom_voice',
      sampleRate: 24000,
      channels: 1,
      voices: ['v'],
      languages: ['english'],
      conditioning: [{ voice: 'preset', instruct: 'supported' }],
      voiceCloning: false,
      audioStreaming: true,
      textStreaming: 'segmented',
    },
    prepareVoice: vi.fn(),
    releaseVoice: vi.fn(),
    dispose: vi.fn(async () => {}),
    start: () => {
      let at = 0;
      return {
        next: async () =>
          at++ === 0
            ? { samples: new Float32Array(24), finished: false }
            : { samples: new Float32Array(), finished: true, finishReason: 'eos', synthesisMs: 1 },
        cancel() {},
        waitFinished: async () => {},
      };
    },
  };
}
const config = JSON.stringify({ type: 'config', model: 'tts', voice: 'v' }) + '\n';
const text = JSON.stringify({ type: 'text', text: '你好。' }) + '\n';

it('OpenAI SDK reports a midstream PCM failure without replaying speech, with default retries enabled', async () => {
  const { server, url } = await start();
  const native = backend();
  const completion = deferred<BackendChunk>();
  const waitFinished = vi.fn(async () => {});
  const generate = vi.fn(() => {
    let first = true;
    return {
      next: async () => {
        if (!first) return completion.promise;
        first = false;
        return { samples: new Float32Array(24), finished: false };
      },
      cancel() {},
      waitFinished,
    };
  });
  native.start = generate;
  server.registry.audio.register('tts', { kind: 'tts', model: createTtsModel(native) });
  const transport = vi.fn(fetch);
  const client = new OpenAI({ baseURL: `${url}/v1`, apiKey: 'local-test', fetch: transport });
  const response = await client.audio.speech.create({
    model: 'tts',
    voice: 'v',
    input: 'Hello.',
    response_format: 'pcm',
  });
  expect(response.status).toBe(200);
  const reader = response.body!.getReader();
  expect((await reader.read()).value?.byteLength).toBe(48);
  completion.reject(new Error('injected late inference failure'));
  await expect(reader.read()).rejects.toThrow();
  reader.releaseLock();
  expect(transport).toHaveBeenCalledOnce();
  expect(generate).toHaveBeenCalledOnce();
  expect(waitFinished).toHaveBeenCalled();
  expect(server.registry.audio.pending).toBe(0);
});

function asrResult(): Qwen3AsrResult {
  return {
    text: 'test',
    stableText: 'test',
    provisionalText: '',
    tokenIds: [],
    reachedMaxTokens: false,
    audioSeconds: 1,
    segmentAudioSeconds: 1,
    featureMs: 0,
    prefillMs: 0,
    decodeMs: 1,
    totalMs: 1,
    tokensPerSecond: 1,
    realTimeFactor: 0.001,
    revision: 1,
    isFinal: false,
  };
}

it.each(['start', 'result', 'createStream', 'feed', 'finish'] as const)(
  'ASR %s failure suppresses retries before the first HTTP output',
  async (stage) => {
    const { server, url } = await start();
    const stream = stage !== 'start' && stage !== 'result';
    const failure = new Error('injected ASR failure');
    const disposeStream = vi.fn(async () => {});
    const model = {
      startTranscription: vi.fn(() => {
        if (stage === 'start') throw failure;
        return {
          cancel: vi.fn(),
          result: async () => {
            throw failure;
          },
        };
      }),
      createStream: vi.fn(async () => {
        if (stage === 'createStream') throw failure;
        return {
          feed: async () => {
            if (stage === 'feed') throw failure;
            return null;
          },
          finish: async () => {
            throw failure;
          },
          cancel: vi.fn(),
          dispose: disposeStream,
        };
      }),
      dispose: vi.fn(async () => {}),
    } satisfies AsrModel;
    server.registry.audio.register('asr', { kind: 'asr', model });
    const pcm = audioEncodePcm16(new Float32Array(100));
    const response = await fetch(`${url}/v1/audio/transcriptions?model=asr&stream=${stream}`, {
      method: 'POST',
      headers: { 'Content-Type': 'audio/wav' },
      body: Buffer.concat([audioWavHeader(16000, 1, pcm.length), pcm]),
    });
    expect(response.status).toBe(500);
    expect(response.headers.get('x-should-retry')).toBe('false');
    expect(response.headers.get('retry-after')).toBeNull();
    expect(response.headers.get('access-control-expose-headers')).toContain('X-Should-Retry');
    expect((await response.json()).error.code).toBe('audio_inference_failed');
    expect(model.startTranscription).toHaveBeenCalledTimes(stream ? 0 : 1);
    expect(model.createStream).toHaveBeenCalledTimes(stream ? 1 : 0);
    expect(disposeStream).toHaveBeenCalledTimes(stage === 'feed' || stage === 'finish' ? 1 : 0);
    expect(server.registry.audio.pending).toBe(0);
  },
);

it.each(['json', 'text'] as const)(
  'shutdown terminates an uploaded ASR %s request after native acknowledgment',
  async (format) => {
    const { server, url } = await start();
    const completion = deferred<Qwen3AsrResult>();
    const cancelled = deferred();
    const startTranscription = vi.fn(() => ({ cancel: () => cancelled.resolve(), result: () => completion.promise }));
    const dispose = vi.fn(async () => {});
    const model: AsrModel = {
      startTranscription,
      createStream: vi.fn(),
      dispose,
    };
    server.registry.audio.register('asr', { kind: 'asr', model });
    const pcm = audioEncodePcm16(new Float32Array(100));
    const response = fetch(`${url}/v1/audio/transcriptions?model=asr&response_format=${format}`, {
      method: 'POST',
      headers: { 'Content-Type': 'audio/wav' },
      body: Buffer.concat([audioWavHeader(16000, 1, pcm.length), pcm]),
    });
    // Observe any transport failure immediately, including against the old implementation.
    void response.catch(() => {});
    await vi.waitFor(() => expect(startTranscription).toHaveBeenCalledOnce());
    let closed = false;
    const closing = server.close({ timeoutMs: 1000 }).then((result) => {
      closed = true;
      return result;
    });
    await cancelled.promise;
    expect(closed).toBe(false);
    expect(dispose).not.toHaveBeenCalled();
    completion.reject(new Error('native cancellation acknowledged'));
    expect((await closing).forced).toBe(false);
    const res = await response;
    expect(res.status).toBe(503);
    expect(res.headers.get('x-should-retry')).toBe('false');
    expect(res.headers.get('retry-after')).toBeNull();
    expect((await res.json()).error.code).toBe('server_shutdown');
    expect(dispose).toHaveBeenCalledOnce();
    expect(server.registry.audio.pending).toBe(0);
  },
);

it('shutdown terminates ASR SSE with one error after disposing its stream', async () => {
  const { server, url } = await start();
  const completion = deferred<Qwen3AsrResult>();
  const feed = vi
    .fn()
    .mockResolvedValueOnce(asrResult())
    .mockImplementation(() => completion.promise);
  const dispose = vi.fn(async () => {});
  server.registry.audio.register('asr', {
    kind: 'asr',
    model: {
      startTranscription: vi.fn(),
      dispose: vi.fn(async () => {}),
      createStream: async () => ({
        feed,
        finish: vi.fn(),
        dispose,
        cancel: () => completion.reject(new Error('cancelled')),
      }),
    },
  });
  const pcm = audioEncodePcm16(new Float32Array(3200));
  const query = new URLSearchParams({
    model: 'asr',
    stream: 'true',
    stream_options: JSON.stringify({ chunk_seconds: 0.1 }),
  });
  const res = await fetch(`${url}/v1/audio/transcriptions?${query}`, {
    method: 'POST',
    headers: { 'Content-Type': 'audio/wav' },
    body: Buffer.concat([audioWavHeader(16000, 1, pcm.length), pcm]),
  });
  const body = res.text();
  void body.catch(() => {});
  await vi.waitFor(() => expect(feed).toHaveBeenCalledTimes(2));
  expect((await server.close({ timeoutMs: 1000 })).forced).toBe(false);
  expect(dispose).toHaveBeenCalledOnce();
  const events = await body;
  expect(events.match(/event: error/g)).toHaveLength(1);
  expect(events).toContain('server_shutdown');
  expect(events).not.toContain('"is_final":true');
});

it.each(['wav', 'pcm', 'sse'] as const)(
  'shutdown closes TTS %s without waiting for the force timer',
  async (format) => {
    const { server, url } = await start();
    const native = backend();
    const blocked = deferred();
    const completion = deferred<BackendChunk | null | undefined>();
    const cancel = vi.fn(() => completion.reject(new Error('native cancellation acknowledged')));
    const waitFinished = vi.fn(async () => {});
    native.start = () => {
      let first = true;
      return {
        next: async () => {
          if (first) {
            first = false;
            return { samples: new Float32Array(24), finished: false };
          }
          blocked.resolve();
          return completion.promise;
        },
        cancel,
        waitFinished,
      };
    };
    server.registry.audio.register('tts', { kind: 'tts', model: createTtsModel(native) });
    const response = fetch(`${url}/v1/audio/speech`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        model: 'tts',
        voice: 'v',
        input: 'Hello.',
        response_format: format === 'sse' ? 'pcm' : format,
        stream_format: format === 'sse' ? 'sse' : 'audio',
      }),
    });
    const body = response.then(async (res) => ({ status: res.status, text: await res.text() })).catch(() => undefined);
    await blocked.promise;
    if (format !== 'wav') await response;
    expect((await server.close({ timeoutMs: 1000 })).forced).toBe(false);
    expect(cancel).toHaveBeenCalled();
    expect(waitFinished).toHaveBeenCalled();
    expect(native.dispose).toHaveBeenCalledOnce();
    const received = await body;
    if (format === 'pcm') expect(received).toBeUndefined();
    else {
      expect(received?.status).toBe(format === 'wav' ? 503 : 200);
      expect(received?.text).toContain('server_shutdown');
      expect(received?.text).not.toContain('speech.audio.done');
      if (format === 'sse') expect(received?.text.match(/event: error/g)).toHaveLength(1);
    }
  },
);

it('does not release inference on disconnect until ASR acknowledges native completion', async () => {
  const { server, url } = await start();
  const completion = deferred<Qwen3AsrResult>();
  const cancel = vi.fn();
  const startTranscription = vi.fn(() => ({ cancel, result: () => completion.promise }));
  const asr: AsrModel = {
    startTranscription,
    createStream: vi.fn(),
    dispose: vi.fn(async () => {}),
  };
  server.registry.audio.register('asr', { kind: 'asr', model: asr });
  const pcm = audioEncodePcm16(new Float32Array(100));
  const control = new AbortController();
  const pending = fetch(`${url}/v1/audio/transcriptions?model=asr`, {
    method: 'POST',
    signal: control.signal,
    headers: { 'Content-Type': 'audio/wav' },
    body: Buffer.concat([audioWavHeader(16000, 1, pcm.length), pcm]),
  }).catch(() => undefined);
  await vi.waitFor(() => expect(startTranscription).toHaveBeenCalledOnce());
  control.abort();
  await vi.waitFor(() => expect(cancel).toHaveBeenCalledOnce());
  let unloaded = false;
  const writer = server.modelWork.withModelLoad(() => {
    unloaded = true;
  });
  await new Promise((resolve) => setImmediate(resolve));
  expect(unloaded).toBe(false);
  completion.reject(new Error('cancel acknowledged'));
  await writer;
  await pending;
  expect(unloaded).toBe(true);
  expect(server.registry.audio.pending).toBe(0);
});

it('aliases share admission and cancelling a waiter cannot overtake its predecessor', async () => {
  const registry = new AudioRegistry(2);
  const model = createTtsModel(backend());
  registry.register('one', { kind: 'tts', model });
  registry.register('two', { kind: 'tts', model });
  const a = registry.acquire('one')!;
  const b = registry.acquire('two')!;
  const c = registry.acquire('one')!;
  expect(() => registry.acquire('two')).toThrow('queue is full');
  const gate = deferred();
  const started = deferred();
  const abort = new AbortController();
  const order: number[] = [];
  const first = a
    .run(async () => {
      order.push(1);
      started.resolve();
      await gate.promise;
    })
    .finally(() => a.release());
  await started.promise;
  const second = b
    .run(async () => {
      order.push(2);
    }, abort.signal)
    .catch(() => {})
    .finally(() => b.release());
  const third = c
    .run(async () => {
      order.push(3);
    })
    .finally(() => c.release());
  abort.abort();
  await second;
  expect(order).toEqual([1]);
  gate.resolve();
  await Promise.all([first, third]);
  expect(order).toEqual([1, 3]);
  expect(registry.pending).toBe(0);
  await registry.disposeAll();
});

it('shutdown interrupts NDJSON waiting for input and disposes the native model', async () => {
  const { server, url } = await start();
  const native = backend();
  server.registry.audio.register('tts', { kind: 'tts', model: createTtsModel(native) });
  const req = httpRequest(`${url}/v1/audio/speech/stream`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/x-ndjson' },
  });
  req.on('error', () => {});
  req.write(config);
  await vi.waitFor(() => expect(server.registry.audio.pending).toBe(1));
  expect((await server.close({ timeoutMs: 1000 })).forced).toBe(false);
  expect(native.dispose).toHaveBeenCalledOnce();
  expect(server.registry.audio.pending).toBe(0);
  req.destroy();
});

it('reports input idle timeout before committing audio headers', async () => {
  const { server, url } = await start({ audio: { inputIdleMs: 20 } });
  server.registry.audio.register('tts', { kind: 'tts', model: createTtsModel(backend()) });
  const status = await new Promise<number>((resolve, reject) => {
    const req = httpRequest(
      `${url}/v1/audio/speech/stream`,
      { method: 'POST', headers: { 'Content-Type': 'application/x-ndjson' } },
      (res) => {
        res.resume();
        res.on('end', () => resolve(res.statusCode!));
      },
    );
    req.on('error', reject);
    req.write(config);
  });
  expect(status).toBe(408);
});

it('aborts binary output on a later invalid event rather than appending JSON or a success trailer', async () => {
  const { server, url } = await start();
  server.registry.audio.register('tts', { kind: 'tts', model: createTtsModel(backend()) });
  await new Promise<void>((resolve, reject) => {
    let wrote = false;
    const req = httpRequest(
      `${url}/v1/audio/speech/stream`,
      { method: 'POST', headers: { 'Content-Type': 'application/x-ndjson' } },
      (res) => {
        res.on('data', (bytes) => {
          expect(bytes.length % 2).toBe(0);
          if (!wrote) {
            wrote = true;
            req.end('{invalid json}\n');
          }
        });
        res.on('aborted', () => {
          expect(res.trailers['x-finish-reason']).toBeUndefined();
          resolve();
        });
        res.on('error', () => {});
        res.on('end', () => reject(new Error('Unexpected successful end')));
      },
    );
    req.on('error', reject);
    req.write(config + text);
  });
  await vi.waitFor(() => expect(server.registry.audio.pending).toBe(0));
});

it('multipart byte limits terminate file consumption and release upload resources', async () => {
  const { url } = await start({ audio: { maxUploadBytes: 64 } });
  const form = new FormData();
  form.set('file', new Blob([Buffer.alloc(1024)]), 'large.wav');
  form.set('model', 'asr');
  const res = await fetch(`${url}/v1/audio/transcriptions`, {
    method: 'POST',
    body: form,
    signal: AbortSignal.timeout(2000),
  });
  expect(res.status).toBe(413);
});

it.each(['storage', 'file_size'] as const)(
  'multipart %s errors interrupt an open upload without waiting for input idleness',
  async (failure) => {
    const { server, url } = await start({
      audio: {
        maxTempBytes: 96,
        maxUploadBytes: failure === 'file_size' ? 96 : 2048,
        maxConcurrentRequests: 1,
        inputIdleMs: 10_000,
      },
    });
    await new Promise<void>((resolve, reject) => {
      const req = httpRequest(
        `${url}/v1/audio/transcriptions`,
        { method: 'POST', headers: { 'Content-Type': 'multipart/form-data; boundary=x' } },
        (res) => {
          let body = '';
          res.on('data', (bytes) => (body += bytes));
          res.on('error', reject);
          res.on('end', () => {
            try {
              expect(res.statusCode).toBe(failure === 'storage' ? 503 : 413);
              expect(res.headers['retry-after']).toBe(failure === 'storage' ? '1' : undefined);
              expect(res.headers['x-should-retry']).toBeUndefined();
              expect(JSON.parse(body).error.code).toBe(
                failure === 'storage' ? 'audio_storage_full' : 'request_too_large',
              );
              resolve();
            } catch (error) {
              reject(error);
            } finally {
              req.destroy();
            }
          });
        },
      );
      req.on('error', reject);
      req.setTimeout(2000, () => req.destroy(new Error('Multipart failure did not interrupt input wait')));
      req.write(
        '--x\r\nContent-Disposition: form-data; name="file"; filename="test.wav"\r\nContent-Type: audio/wav\r\n\r\n',
      );
      req.write(Buffer.alloc(1024));
      // Leave the body open: neither another chunk nor EOF should be needed.
    });

    server.registry.audio.register('asr', {
      kind: 'asr',
      model: {
        startTranscription: () => ({ cancel() {}, result: async () => asrResult() }),
        createStream: vi.fn(),
        dispose: vi.fn(async () => {}),
      },
    });
    const pcm = audioEncodePcm16(new Float32Array(10));
    const res = await fetch(`${url}/v1/audio/transcriptions?model=asr`, {
      method: 'POST',
      headers: { 'Content-Type': 'audio/wav' },
      body: Buffer.concat([audioWavHeader(16000, 1, pcm.length), pcm]),
    });
    expect(res.status).toBe(200);
    expect((await res.json()).text).toBe('test');
  },
);

it('audio logging preserves duplex reads and records counters without payloads', async () => {
  const { server, url } = await start();
  server.registry.audio.register('tts', { kind: 'tts', model: createTtsModel(backend()) });
  const dir = await mkdtemp(join(tmpdir(), 'audio-log-'));
  const log = attachLogger(server.server, dir);
  try {
    await new Promise<void>((resolve, reject) => {
      let sent = false;
      const req = httpRequest(
        `${url}/v1/audio/speech/stream`,
        { method: 'POST', headers: { 'Content-Type': 'application/x-ndjson' } },
        (res) => {
          res.on('data', () => {
            if (!sent) {
              sent = true;
              req.end(text);
            }
          });
          res.on('end', resolve);
          res.on('error', reject);
        },
      );
      req.setTimeout(2000, () => req.destroy(new Error('Logging broke duplex input')));
      req.on('error', reject);
      req.write(config + text);
    });
    await server.close();
    await log.close();
    const entries = (await readFile(join(dir, 'requests.ndjson'), 'utf8'))
      .trim()
      .split('\n')
      .map((line) => JSON.parse(line));
    expect(entries[0]).toMatchObject({
      audio: true,
      reqBody: null,
      resBody: '',
      audioMetrics: { task: 'tts', model: 'tts', state: 'completed', finishReason: 'eos' },
    });
    expect(entries[0].reqBytes).toBeGreaterThan(0);
    expect(entries[0].resBytes).toBeGreaterThan(0);
  } finally {
    await log.close();
    await rm(dir, { recursive: true, force: true });
  }
});

it('audio logs omit query payloads from both structured and pretty logs', async () => {
  const { server, url } = await start();
  const transcribe = vi.fn(() => ({ cancel() {}, result: async () => asrResult() }));
  server.registry.audio.register('asr', {
    kind: 'asr',
    model: {
      startTranscription: transcribe,
      createStream: vi.fn(),
      dispose: async () => {},
    },
  });
  const dir = await mkdtemp(join(tmpdir(), 'audio-query-log-'));
  const log = attachLogger(server.server, dir);
  const prompt = 'PRIVATE_PROMPT_你好';
  const query = new URLSearchParams({ model: 'asr', prompt });
  const pcm = audioEncodePcm16(new Float32Array(100));
  try {
    const res = await fetch(`${url}/v1/audio/transcriptions?${query}`, {
      method: 'POST',
      headers: { 'Content-Type': 'audio/wav' },
      body: Buffer.concat([audioWavHeader(16000, 1, pcm.length), pcm]),
    });
    expect(res.status).toBe(200);
    await res.text();
    expect(transcribe.mock.calls[0]).toEqual([expect.any(Float32Array), expect.objectContaining({ prompt })]);
    await server.close();
    await log.close();
    const structured = await readFile(join(dir, 'requests.ndjson'), 'utf8');
    const pretty = await readFile(join(dir, 'session.log'), 'utf8');
    for (const content of [structured, pretty]) {
      expect(content).not.toContain('PRIVATE_PROMPT');
      expect(content).not.toContain('prompt=');
    }
    expect(JSON.parse(structured)).toMatchObject({
      path: '/v1/audio/transcriptions',
      audioMetrics: { model: 'asr' },
      reqBody: null,
      resBody: '',
    });
  } finally {
    await log.close();
    await rm(dir, { recursive: true, force: true });
  }
});

it('audio logs record ASR stage times and TTS frames in structured and pretty logs', async () => {
  const { server, url } = await start();
  const result = {
    ...asrResult(),
    tokenIds: [1, 2, 3],
    featureMs: 2,
    prefillMs: 20,
    decodeMs: 100,
    totalMs: 155,
    isFinal: true,
  };
  server.registry.audio.register('asr', {
    kind: 'asr',
    model: {
      startTranscription: vi.fn(() => ({ cancel() {}, result: async () => result })),
      createStream: vi.fn(),
      dispose: async () => {},
    },
  });
  server.registry.audio.register('tts', { kind: 'tts', model: createTtsModel(backend()) });
  const dir = await mkdtemp(join(tmpdir(), 'audio-stage-log-'));
  const log = attachLogger(server.server, dir);
  const pcm = audioEncodePcm16(new Float32Array(100));
  try {
    const asr = await fetch(`${url}/v1/audio/transcriptions?model=asr`, {
      method: 'POST',
      headers: { 'Content-Type': 'audio/wav' },
      body: Buffer.concat([audioWavHeader(16000, 1, pcm.length), pcm]),
    });
    expect(asr.status).toBe(200);
    await asr.text();
    const speech = await fetch(`${url}/v1/audio/speech`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ model: 'tts', voice: 'v', input: 'Hello.', response_format: 'pcm' }),
    });
    expect(speech.status).toBe(200);
    expect((await speech.arrayBuffer()).byteLength).toBe(48);
    await server.close();
    await log.close();
    const entries = (await readFile(join(dir, 'requests.ndjson'), 'utf8'))
      .trim()
      .split('\n')
      .map((line) => JSON.parse(line));
    expect(entries).toHaveLength(2);
    const asrEntry = entries.find((entry) => entry.path === '/v1/audio/transcriptions');
    expect(asrEntry.audioMetrics).toMatchObject({
      task: 'asr',
      model: 'asr',
      state: 'completed',
      featureMs: 2,
      prefillMs: 20,
      decodeMs: 100,
      computeMs: 155,
      outputTokens: 3,
      audioSeconds: 1,
      finishReason: 'stop',
    });
    for (const key of ['uploadMs', 'queueWaitMs', 'wavReadMs', 'totalMs'])
      expect(asrEntry.audioMetrics[key], key).toBeGreaterThanOrEqual(0);
    const ttsEntry = entries.find((entry) => entry.path === '/v1/audio/speech');
    expect(ttsEntry.audioMetrics).toMatchObject({
      task: 'tts',
      state: 'completed',
      pcmFrames: 24,
      finishReason: 'eos',
    });
    for (const key of ['queueWaitMs', 'firstPcmMs', 'computeMs', 'totalMs'])
      expect(ttsEntry.audioMetrics[key], key).toBeGreaterThanOrEqual(0);
    const pretty = await readFile(join(dir, 'session.log'), 'utf8');
    expect(pretty).toMatch(
      /200 POST \/v1\/audio\/transcriptions \d+ms 0B model=asr task=asr server\(upload=\d+ms queue=\d+ms wav=\d+ms total=\d+ms\) perf\(feature=2ms prefill=20ms decode=100ms compute=155ms\) audio\(sec=1\.00 out=3\) stop=stop\n/,
    );
    expect(pretty).toMatch(
      /200 POST \/v1\/audio\/speech \d+ms 0B model=tts task=tts server\(queue=\d+ms total=\d+ms\) perf\(compute=\d+ms first_pcm=\d+ms\) audio\(sec=0\.00 frames=24 rtf=[\d.]+\) stop=eos\n/,
    );
  } finally {
    await log.close();
    await rm(dir, { recursive: true, force: true });
  }
});

it('bounds generation for a stalled PCM client and waits for cancellation on drain timeout', async () => {
  const { server, url } = await start({ audio: { outputDrainMs: 40 } });
  const native = backend();
  let produced = 0;
  const cancel = vi.fn();
  const waitFinished = vi.fn(async () => {});
  native.start = () => ({
    next: async () => {
      produced++;
      return { samples: new Float32Array(24000), finished: false };
    },
    cancel,
    waitFinished,
  });
  server.registry.audio.register('tts', { kind: 'tts', model: createTtsModel(native) });
  const req = httpRequest(`${url}/v1/audio/speech`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
  });
  req.on('error', () => {});
  const response = new Promise<void>((resolve) =>
    req.on('response', (res) => {
      res.on('error', () => {});
      res.pause();
      resolve();
    }),
  );
  req.end(JSON.stringify({ model: 'tts', voice: 'v', input: 'Hello.', response_format: 'pcm' }));
  try {
    await response;
    await vi.waitFor(() => expect(cancel).toHaveBeenCalled(), { timeout: 3000 });
    await vi.waitFor(() => expect(server.registry.audio.pending).toBe(0));
    expect(waitFinished).toHaveBeenCalled();
    expect(produced).toBeLessThan(300);
  } finally {
    req.destroy();
  }
});

it('bounds ingress before a streaming request supplies text or a multipart model field', async () => {
  const { server, url } = await start({ audio: { maxConcurrentRequests: 1 } });
  server.registry.audio.register('tts', { kind: 'tts', model: createTtsModel(backend()) });
  const req = httpRequest(`${url}/v1/audio/speech/stream`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/x-ndjson' },
  });
  req.on('error', () => {});
  req.write(config);
  try {
    await vi.waitFor(() => expect(server.registry.audio.pending).toBe(1));
    const res = await fetch(`${url}/v1/audio/transcriptions`, {
      method: 'POST',
      headers: { 'Content-Type': 'multipart/form-data; boundary=x' },
      body: '--x--',
    });
    expect(res.status).toBe(429);
    expect(res.headers.get('retry-after')).toBe('1');
    await res.text();
  } finally {
    req.destroy();
  }
  await vi.waitFor(() => expect(server.registry.audio.pending).toBe(0));
});
