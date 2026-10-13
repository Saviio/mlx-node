import { request as httpRequest } from 'node:http';
import type { AddressInfo } from 'node:net';

import OpenAI from 'openai';
import { afterEach, expect, it, vi } from 'vite-plus/test';

import { AudioError } from '../../packages/server/src/audio-io.js';
import { ttsCapabilities } from '../../packages/server/src/audio-registry.js';
import { createServer, type ServerInstance } from '../../packages/server/src/server.js';
import { createTtsModel, type TtsBackend } from '../../packages/tts/src/model.js';

const servers: ServerInstance[] = [];
afterEach(async () => {
  for (const server of servers.splice(0)) await server.close({ timeoutMs: 100 });
});

interface Outcome {
  status: number;
  bytes?: number;
  code?: string;
}

async function speech(url: string, format: 'pcm' | 'wav' | 'sse' = 'pcm'): Promise<Outcome> {
  // Deliberately keep both the default retry policy and default HTTP transport.
  const sdk = new OpenAI({ baseURL: `${url}/v1`, apiKey: 'local-test', timeout: 10_000 });
  try {
    const response = await sdk.audio.speech.create(
      {
        model: 'tts',
        voice: 'v',
        input: 'Hello.',
        response_format: format === 'sse' ? 'pcm' : format,
        stream_format: format === 'sse' ? 'sse' : 'audio',
      },
      { headers: { 'x-sdk-retry-probe': 'true' } },
    );
    return { status: response.status, bytes: (await response.arrayBuffer()).byteLength };
  } catch (error) {
    if (!(error instanceof OpenAI.APIError)) throw error;
    return { status: error.status!, code: error.code ?? undefined };
  }
}

function backend(failure?: 'start' | 'next') {
  const waitFinished = vi.fn(async () => {});
  const generate = vi.fn(() => {
    if (failure === 'start') throw new Error('injected native start failure');
    let first = true;
    return {
      next: async () => {
        if (failure === 'next') throw new Error('injected native decode failure');
        if (!first)
          return { samples: new Float32Array(), finished: true, finishReason: 'eos' as const, synthesisMs: 1 };
        first = false;
        return { samples: new Float32Array(24), finished: false };
      },
      cancel() {},
      waitFinished,
    };
  });
  const native: TtsBackend = {
    capabilities: {
      family: 'test',
      variant: 'custom_voice',
      sampleRate: 24000,
      channels: 1,
      voices: ['v'],
      languages: ['english'],
      conditioning: [{ voice: 'preset', instruct: 'unsupported' }],
      voiceCloning: false,
      audioStreaming: true,
      textStreaming: 'segmented',
    },
    start: generate,
    prepareVoice: vi.fn(),
    releaseVoice: vi.fn(),
    dispose: vi.fn(async () => {}),
  };
  return { model: createTtsModel(native), generate, waitFinished };
}

function observeAttempts(server: ServerInstance) {
  const attempts: { at: number; status?: number; retryAfter?: string; shouldRetry?: string }[] = [];
  // Observe actual server traffic; never fabricate an HTTP failure in a client mock.
  server.server.prependListener('request', (req, res) => {
    if (req.headers['x-sdk-retry-probe'] !== 'true') return;
    const attempt: (typeof attempts)[number] = { at: performance.now() };
    attempts.push(attempt);
    res.once('finish', () => {
      attempt.status = res.statusCode;
      const retryAfter = res.getHeader('Retry-After');
      if (retryAfter !== undefined) attempt.retryAfter = String(retryAfter);
      const shouldRetry = res.getHeader('X-Should-Retry');
      if (shouldRetry !== undefined) attempt.shouldRetry = String(shouldRetry);
    });
  });
  return attempts;
}

const cases = (['ingress', 'resident queue', 'cold load queue'] as const).flatMap((layer) =>
  [true, false].map((recover) => ({ layer, recover })),
);

it.each(cases)(
  `node sdk default retries: $layer, recover=$recover, rejected attempts never synthesize`,
  async ({ layer, recover }) => {
    const { model, generate } = backend();
    const load = vi.fn(async () => {
      server.registry.audio.register('tts', { kind: 'tts', model });
    });
    const server = await createServer({
      host: '127.0.0.1',
      port: 0,
      disableStore: true,
      idleClearCacheMs: 0,
      maxQueueDepthPerModel: 1,
      audio: { maxConcurrentRequests: layer === 'ingress' ? 1 : 8 },
      resolveModel: load,
      listModels: () => [
        {
          id: 'tts',
          object: 'model',
          created: 0,
          owned_by: 'test',
          modality: 'audio',
          task: 'tts',
          capabilities: ttsCapabilities(model.capabilities),
        },
      ],
    });
    servers.push(server);
    if (layer !== 'cold load queue') server.registry.audio.register('tts', { kind: 'tts', model });
    const url = `http://127.0.0.1:${(server.server.address() as AddressInfo).port}`;
    const attempts = observeAttempts(server);
    let release: () => Promise<void>;
    if (layer === 'ingress') {
      const blocker = httpRequest(`${url}/v1/audio/speech/stream`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/x-ndjson' },
      });
      blocker.on('error', () => {});
      blocker.write(JSON.stringify({ type: 'config', model: 'tts', voice: 'v' }) + '\n');
      release = async () => {
        blocker.destroy();
        await vi.waitFor(() => expect(server.registry.audio.pending).toBe(0));
      };
      await vi.waitFor(() => expect(server.registry.audio.pending).toBe(1));
    } else {
      // Real admission permits fill the configured runner + waiter budget.
      const permits = [0, 1].map(() =>
        layer === 'resident queue'
          ? server.registry.audio.acquire('tts')!
          : server.modelWork.beginRequestLoadAdmission('tts'),
      );
      release = async () => {
        for (const permit of permits) permit.release();
      };
    }
    try {
      const result = speech(url);
      void result.catch(() => {});
      await vi.waitFor(() => expect(attempts[0]?.status).toBe(429), { timeout: 5000 });
      expect(generate).not.toHaveBeenCalled();
      expect(load).not.toHaveBeenCalled();
      if (recover) await release();
      const outcome = await result;
      expect(outcome).toEqual(
        recover
          ? { status: 200, bytes: 48 }
          : {
              status: 429,
              code: layer === 'ingress' ? 'audio_capacity_exceeded' : 'queue_full',
            },
      );
      expect(attempts.map((a) => a.status)).toEqual(recover ? [429, 200] : [429, 429, 429]);
      expect(attempts.every((a) => a.shouldRetry === undefined)).toBe(true);
      for (let i = 1; i < attempts.length; i++) {
        expect(attempts[i - 1].retryAfter).toBe('1');
        expect(attempts[i].at - attempts[i - 1].at).toBeGreaterThanOrEqual(900);
      }
      expect(generate).toHaveBeenCalledTimes(recover ? 1 : 0);
      expect(load).toHaveBeenCalledTimes(recover && layer === 'cold load queue' ? 1 : 0);
    } finally {
      await release();
    }
    expect(server.registry.audio.pending).toBe(0);
  },
  15_000,
);

it.each([
  { failure: 'start', format: 'pcm' },
  { failure: 'next', format: 'pcm' },
  { failure: 'next', format: 'sse' },
  { failure: 'next', format: 'wav' },
  { failure: 'storage', format: 'wav' },
] as const)(
  `node sdk default retries: $failure failure during $format synthesis executes only once`,
  async ({ failure, format }) => {
    const { model, generate, waitFinished } = backend(failure === 'storage' ? undefined : failure);
    const server = await createServer({
      host: '127.0.0.1',
      port: 0,
      disableStore: true,
      idleClearCacheMs: 0,
      // The WAV header fits; storing the first 48 PCM bytes exhausts the quota.
      audio: failure === 'storage' ? { maxTempBytes: 44 } : undefined,
    });
    servers.push(server);
    server.registry.audio.register('tts', { kind: 'tts', model });
    const attempts = observeAttempts(server);
    const url = `http://127.0.0.1:${(server.server.address() as AddressInfo).port}`;
    const status = failure === 'storage' ? 503 : 500;
    expect(await speech(url, format)).toEqual({
      status,
      code: failure === 'storage' ? 'audio_storage_full' : 'audio_inference_failed',
    });
    expect(attempts).toHaveLength(1);
    expect(attempts[0]).toMatchObject({ status, shouldRetry: 'false' });
    expect(attempts[0].retryAfter).toBeUndefined();
    expect(generate).toHaveBeenCalledOnce();
    if (failure !== 'start') expect(waitFinished).toHaveBeenCalledOnce();
    expect(server.registry.audio.pending).toBe(0);
  },
  15_000,
);

it(`node sdk default retries: pre-dispatch model loading failures remain retryable`, async () => {
  const { model, generate } = backend();
  const load = vi.fn(async () => {
    throw new AudioError(503, 'model_unavailable', 'injected temporary loading failure');
  });
  const server = await createServer({
    host: '127.0.0.1',
    port: 0,
    disableStore: true,
    idleClearCacheMs: 0,
    resolveModel: load,
    listModels: () => [
      {
        id: 'tts',
        object: 'model',
        created: 0,
        owned_by: 'test',
        modality: 'audio',
        task: 'tts',
        capabilities: ttsCapabilities(model.capabilities),
      },
    ],
  });
  servers.push(server);
  const attempts = observeAttempts(server);
  const url = `http://127.0.0.1:${(server.server.address() as AddressInfo).port}`;
  expect(await speech(url)).toEqual({ status: 503, code: 'model_unavailable' });
  expect(attempts.map((a) => a.status)).toEqual([503, 503, 503]);
  expect(attempts.every((a) => a.retryAfter === '1' && a.shouldRetry === undefined)).toBe(true);
  expect(load).toHaveBeenCalledTimes(3);
  expect(generate).not.toHaveBeenCalled();
  expect(server.registry.audio.pending).toBe(0);
  await model.dispose();
}, 15_000);
