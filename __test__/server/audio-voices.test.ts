import type { AddressInfo } from 'node:net';

import { audioEncodePcm16, audioWavHeader } from '@mlx-node/core';
import { InvalidReferenceAudioError } from '@mlx-node/tts';
import { afterEach, describe, expect, it, vi } from 'vite-plus/test';

import { AudioRegistry } from '../../packages/server/src/audio-registry.js';
import type { AudioServiceOptions } from '../../packages/server/src/audio-service.js';
import { AudioVoices, MemoryAudioVoiceStore } from '../../packages/server/src/audio-voices.js';
import { createServer, type ServerInstance } from '../../packages/server/src/server.js';
import { createTtsModel, type BackendOptions, type TtsBackend } from '../../packages/tts/src/model.js';

function deferred<T>() {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>((yes) => {
    resolve = yes;
  });
  return { promise, resolve };
}
function base(capacity = 2) {
  const cache = new Set<string>();
  const submitted: BackendOptions[] = [];
  let counter = 0;
  const backend: TtsBackend = {
    capabilities: {
      family: 'test',
      variant: 'base',
      sampleRate: 24000,
      channels: 1,
      voices: [],
      languages: ['english'],
      voiceCloning: true,
      conditioning: [{ voice: 'reference', instruct: 'unsupported' }],
      textStreaming: 'segmented',
      audioStreaming: true,
    },
    prepareVoice: vi.fn(async () => {
      if (cache.size >= capacity) cache.delete(cache.values().next().value!);
      const id = `native-${++counter}`;
      cache.add(id);
      return id;
    }),
    hasVoice: vi.fn(async (id) => cache.has(id)),
    cancelVoicePreparation: vi.fn(),
    releaseVoice: vi.fn(async (id) => {
      cache.delete(id);
    }),
    dispose: vi.fn(async () => {
      cache.clear();
    }),
    start: vi.fn((_text, options) => {
      submitted.push(options);
      let at = 0;
      return {
        next: async () =>
          at++ === 0
            ? { samples: new Float32Array([0.25, 0, -0.25]), finished: false }
            : { samples: new Float32Array(), finished: true, finishReason: 'eos', synthesisMs: 1 },
        cancel: vi.fn(),
        waitFinished: async () => {},
      };
    }),
  };
  return { model: createTtsModel(backend), backend, submitted, cache };
}
const servers: ServerInstance[] = [];
afterEach(async () => {
  for (const server of servers.splice(0)) await server.close({ timeoutMs: 100 });
});
async function start(audio?: AudioServiceOptions) {
  const server = await createServer({ port: 0, host: '127.0.0.1', disableStore: true, idleClearCacheMs: 0, audio });
  servers.push(server);
  const fake = base();
  server.registry.audio.register('base', { kind: 'tts', model: fake.model });
  const url = `http://127.0.0.1:${(server.server.address() as AddressInfo).port}/v1/audio`;
  return { server, fake, url };
}
function wav(rate = 16000, channels = 2, seconds = 0.01): Buffer<ArrayBuffer> {
  const pcm = audioEncodePcm16(new Float32Array(Math.round(rate * seconds) * channels).fill(0.25));
  return Buffer.concat([audioWavHeader(rate, channels, pcm.length), pcm]);
}
async function create(url: string, fields: Record<string, string> = {}, bytes = wav(), fileField = 'audio_sample') {
  const form = new FormData();
  form.set(fileField, new Blob([bytes]), 'reference.wav');
  for (const [key, value] of Object.entries({ model: 'base', name: 'Reference', transcript: 'Hello.', ...fields }))
    form.set(key, value);
  return fetch(`${url}/voices`, { method: 'POST', body: form });
}
async function speech(url: string, id: string, extra: Record<string, unknown> = {}) {
  return fetch(`${url}/speech`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ model: 'base', input: 'Hello.', voice: { id }, ...extra }),
  });
}
async function error(response: Response, status: number, code: string, param?: string) {
  expect(response.status).toBe(status);
  expect((await response.json()).error).toMatchObject({ code, ...(param ? { param } : {}) });
}

describe('temporary HTTP voice resources', () => {
  it('maps checkpoint reference validation to 400 and recovers capacity after failure', async () => {
    const { url, fake } = await start({ maxVoiceRecords: 1, maxTempBytes: wav().length });
    vi.mocked(fake.backend.prepareVoice).mockRejectedValueOnce(new InvalidReferenceAudioError('Reference rejected'));
    await error(await create(url), 400, 'invalid_audio', 'audio_sample');
    const voice = await (await create(url)).json();
    expect(voice.status).toBe('active');
    expect((await speech(url, voice.id)).status).toBe(200);
  });
  it('rejects an unsupported cold checkpoint without loading or evicting a resident model', async () => {
    const resolveModel = vi.fn(async () => {
      throw new Error('Unexpected model load');
    });
    const server = await createServer({
      port: 0,
      host: '127.0.0.1',
      disableStore: true,
      idleClearCacheMs: 0,
      resolveModel,
      listModels: () => [
        {
          id: 'preset',
          task: 'tts',
          modality: 'audio',
          object: 'model',
          created: 1,
          owned_by: 'test',
          capabilities: { voice_cloning: false },
        },
      ],
    });
    servers.push(server);
    const fake = base();
    server.registry.audio.register('base', { kind: 'tts', model: fake.model });
    const url = `http://127.0.0.1:${(server.server.address() as AddressInfo).port}/v1/audio`;
    await error(await create(url, { model: 'preset' }), 400, 'model_capability_mismatch', 'model');
    expect(resolveModel).not.toHaveBeenCalled();
    expect(fake.backend.dispose).not.toHaveBeenCalled();
    expect(server.registry.audio.get('base')?.model).toBe(fake.model);
  });
  it('keeps release metadata atomic against an asynchronous store capacity check', async () => {
    const fake = base();
    const other = base();
    const registry = new AudioRegistry();
    registry.register('base', { kind: 'tts', model: fake.model });
    const store = new MemoryAudioVoiceStore();
    const voices = new AudioVoices(registry, 1, store);
    try {
      const reservation = await voices.reserve(fake.model);
      const prepared = await fake.model.prepareVoice({
        audio: { samples: new Float32Array([0]), sampleRate: 16000 },
        transcript: 'Hi',
      });
      const record = await voices.add('Name', 'base', fake.model, prepared, reservation);
      const listing = deferred<void>();
      const proceed = deferred<void>();
      const list = store.list.bind(store);
      vi.spyOn(store, 'list').mockImplementationOnce(async () => {
        const records = await list();
        listing.resolve();
        await proceed.promise;
        return records;
      });
      const competing = voices.reserve(other.model);
      const rejected = expect(competing).rejects.toMatchObject({ code: 'voice_capacity_exceeded' });
      await listing.promise;
      const released = voices.release(record);
      await prepared.dispose();
      proceed.resolve();
      await rejected;
      await released;
      expect(await voices.describe(record)).toMatchObject({ status: 'released' });
    } finally {
      await voices.close();
      await Promise.all([fake.model.dispose(), other.model.dispose()]);
    }
  });
  it('creates separate resource IDs, keeps stable metadata, and reclaims upload quota', async () => {
    const bytes = wav();
    const { url, fake } = await start({ maxTempBytes: bytes.length });
    const first = await create(url);
    expect(first.status).toBe(200);
    const voice = await first.json();
    expect(voice).toMatchObject({
      object: 'audio.voice',
      name: 'Reference',
      model: 'base',
      status: 'active',
      persistent: false,
      type: 'audio_sample',
    });
    expect(voice.id).toMatch(/^voice_/);
    expect(first.headers.get('location')).toBe(`/v1/audio/voices/${voice.id}`);
    expect(voice.id).not.toBe('native-1');
    expect(fake.backend.prepareVoice).toHaveBeenCalledWith(new Float32Array(160).fill(0.25), 16000, 'Hello.');
    expect(await (await fetch(`${url}/voices/${voice.id}`)).json()).toEqual(voice);
    expect((await create(url)).status).toBe(200);
    const catalog = await (await fetch(`${url}/voices?model=base`)).json();
    expect(catalog.data).toEqual([
      expect.objectContaining({ model: 'base', http_speech: true, http_voice_cloning: true, voices: [] }),
    ]);
    expect(JSON.stringify(voice)).not.toMatch(/samples|transcript|native-|path/);
  });
  it.each([16000, 22050, 24000, 44100, 48000, 96000])('preserves %i Hz for native preparation', async (rate) => {
    const { url, fake } = await start();
    const res = await fetch(`${url}/voices?model=base&name=Test&transcript=Hello`, {
      method: 'POST',
      headers: { 'Content-Type': 'audio/wav' },
      body: wav(rate, 1),
    });
    expect(res.status).toBe(200);
    expect(fake.backend.prepareVoice).toHaveBeenCalledWith(expect.any(Float32Array), rate, 'Hello');
  });
  it.each([{ response_format: 'wav' }, { response_format: 'pcm' }, { response_format: 'pcm', stream_format: 'sse' }])(
    'uses prepared references for $response_format / $stream_format',
    async (format) => {
      const { url, fake } = await start();
      const voice = await (await create(url)).json();
      const res = await speech(url, voice.id, format);
      expect(res.status).toBe(200);
      const bytes = Buffer.from(await res.arrayBuffer());
      expect(bytes.length).toBeGreaterThan(0);
      expect(fake.submitted[0].preparedVoiceId).toBe('native-1');
      if (format.stream_format === 'sse') expect(bytes.toString()).toContain('speech.audio.done');
    },
  );
  it('accepts reference IDs in NDJSON configuration', async () => {
    const { url, fake } = await start();
    const voice = await (await create(url)).json();
    const res = await fetch(`${url}/speech/stream`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/x-ndjson' },
      body: [
        { type: 'config', model: 'base', voice: { id: voice.id }, response_format: 'pcm' },
        { type: 'text', text: 'Hello.' },
        { type: 'flush' },
      ]
        .map((item) => JSON.stringify(item))
        .join('\n'),
    });
    expect(res.status).toBe(200);
    expect((await res.arrayBuffer()).byteLength).toBe(6);
    expect(fake.submitted[0].preparedVoiceId).toBe('native-1');
  });
  it('releases once, preserves a released tombstone and never synthesizes with it', async () => {
    const { url, fake } = await start();
    const voice = await (await create(url)).json();
    for (let i = 0; i < 2; i++) {
      const res = await fetch(`${url}/voices/${voice.id}`, { method: 'DELETE' });
      expect(await res.json()).toEqual({ id: voice.id, object: 'audio.voice.deleted', deleted: true });
    }
    expect(fake.backend.releaseVoice).toHaveBeenCalledExactlyOnceWith('native-1');
    expect(await (await fetch(`${url}/voices/${voice.id}`)).json()).toMatchObject({
      status: 'released',
      released_at: expect.any(Number),
    });
    await error(await speech(url, voice.id), 410, 'voice_unavailable', 'voice');
    expect(fake.submitted).toEqual([]);
    await error(await fetch(`${url}/voices/unknown`), 404, 'voice_not_found');
  });
  it('observes native cache eviction and external native releases', async () => {
    const { url, fake } = await start();
    const first = await (await create(url)).json();
    await create(url);
    const third = await (await create(url)).json();
    expect(await (await fetch(`${url}/voices/${first.id}`)).json()).toMatchObject({ status: 'evicted' });
    await error(await speech(url, first.id), 410, 'voice_unavailable');
    fake.cache.clear();
    await error(await speech(url, third.id), 410, 'voice_unavailable');
    expect(fake.submitted).toEqual([]);
  });
  it('shares aliases but revokes preparations when the final model binding is removed', async () => {
    const { url, fake, server } = await start();
    server.registry.audio.register('alias', { kind: 'tts', model: fake.model });
    const voice = await (await create(url)).json();
    server.registry.audio.register('base', { kind: 'tts', model: fake.model });
    server.registry.audio.unregister('base');
    expect((await speech(url, voice.id, { model: 'alias' })).status).toBe(200);
    expect(await (await fetch(`${url}/voices/${voice.id}`)).json()).toMatchObject({ status: 'active' });
    server.registry.audio.unregister('alias');
    server.registry.audio.register('base', { kind: 'tts', model: fake.model });
    expect(await (await fetch(`${url}/voices/${voice.id}`)).json()).toMatchObject({ status: 'evicted' });
    await error(await speech(url, voice.id), 410, 'voice_unavailable');
  });
  it('never sends a reference to a different model or accepts instruction controls for Base', async () => {
    const { url, server } = await start();
    server.registry.audio.register('other', { kind: 'tts', model: base().model });
    const voice = await (await create(url)).json();
    await error(await speech(url, voice.id, { model: 'other' }), 400, 'voice_model_mismatch');
    await error(await speech(url, voice.id, { model: 'missing' }), 404, 'model_not_found');
    await error(await speech(url, voice.id, { instructions: 'Whisper' }), 400, 'invalid_request', 'instructions');
  });
  it('bounds records before native dispatch and reuses tombstone capacity', async () => {
    const { url, fake } = await start({ maxVoiceRecords: 1 });
    const first = await (await create(url)).json();
    const full = await create(url);
    expect(full.headers.get('retry-after')).toBe('1');
    await error(full, 429, 'voice_capacity_exceeded');
    expect(fake.backend.prepareVoice).toHaveBeenCalledTimes(1);
    await fetch(`${url}/voices/${first.id}`, { method: 'DELETE' });
    expect((await create(url)).status).toBe(200);
    await error(await fetch(`${url}/voices/${first.id}`), 404, 'voice_not_found');
    // An externally evicted native entry also releases metadata capacity.
    fake.cache.clear();
    expect((await create(url)).status).toBe(200);
  });
  it('validates reference parameters and WAVs before preparing anything', async () => {
    const { url, fake } = await start({ maxReferenceSeconds: 1 });
    await error(await create(url, { transcript: '' }), 400, 'invalid_request', 'transcript');
    await error(await create(url, { type: 'other' }), 400, 'invalid_request', 'type');
    await error(await create(url, { consent: 'not-implemented' }), 400, 'invalid_request', 'consent');
    await error(await create(url, {}, wav(), 'file'), 400, 'invalid_request', 'audio_sample');
    await error(await create(url, {}, Buffer.from('invalid')), 400, 'invalid_audio', 'audio_sample');
    await error(await create(url, {}, wav(16000, 1, 1.01)), 400, 'invalid_request', 'audio_sample');
    expect(fake.backend.prepareVoice).not.toHaveBeenCalled();
  });
  it('keeps a global reservation across concurrent preparations on different models', async () => {
    const { url, fake, server } = await start({ maxVoiceRecords: 1 });
    const other = base();
    server.registry.audio.register('other', { kind: 'tts', model: other.model });
    const native = deferred<string>();
    vi.mocked(fake.backend.prepareVoice).mockImplementationOnce(() => native.promise);
    const first = create(url);
    await vi.waitFor(() => expect(fake.backend.prepareVoice).toHaveBeenCalledTimes(1));
    await error(await create(url, { model: 'other' }), 429, 'voice_capacity_exceeded');
    expect(other.backend.prepareVoice).not.toHaveBeenCalled();
    fake.cache.add('reserved');
    native.resolve('reserved');
    expect((await first).status).toBe(200);
  });
  it('queues a concurrent same-reference create behind native preparation and deletes only one ID', async () => {
    const { url, fake, server } = await start();
    const native = deferred<string>();
    vi.mocked(fake.backend.prepareVoice).mockImplementationOnce(async () => {
      const id = await native.promise;
      fake.cache.add(id);
      return id;
    });
    const first = create(url);
    const second = create(url);
    await vi.waitFor(() => expect(server.registry.audio.pending).toBe(2));
    expect(fake.backend.prepareVoice).toHaveBeenCalledTimes(1);
    native.resolve('shared');
    const [a, b] = await Promise.all([first, second]);
    expect([a.status, b.status]).toEqual([200, 200]);
    const [voiceA, voiceB] = [await a.json(), await b.json()];
    expect(voiceA.id).not.toBe(voiceB.id);
    expect([voiceA.status, voiceB.status]).toEqual(['active', 'active']);
    const calls = vi.mocked(fake.backend.prepareVoice).mock.calls;
    expect(calls).toHaveLength(2);
    expect(calls[1]).toEqual(calls[0]);
    expect((await fetch(`${url}/voices/${voiceA.id}`, { method: 'DELETE' })).status).toBe(200);
    expect(fake.backend.releaseVoice).toHaveBeenCalledTimes(1);
    expect(fake.cache.size).toBe(1);
    expect(await (await fetch(`${url}/voices/${voiceB.id}`)).json()).toMatchObject({ status: 'active' });
    const again = await create(url);
    expect(again.status).toBe(200);
    expect(await again.json()).toMatchObject({ status: 'active' });
    expect(calls).toHaveLength(3);
    expect(calls[2]).toEqual(calls[0]);
  });
  it('reacquires a surviving alias lease before deleting behind another synthesis', async () => {
    const { url, fake, server } = await start();
    const voice = await (await create(url)).json();
    server.registry.audio.register('alias', { kind: 'tts', model: fake.model });
    const gates = [deferred<void>(), deferred<void>()];
    let starts = 0;
    vi.spyOn(fake.backend, 'start').mockImplementation(() => {
      const gate = gates[starts++];
      let sent = false;
      return {
        next: async () => {
          await gate.promise;
          if (sent) return { samples: new Float32Array(), finished: true, finishReason: 'eos' };
          sent = true;
          return { samples: new Float32Array([0]), finished: false };
        },
        cancel: vi.fn(),
        waitFinished: async () => {},
      };
    });
    const first = speech(url, voice.id, { response_format: 'pcm' });
    await vi.waitFor(() => expect(starts).toBe(1));
    const deletion = fetch(`${url}/voices/${voice.id}`, { method: 'DELETE' });
    await vi.waitFor(() => expect(server.registry.audio.pending).toBe(2));
    const second = speech(url, voice.id, { model: 'alias', response_format: 'pcm' });
    await vi.waitFor(() => expect(server.registry.audio.pending).toBe(3));
    server.registry.audio.unregister('base');
    gates[0].resolve();
    await (await first).arrayBuffer();
    await vi.waitFor(() => expect(starts).toBe(2));
    expect(fake.backend.releaseVoice).not.toHaveBeenCalled();
    gates[1].resolve();
    await (await second).arrayBuffer();
    expect((await deletion).status).toBe(200);
    expect(fake.backend.releaseVoice).toHaveBeenCalledExactlyOnceWith('native-1');
  });
  it('rolls back a disconnected creation only after native completion and permits subsequent uploads', async () => {
    const { url, fake, server } = await start({ maxVoiceRecords: 1, maxTempBytes: wav().length });
    const native = deferred<string>();
    vi.mocked(fake.backend.prepareVoice).mockImplementationOnce(() => native.promise);
    const controller = new AbortController();
    const response = fetch(`${url}/voices?model=base&name=Test&transcript=Hi`, {
      method: 'POST',
      headers: { 'Content-Type': 'audio/wav' },
      body: wav(),
      signal: controller.signal,
    }).catch(() => undefined);
    await vi.waitFor(() => expect(fake.backend.prepareVoice).toHaveBeenCalledTimes(1));
    controller.abort();
    await response;
    await vi.waitFor(() => expect(fake.backend.cancelVoicePreparation).toHaveBeenCalledTimes(1));
    expect(server.registry.audio.pending).toBe(1);
    native.resolve('abandoned');
    await vi.waitFor(() => expect(server.registry.audio.pending).toBe(0));
    expect(fake.backend.releaseVoice).toHaveBeenCalledExactlyOnceWith('abandoned');
    expect((await create(url)).status).toBe(200);
  });
  it('rejects oversized uploads using the reference field name and refunds disk capacity', async () => {
    const bytes = wav();
    const { url, fake } = await start({ maxUploadBytes: bytes.length + 1, maxTempBytes: bytes.length + 1 });
    await error(await create(url, {}, wav(48000)), 413, 'request_too_large', 'audio_sample');
    expect((await create(url)).status).toBe(200);
    expect(fake.backend.prepareVoice).toHaveBeenCalledTimes(1);
  });
  it('waits for native preparation acknowledgement during shutdown and releases a late success', async () => {
    const { url, fake, server } = await start();
    const native = deferred<string>();
    vi.mocked(fake.backend.prepareVoice).mockImplementationOnce(() => native.promise);
    const response = create(url);
    await vi.waitFor(() => expect(fake.backend.prepareVoice).toHaveBeenCalledTimes(1));
    let closed = false;
    const closing = server.close({ timeoutMs: 1000 }).then(() => {
      closed = true;
    });
    await vi.waitFor(() => expect(fake.backend.cancelVoicePreparation).toHaveBeenCalledTimes(1));
    expect(closed).toBe(false);
    expect(fake.backend.dispose).not.toHaveBeenCalled();
    native.resolve('late-native');
    const res = await response;
    expect(res.headers.get('x-should-retry')).toBe('false');
    await error(res, 503, 'server_shutdown');
    await closing;
    expect(fake.backend.releaseVoice).toHaveBeenCalledExactlyOnceWith('late-native');
    expect(fake.backend.dispose).toHaveBeenCalledTimes(1);
  });
});
