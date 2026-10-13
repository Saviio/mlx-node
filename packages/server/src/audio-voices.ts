import { randomUUID } from 'node:crypto';

import type { PreparedVoice, TtsModel } from '@mlx-node/tts';

import { AudioError } from './audio-io.js';
import type { AudioRegistry } from './audio-registry.js';

/** Resource metadata is independent of native cache IDs and model instances.
 * A future durable store can retain source material alongside these records. */
export interface AudioVoiceRecord {
  readonly id: string;
  readonly name: string;
  readonly model: string;
  readonly created_at: number;
  readonly released_at?: number;
}
export interface AudioVoiceStore {
  readonly persistent: boolean;
  get(id: string): Promise<AudioVoiceRecord | undefined>;
  list(): Promise<readonly AudioVoiceRecord[]>;
  put(record: AudioVoiceRecord): Promise<void>;
  remove(id: string): Promise<void>;
  close(): Promise<void>;
}
export class MemoryAudioVoiceStore implements AudioVoiceStore {
  readonly persistent = false;
  private readonly records = new Map<string, AudioVoiceRecord>();
  async get(id: string) {
    return this.records.get(id);
  }
  async list() {
    return [...this.records.values()];
  }
  async put(record: AudioVoiceRecord) {
    this.records.set(record.id, Object.freeze({ ...record }));
  }
  async remove(id: string) {
    this.records.delete(id);
  }
  async close() {
    this.records.clear();
  }
}
interface RuntimeVoice {
  model: TtsModel;
  voice: PreparedVoice;
}

/** Manages ephemeral preparations separately from the resource repository.
 * Native operations must run under the caller's model lease. Repository writes
 * are serialized independently, so future async storage cannot race capacity. */
export class AudioVoices {
  private readonly runtime = new Map<string, RuntimeVoice>();
  private mutation: Promise<unknown> = Promise.resolve();
  private readonly reservations = new Set<symbol>();
  private readonly unsubscribe: () => void;
  constructor(
    private readonly registry: AudioRegistry,
    private readonly maxRecords: number,
    private readonly store: AudioVoiceStore = new MemoryAudioVoiceStore(),
  ) {
    this.unsubscribe = registry.onUnregister((binding) => {
      for (const [id, runtime] of this.runtime) if (runtime.model === binding.model) this.runtime.delete(id);
    });
  }
  private update<T>(run: () => Promise<T>): Promise<T> {
    const result = this.mutation.then(run);
    this.mutation = result.catch(() => {});
    return result;
  }
  async record(id: string): Promise<AudioVoiceRecord> {
    const record = await this.store.get(id);
    if (!record) throw new AudioError(404, 'voice_not_found', 'Voice not found or its metadata has expired', 'voice');
    return record;
  }
  binding(record: AudioVoiceRecord): RuntimeVoice | undefined {
    return this.runtime.get(record.id);
  }
  async reserve(model: TtsModel): Promise<symbol> {
    await this.refresh(model);
    return this.update(async () => {
      const records = await this.store.list();
      if (records.length + this.reservations.size >= this.maxRecords) {
        const expired = records.find((record) => !this.runtime.has(record.id));
        if (!expired) throw new AudioError(429, 'voice_capacity_exceeded', 'Voice record capacity is full');
        await this.store.remove(expired.id);
      }
      const reservation = Symbol('voice reservation');
      this.reservations.add(reservation);
      return reservation;
    });
  }
  unreserve(reservation: symbol): void {
    this.reservations.delete(reservation);
  }
  async add(
    name: string,
    modelName: string,
    model: TtsModel,
    voice: PreparedVoice,
    reservation: symbol,
  ): Promise<AudioVoiceRecord> {
    await this.refresh(model);
    return this.update(async () => {
      if (!this.reservations.has(reservation)) throw new Error('Invalid voice reservation');
      const record = Object.freeze({
        id: `voice_${randomUUID()}`,
        name,
        model: modelName,
        created_at: Math.floor(Date.now() / 1000),
      });
      await this.store.put(record);
      this.runtime.set(record.id, { model, voice });
      this.reservations.delete(reservation);
      return record;
    });
  }
  private async refresh(model: TtsModel): Promise<void> {
    // Preparation may have evicted an older native entry. Observe its actual
    // state before pruning metadata; never duplicate the native cache policy.
    for (const [id, runtime] of this.runtime) {
      if (runtime.model === model && (await runtime.voice.status()) !== 'active') this.runtime.delete(id);
    }
  }
  async describe(record: AudioVoiceRecord) {
    // Read fresh metadata after waiting for a model lease: DELETE may have run.
    record = await this.record(record.id);
    const runtime = this.runtime.get(record.id);
    const status = record.released_at !== undefined ? 'released' : runtime ? await runtime.voice.status() : 'evicted';
    if (status !== 'active') this.runtime.delete(record.id);
    return { ...record, object: 'audio.voice', type: 'audio_sample', persistent: this.store.persistent, status };
  }
  async resolve(record: AudioVoiceRecord, model: TtsModel): Promise<PreparedVoice> {
    const runtime = this.runtime.get(record.id);
    if (runtime && runtime.model !== model)
      throw new AudioError(400, 'voice_model_mismatch', 'Voice belongs to a different model instance', 'voice');
    if (!runtime || (await this.describe(record)).status !== 'active')
      throw new AudioError(
        410,
        'voice_unavailable',
        'Voice was released or evicted; upload the reference again',
        'voice',
      );
    return runtime.voice;
  }
  async release(record: AudioVoiceRecord): Promise<void> {
    const runtime = this.runtime.get(record.id);
    await runtime?.voice.dispose();
    await this.update(async () => {
      const current = await this.record(record.id);
      if (current.released_at === undefined)
        await this.store.put({ ...current, released_at: Math.floor(Date.now() / 1000) });
      // Keep the binding until the tombstone is committed. A reservation on a
      // different model must not prune this record mid-release in an async store.
      this.runtime.delete(record.id);
    });
  }
  modelName(record: AudioVoiceRecord): string | undefined {
    const runtime = this.runtime.get(record.id);
    return runtime && this.registry.nameForModel(runtime.model);
  }
  async close(): Promise<void> {
    this.unsubscribe();
    this.runtime.clear();
    this.reservations.clear();
    await this.mutation;
    await this.store.close();
  }
}
