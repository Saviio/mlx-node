import { open } from 'node:fs/promises';
import type { FileHandle } from 'node:fs/promises';

import type { AudioChunk, AudioData } from './types.js';

/** Malformed or unsupported WAV data; filesystem failures keep their original type. */
export class InvalidWavError extends Error {
  override readonly name = 'InvalidWavError';
}

function decodeWav<T>(decode: () => T): T {
  try {
    return decode();
  } catch (cause) {
    throw new InvalidWavError(cause instanceof Error ? cause.message : String(cause), { cause });
  }
}

/** A seekable WAV reader: metadata stays small and decoding is bounded per pull. */
export class WavReader {
  private constructor(
    readonly file: FileHandle,
    readonly format: import('@mlx-node/core').AudioWavFormat,
    readonly blocks: readonly { offset: number; bytes: number }[],
    readonly frames: number,
  ) {}
  get sampleRate(): number {
    return this.format.sampleRate;
  }
  get channels(): number {
    return this.format.channels;
  }
  static async open(path: string): Promise<WavReader> {
    const { audioWavFormat } = await import('@mlx-node/core');
    const file = await open(path, 'r');
    const read = async (offset: number, size: number): Promise<Buffer> => {
      const buffer = Buffer.alloc(size);
      let done = 0;
      while (done < size) {
        const { bytesRead } = await file.read(buffer, done, size - done, offset + done);
        if (!bytesRead) throw new InvalidWavError('Truncated WAV file');
        done += bytesRead;
      }
      return buffer;
    };
    try {
      const header = await read(0, 12);
      if (header.toString('ascii', 0, 4) !== 'RIFF' || header.toString('ascii', 8, 12) !== 'WAVE')
        throw new InvalidWavError('Expected a RIFF/WAVE file');
      const end = header.readUInt32LE(4) + 8;
      if (end < 12 || end !== (await file.stat()).size) throw new InvalidWavError('Invalid WAV container length');
      let format: import('@mlx-node/core').AudioWavFormat | undefined;
      const blocks: { offset: number; bytes: number }[] = [];
      let at = 12;
      let chunks = 0;
      while (at < end) {
        if (++chunks > 65536 || at + 8 > end) throw new InvalidWavError('Invalid WAV chunks');
        const chunk = await read(at, 8);
        const size = chunk.readUInt32LE(4);
        const body = at + 8;
        if (body + size + (size & 1) > end) throw new InvalidWavError('Truncated WAV chunk');
        const name = chunk.toString('ascii', 0, 4);
        if (name === 'fmt ') {
          if (format) throw new InvalidWavError('Duplicate WAV format');
          const bytes = await read(body, Math.min(size, 40));
          format = decodeWav(() => audioWavFormat(bytes));
        } else if (name === 'data') blocks.push({ offset: body, bytes: size });
        at = body + size + (size & 1);
      }
      if (!format || !blocks.length) throw new InvalidWavError('WAV is missing format or samples');
      if (blocks.some((b) => b.bytes % format.blockAlign)) throw new InvalidWavError('Incomplete WAV sample frame');
      const frames = blocks.reduce((n, b) => n + b.bytes / format.blockAlign, 0);
      return new WavReader(file, format, blocks, frames);
    } catch (error) {
      await file.close();
      throw error;
    }
  }
  async *chunks(framesPerChunk = 16384, mono = false): AsyncGenerator<Float32Array> {
    if (!Number.isSafeInteger(framesPerChunk) || framesPerChunk < 1) throw new Error('Invalid WAV chunk size');
    const { audioDecodeWavFrames } = await import('@mlx-node/core');
    for (const block of this.blocks) {
      let at = 0;
      while (at < block.bytes) {
        const size = Math.min(framesPerChunk * this.format.blockAlign, block.bytes - at);
        const buffer = Buffer.allocUnsafe(size);
        let done = 0;
        while (done < size) {
          const { bytesRead } = await this.file.read(buffer, done, size - done, block.offset + at + done);
          if (!bytesRead) throw new InvalidWavError('Truncated WAV samples');
          done += bytesRead;
        }
        yield decodeWav(() => audioDecodeWavFrames(buffer, this.format, mono));
        at += size;
      }
    }
  }
  close(): Promise<void> {
    return this.file.close();
  }
}

/** Read uncompressed RIFF/WAVE audio, preserving source sample rate and channels. */
export async function readWav(path: string): Promise<AudioData> {
  const reader = await WavReader.open(path);
  try {
    const samples = new Float32Array(reader.frames * reader.channels);
    let offset = 0;
    for await (const chunk of reader.chunks()) {
      samples.set(chunk, offset);
      offset += chunk.length;
    }
    return { samples, sampleRate: reader.sampleRate, channels: reader.channels };
  } finally {
    await reader.close();
  }
}

/** Streaming PCM16 WAV writer. Memory use is bounded by the caller's chunk. */
export class WavWriter {
  readonly #file: FileHandle;
  #bytes = 0;
  #closed = false;
  #closing: Promise<void> | undefined;
  #pending = Promise.resolve();
  private constructor(
    file: FileHandle,
    readonly sampleRate: number,
    readonly channels: number,
    private readonly codec: Pick<typeof import('@mlx-node/core'), 'audioWavHeader' | 'audioEncodePcm16'>,
  ) {
    this.#file = file;
  }
  static async open(path: string, sampleRate: number, channels = 1): Promise<WavWriter> {
    if (
      !Number.isSafeInteger(sampleRate) ||
      sampleRate < 1 ||
      !Number.isSafeInteger(channels) ||
      channels < 1 ||
      channels > 32 ||
      sampleRate * channels * 2 > 0xffffffff
    )
      throw new Error('Invalid WAV format');
    const codec = await import('@mlx-node/core');
    const file = await open(path, 'w');
    const writer = new WavWriter(file, sampleRate, channels, codec);
    try {
      await writer.#header();
      return writer;
    } catch (error) {
      await file.close();
      throw error;
    }
  }
  async #header() {
    const header = this.codec.audioWavHeader(this.sampleRate, this.channels, this.#bytes);
    await this.#writeAll(header, 0);
  }
  async #writeAll(buffer: Buffer, position: number) {
    let offset = 0;
    while (offset < buffer.length) {
      const { bytesWritten } = await this.#file.write(buffer, offset, buffer.length - offset, position + offset);
      if (!bytesWritten) throw new Error('WAV write made no progress');
      offset += bytesWritten;
    }
  }
  write(chunk: AudioChunk | Float32Array): Promise<void> {
    if (this.#closed) return Promise.reject(new Error('WAV writer closed'));
    if (!(chunk instanceof Float32Array) && (chunk.sampleRate !== this.sampleRate || chunk.channels !== this.channels))
      return Promise.reject(new Error('WAV chunk format changed'));
    const samples = chunk instanceof Float32Array ? chunk : chunk.samples;
    if (samples.length % this.channels) return Promise.reject(new Error('Incomplete PCM frame'));
    let buffer: Buffer;
    try {
      buffer = this.codec.audioEncodePcm16(samples);
    } catch (error) {
      return Promise.reject(error);
    }
    this.#pending = this.#pending.then(async () => {
      if (this.#bytes + buffer.length > 0xffffffff - 36) throw new Error('WAV exceeds the RIFF 4 GiB limit');
      await this.#writeAll(buffer, 44 + this.#bytes);
      this.#bytes += buffer.length;
    });
    return this.#pending;
  }
  close(): Promise<void> {
    if (this.#closing) return this.#closing;
    this.#closed = true;
    this.#closing = (async () => {
      try {
        await this.#pending;
        await this.#header();
      } finally {
        await this.#file.close();
      }
    })();
    return this.#closing;
  }
}

export interface PlaybackStats {
  playedSeconds: number;
  underruns: number;
  firstPlaybackMs?: number | null;
}
export interface AudioPlayer {
  write(samples: Float32Array): Promise<void>;
  finish(): Promise<PlaybackStats>;
  cancel(): void;
}
/** CoreAudio performs source-rate conversion. The device callback only reads PCM. */
export async function createAudioPlayer(
  sampleRate: number,
  channels = 1,
  options: { bufferSeconds?: number; prebufferSeconds?: number } = {},
): Promise<AudioPlayer> {
  if (process.platform !== 'darwin') throw new Error('PCM device playback currently requires macOS');
  const { PcmPlayer } = await import('@mlx-node/core');
  return PcmPlayer.open(sampleRate, channels, options.bufferSeconds, options.prebufferSeconds);
}
