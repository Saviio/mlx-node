import { mkdtemp, open, rm, readFile, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

import { it, expect, vi } from 'vite-plus/test';

import { InvalidWavError, WavReader, WavWriter, readWav } from '../src/audio.js';

vi.mock('node:fs/promises', async (importOriginal) => {
  const fs = await importOriginal<typeof import('node:fs/promises')>();
  return { ...fs, open: vi.fn(fs.open) };
});

it('writes a bounded stream with a correct RIFF header, interleaved channels and clipped PCM16', async () => {
  const dir = await mkdtemp(join(tmpdir(), 'mlx-tts-'));
  const path = join(dir, 'audio.wav');
  try {
    const writer = await WavWriter.open(path, 24000, 2);
    await writer.write(new Float32Array([-2, 2, -0.5, 0.5]));
    await writer.write(new Float32Array([0, 0]));
    await writer.close();
    const bytes = await readFile(path);
    expect(bytes.readUInt32LE(4)).toBe(bytes.length - 8);
    expect(bytes.readUInt32LE(40)).toBe(12);
    const audio = await readWav(path);
    expect(audio.sampleRate).toBe(24000);
    expect(audio.channels).toBe(2);
    expect([...audio.samples]).toEqual([-1, 32767 / 32768, -0.5, 0.5, 0, 0]);
    await expect(writer.write(new Float32Array([0, 0]))).rejects.toThrow('closed');
  } finally {
    await rm(dir, { recursive: true, force: true });
  }
});

it('rejects floating WAV samples that overflow the public Float32 PCM format', async () => {
  const dir = await mkdtemp(join(tmpdir(), 'mlx-tts-'));
  const path = join(dir, 'float64.wav');
  try {
    const bytes = Buffer.alloc(52);
    bytes.write('RIFF');
    bytes.writeUInt32LE(44, 4);
    bytes.write('WAVEfmt ', 8);
    bytes.writeUInt32LE(16, 16);
    bytes.writeUInt16LE(3, 20);
    bytes.writeUInt16LE(1, 22);
    bytes.writeUInt32LE(24000, 24);
    bytes.writeUInt32LE(24000 * 8, 28);
    bytes.writeUInt16LE(8, 32);
    bytes.writeUInt16LE(64, 34);
    bytes.write('data', 36);
    bytes.writeUInt32LE(8, 40);
    for (const sample of [1e100, -1e100, Infinity, NaN]) {
      bytes.writeDoubleLE(sample, 44);
      await writeFile(path, bytes);
      await expect(readWav(path)).rejects.toThrow('non-finite');
      await expect(readWav(path)).rejects.toBeInstanceOf(InvalidWavError);
    }
    bytes.writeDoubleLE(0.25, 44);
    await writeFile(path, bytes);
    expect([...(await readWav(path)).samples]).toEqual([0.25]);
  } finally {
    await rm(dir, { recursive: true, force: true });
  }
});

it('distinguishes malformed WAV containers and codec formats from filesystem errors', async () => {
  const dir = await mkdtemp(join(tmpdir(), 'mlx-tts-'));
  const path = join(dir, 'audio.wav');
  try {
    await expect(WavReader.open(path)).rejects.toMatchObject({ code: 'ENOENT' });
    await writeFile(path, 'invalid');
    await expect(WavReader.open(path)).rejects.toBeInstanceOf(InvalidWavError);
    const writer = await WavWriter.open(path, 24000);
    await writer.write(new Float32Array([0]));
    await writer.close();
    const bytes = await readFile(path);
    bytes.writeUInt16LE(99, 20);
    await writeFile(path, bytes);
    await expect(WavReader.open(path)).rejects.toBeInstanceOf(InvalidWavError);
    const failure = Object.assign(new Error('WAV disk failed'), { code: 'EIO' });
    vi.mocked(open).mockRejectedValueOnce(failure);
    await expect(WavReader.open(path)).rejects.toBe(failure);
  } finally {
    await rm(dir, { recursive: true, force: true });
  }
});

it.each([
  [20, 24],
  [24, 32],
])('reads %i-bit PCM in %i-bit extensible WAV containers without changing amplitude', async (validBits, bits) => {
  const dir = await mkdtemp(join(tmpdir(), 'mlx-tts-'));
  const path = join(dir, 'extensible.wav');
  const samples = [0.5, -0.25, -1, 0.25];
  const width = bits / 8;
  const bytes = Buffer.alloc(68 + width * samples.length);
  bytes.write('RIFF');
  bytes.writeUInt32LE(bytes.length - 8, 4);
  bytes.write('WAVEfmt ', 8);
  bytes.writeUInt32LE(40, 16);
  bytes.writeUInt16LE(0xfffe, 20);
  bytes.writeUInt16LE(2, 22);
  bytes.writeUInt32LE(48000, 24);
  bytes.writeUInt32LE(48000 * width * 2, 28);
  bytes.writeUInt16LE(width * 2, 32);
  bytes.writeUInt16LE(bits, 34);
  bytes.writeUInt16LE(22, 36);
  bytes.writeUInt16LE(validBits, 38);
  bytes.writeUInt32LE(3, 40); // Front left and right.
  Buffer.from('0100000000001000800000aa00389b71', 'hex').copy(bytes, 44);
  bytes.write('data', 60);
  bytes.writeUInt32LE(width * samples.length, 64);
  samples.forEach((sample, i) => bytes.writeIntLE(sample * 2 ** (bits - 1), 68 + i * width, width));
  try {
    await writeFile(path, bytes);
    const audio = await readWav(path);
    expect(audio.sampleRate).toBe(48000);
    expect(audio.channels).toBe(2);
    expect([...audio.samples]).toEqual(samples);
    const reader = await WavReader.open(path);
    try {
      const mono = [];
      for await (const chunk of reader.chunks(1, true)) mono.push(...chunk);
      expect(mono).toEqual([0.125, -0.375]);
    } finally {
      await reader.close();
    }
  } finally {
    await rm(dir, { recursive: true, force: true });
  }
});

it('validates all WAV header fields before truncating an existing output file', async () => {
  const dir = await mkdtemp(join(tmpdir(), 'mlx-tts-'));
  const path = join(dir, 'existing.wav');
  try {
    await writeFile(path, 'preserve existing audio');
    for (const [sampleRate, channels] of [
      [2 ** 32, 1],
      [2 ** 31, 1],
      [2 ** 26, 32],
    ]) {
      await expect(WavWriter.open(path, sampleRate, channels)).rejects.toThrow('Invalid WAV format');
      expect(await readFile(path, 'utf8')).toBe('preserve existing audio');
    }
  } finally {
    await rm(dir, { recursive: true, force: true });
  }
});

it('makes every concurrent close await file closure and the final header', async () => {
  const dir = await mkdtemp(join(tmpdir(), 'mlx-tts-'));
  const path = join(dir, 'audio.wav');
  let release!: () => void;
  const ready = new Promise<void>((resolve) => {
    release = resolve;
  });
  try {
    const fs = await vi.importActual<typeof import('node:fs/promises')>('node:fs/promises');
    let closeCalls = 0;
    vi.mocked(open).mockImplementationOnce(async (...args) => {
      const file = await fs.open(...args);
      const closeFile = file.close.bind(file);
      vi.spyOn(file, 'close').mockImplementation(async () => {
        closeCalls++;
        await ready;
        await closeFile();
      });
      return file;
    });
    const writer = await WavWriter.open(path, 24000);
    await writer.write(new Float32Array([0.25]));
    const first = writer.close();
    await vi.waitFor(() => expect(closeCalls).toBe(1));
    let secondClosed = false;
    const second = writer.close().then(() => {
      secondClosed = true;
    });
    await Promise.resolve();
    expect(secondClosed).toBe(false);
    release();
    await Promise.all([first, second]);
    expect(secondClosed).toBe(true);
    expect([...(await readWav(path)).samples]).toEqual([0.25]);
  } finally {
    release();
    await rm(dir, { recursive: true, force: true });
  }
});

it('preserves a write failure on repeated close calls', async () => {
  const dir = await mkdtemp(join(tmpdir(), 'mlx-tts-'));
  const path = join(dir, 'audio.wav');
  try {
    const fs = await vi.importActual<typeof import('node:fs/promises')>('node:fs/promises');
    const error = new Error('disk write failed');
    vi.mocked(open).mockImplementationOnce(async (...args) => {
      const file = await fs.open(...args);
      const write = file.write.bind(file);
      vi.spyOn(file, 'write').mockImplementationOnce(write).mockRejectedValueOnce(error);
      return file;
    });
    const writer = await WavWriter.open(path, 24000);
    await expect(writer.write(new Float32Array([0.25]))).rejects.toBe(error);
    await expect(writer.close()).rejects.toBe(error);
    await expect(writer.close()).rejects.toBe(error);
  } finally {
    await rm(dir, { recursive: true, force: true });
  }
});
