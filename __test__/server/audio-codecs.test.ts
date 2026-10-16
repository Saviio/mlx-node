import { audioDecodeWavFrames, audioWavFormat } from '@mlx-node/core';
import { expect, it } from 'vite-plus/test';

it.each([
  [1, 8],
  [1, 16],
  [1, 24],
  [1, 32],
  [3, 32],
  [3, 64],
])('decodes WAV encoding %s / %s-bit in complete frames', (encoding, bits) => {
  const width = bits / 8,
    bytes = Buffer.alloc(width * 2);
  if (encoding === 3) {
    if (bits === 32) {
      bytes.writeFloatLE(-0.5);
      bytes.writeFloatLE(0.5, width);
    } else {
      bytes.writeDoubleLE(-0.5);
      bytes.writeDoubleLE(0.5, width);
    }
  } else if (bits === 8) {
    bytes[0] = 64;
    bytes[1] = 192;
  } else {
    bytes.writeIntLE(-(2 ** (bits - 2)), 0, width);
    bytes.writeIntLE(2 ** (bits - 2), width, width);
  }
  const format = { encoding, sampleRate: 48000, channels: 2, bitsPerSample: bits, blockAlign: width * 2 };
  expect([...audioDecodeWavFrames(bytes, format, false)]).toEqual([-0.5, 0.5]);
  expect([...audioDecodeWavFrames(bytes, format, true)]).toEqual([0]);
  expect(() => audioDecodeWavFrames(bytes.subarray(1), format, true)).toThrow(/frame/);
});
it('validates extensible subformat, byte rate and hostile public format dimensions', () => {
  const fmt = Buffer.alloc(40);
  fmt.writeUInt16LE(0xfffe);
  fmt.writeUInt16LE(2, 2);
  fmt.writeUInt32LE(96000, 4);
  fmt.writeUInt32LE(384000, 8);
  fmt.writeUInt16LE(4, 12);
  fmt.writeUInt16LE(16, 14);
  fmt.writeUInt16LE(22, 16);
  fmt.writeUInt16LE(16, 18);
  Buffer.from('0100000000001000800000aa00389b71', 'hex').copy(fmt, 24);
  expect(audioWavFormat(fmt)).toMatchObject({ sampleRate: 96000, channels: 2, encoding: 1 });
  fmt[8] = 1;
  expect(() => audioWavFormat(fmt)).toThrow();
  expect(() =>
    audioDecodeWavFrames(
      Buffer.alloc(4),
      { encoding: 1, sampleRate: 16000, channels: 32, bitsPerSample: 0xffffffff, blockAlign: 0 },
      true,
    ),
  ).toThrow();
});
it.each([NaN, Infinity, -Infinity, Number.MAX_VALUE])('rejects non-finite decoded float samples %s', (value) => {
  const bytes = Buffer.alloc(8);
  bytes.writeDoubleLE(value);
  expect(() =>
    audioDecodeWavFrames(
      bytes,
      { encoding: 3, sampleRate: 24000, channels: 1, bitsPerSample: 64, blockAlign: 8 },
      true,
    ),
  ).toThrow(/finite/);
});
