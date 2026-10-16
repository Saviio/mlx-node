import { expect, it, vi } from 'vite-plus/test';

import { run } from '../../packages/cli/src/commands/tts.js';

const calls = vi.hoisted(() => [] as Record<string, unknown>[]);
vi.mock('@mlx-node/tts', () => ({
  async loadTtsModel() {
    return {
      capabilities: { family: 'test', variant: 'test', sampleRate: 24000, channels: 1, voices: ['ryan'] },
      synthesizeStream(_input: unknown, options: Record<string, unknown>) {
        calls.push(options);
        return Object.assign((async function* () {})(), { completed: Promise.resolve({ synthesisMs: 0 }) });
      },
      async dispose() {},
    };
  },
  changeAudioSpeed: (source: AsyncIterable<unknown>) =>
    Object.assign(source, { stats: { outputFrames: 0, processingMs: 0 } }),
  WavWriter: { open: async () => ({ write: async () => {}, close: async () => {} }) },
  createAudioPlayer: async () => ({ write: async () => {}, finish: async () => ({}), cancel: () => {} }),
  readWav: async () => {
    throw new Error('unused');
  },
}));

it('keeps the bootstrap frame whenever a file is written and offers the drop to playback only', async () => {
  const error = vi.spyOn(console, 'error').mockImplementation(() => {});
  const args = ['-m', '/model', '--voice', 'ryan', '--language', 'english', '--text', 'Hi.'];
  try {
    await run([...args, '-o', '/unused.wav']);
    await run([...args, '--play']);
    await run([...args, '--play', '-o', '/unused.wav']);
  } finally {
    error.mockRestore();
  }
  expect(calls.map((options) => options.suppressBootstrapSilence)).toEqual([false, true, false]);
});
