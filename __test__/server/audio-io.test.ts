import { mkdir, mkdtemp, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { dirname, join } from 'node:path';

import { expect, it, vi } from 'vite-plus/test';

import { AudioTempStore } from '../../packages/server/src/audio-io.js';

it.each(['retry', 'close'] as const)('can %s after shared temporary directory creation fails', async (operation) => {
  const root = await mkdtemp(join(tmpdir(), 'audio-temp-recovery-'));
  const directory = join(root, 'temporarily-unavailable');
  const store = new AudioTempStore();
  vi.stubEnv('TMPDIR', directory);
  try {
    const failed = await Promise.allSettled([store.create(), store.create()]);
    for (const result of failed) expect(result).toMatchObject({ status: 'rejected', reason: { code: 'ENOENT' } });
    if (operation === 'close') {
      await expect(store.close()).resolves.toBeUndefined();
      return;
    }

    await mkdir(directory);
    const files = await Promise.all([store.create(), store.create()]);
    try {
      expect(files[0].path).not.toBe(files[1].path);
      expect(dirname(files[0].path)).toBe(dirname(files[1].path));
      await Promise.all(files.map((file) => file.write(Buffer.from('recovered'))));
    } finally {
      await Promise.all(files.map((file) => file.dispose()));
    }
  } finally {
    vi.unstubAllEnvs();
    await store.close().catch(() => {});
    await rm(root, { recursive: true, force: true });
  }
});
