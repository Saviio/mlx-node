/** Frozen chat/Jina control and chat with idle/concurrent BGE, through the real host. */
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { mkdtemp, readFile, rm, symlink, writeFile } from 'node:fs/promises';
import { cpus, totalmem, platform, release } from 'node:os';
import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';

import { getMemorySnapshot, resetPeakMemory } from '@mlx-node/core';

import { createInferenceHost } from '../../../packages/server/src/host/index.js';

const [chatPath, jinaPath, output, bgePath] = process.argv.slice(2);
if (!output) throw Error('Usage: oxnode regression.ts CHAT JINA REPORT [BGE]');
const manifest = JSON.parse(
  await readFile(new URL('../../../fixtures/rerank/bge/manifest.json', import.meta.url), 'utf8'),
);
const root = await mkdtemp(join(tmpdir(), 'mlx-bge-regression-'));
for (const [name, path] of [['chat', chatPath], ['jina', jinaPath], ...(bgePath ? [['bge', bgePath]] : [])]) {
  await symlink(resolve(path), join(root, name));
}
const memory = () =>
  Object.fromEntries(Object.entries(getMemorySnapshot()).map(([k, v]) => [k, typeof v === 'bigint' ? Number(v) : v]));
const host = await createInferenceHost({
  modelsDir: root,
  model: 'chat',
  port: 0,
  disableStore: true,
  sweepOrphanTempRoots: false,
  logDir: join(root, 'logs'),
});
const report: Record<string, unknown> = {
  recordedAt: new Date().toISOString(),
  addon: process.env.NAPI_RS_NATIVE_LIBRARY_PATH ?? 'workspace',
  addonSha256: createHash('sha256')
    .update(
      await readFile(
        process.env.NAPI_RS_NATIVE_LIBRARY_PATH ??
          new URL('../../../packages/core/mlx-core.darwin-arm64.node', import.meta.url),
      ),
    )
    .digest('hex'),
  hardware: { cpu: cpus()[0]?.model, totalMemory: totalmem(), platform: platform(), release: release() },
  workload: 'unified-memory-chat8-rerank50-v1',
  chatPath,
  jinaPath,
  bgePath,
  rounds: [],
};
async function post(path: string, body: unknown) {
  const at = performance.now();
  const response = await fetch(host.url + path, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  const value = await response.json();
  assert.equal(response.status, 200, JSON.stringify(value));
  return { ms: performance.now() - at, value };
}
const chat = () =>
  post('/v1/messages', {
    model: 'chat',
    messages: [{ role: 'user', content: 'Write one sentence explaining unified memory. /no_think' }],
    temperature: 0,
    max_tokens: 8,
  });
const rank = (model: string) =>
  post('/v1/rerank', {
    model,
    query: 'What is unified memory?',
    documents: Array.from(
      { length: 50 },
      (_, i) => `Passage ${i}. ` + 'Apple Silicon uses unified memory for local machine learning inference. '.repeat(4),
    ),
  });
async function sample(run: () => Promise<{ ms: number; value: unknown }>) {
  const values: number[] = [];
  resetPeakMemory();
  for (let i = 0; i < manifest.performance.warmup + manifest.performance.iterations; i++) {
    const result = await run();
    if (i >= manifest.performance.warmup) values.push(result.ms);
  }
  const sorted = [...values].sort((a, b) => a - b);
  return { values, medianMs: (sorted[4] + sorted[5]) / 2, p95Ms: sorted[9], memory: memory() };
}
try {
  const rounds = [];
  for (let i = 0; i < 3; i++) {
    const row = { chat: await sample(chat), jina: await sample(() => rank('jina')) };
    rounds.push(row);
    report.rounds = rounds;
    await writeFile(output, JSON.stringify(report, null, 2) + '\n');
    console.log(JSON.stringify({ round: i, ...row }));
  }
  if (bgePath) {
    const at = performance.now();
    await host.loadModel('bge');
    report.bgeLoadMs = performance.now() - at;
    assert.ok(host.server.registry.getAny('chat'));
    assert.ok(host.server.registry.rerank.get('bge'));
    assert.equal(host.server.registry.rerank.get('jina'), undefined);
    report.chatWithIdleBge = await sample(chat);
    report.bgeAlone = await sample(() => rank('bge'));
    const contention: number[] = [];
    report.chatWithConcurrentBge = await sample(async () => {
      const [c, b] = await Promise.all([chat(), rank('bge')]);
      contention.push(b.ms);
      return c;
    });
    report.concurrentBgeMs = contention.slice(manifest.performance.warmup);
  }
} finally {
  await host.close();
  report.afterClose = memory();
  await rm(root, { recursive: true, force: true });
}
await writeFile(output, JSON.stringify(report, null, 2) + '\n');
