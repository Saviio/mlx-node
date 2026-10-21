import { mkdtemp, mkdir, writeFile, rm, readFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

import { expect, it, vi } from 'vite-plus/test';

import type { RerankerModel } from '../../../packages/rerank/src/types.js';
import { createInferenceHost } from '../../../packages/server/src/host/index.js';
it('starts with only rerank models and never logs input or returned documents', async () => {
  const root = await mkdtemp(join(tmpdir(), 'rerank-host-'));
  const path = join(root, 'rank');
  await mkdir(path);
  for (const [file, content] of Object.entries({
    'config.json': JSON.stringify({ model_type: 'qwen3', architectures: ['JinaForRanking'] }),
    'tokenizer.json': '{}',
    'tokenizer_config.json': '{}',
    'model.safetensors': 'fixture',
  }))
    await writeFile(join(path, file), content);
  const dispose = vi.fn(async () => {});
  const load = vi.fn(
    async () =>
      ({
        info: { family: 'jina_reranker', profile: 'jina-v3', quantized: false, policy: {} },
        dispose,
        rerank: async () => ({
          results: [{ index: 0, relevanceScore: 0.9, document: { text: 'SECRET_DOCUMENT' } }],
          usage: { inputTokens: 5, promptTokens: 25 },
          meta: {
            profile: 'jina-v3',
            blocks: 1,
            processedDocuments: 1,
            maxBlockTokens: 32768,
            maxDocumentsPerBlock: 125,
            truncatedQuery: false,
            truncatedDocumentIndices: [],
          },
        }),
      }) as unknown as RerankerModel,
  );
  const logDir = join(root, 'logs');
  const host = await createInferenceHost({
    modelsDir: root,
    port: 0,
    logDir,
    loadRerankModel: load,
    disableStore: true,
  });
  try {
    expect(host.models).toEqual([]);
    expect(host.rerankModels.map((m) => m.name)).toEqual(['rank']);
    expect(host.boundModel).toBe('rank');
    const result = await fetch(host.url + '/v1/rerank', {
      method: 'POST',
      body: JSON.stringify({
        model: 'rank',
        query: 'SECRET_QUERY',
        documents: ['SECRET_DOCUMENT'],
        return_documents: true,
      }),
    });
    expect(result.status).toBe(200);
    await result.text();
    expect(load).toHaveBeenCalledTimes(1);
    expect(
      (
        await fetch(host.url + '/v1/rerank', {
          method: 'POST',
          body: JSON.stringify({ model: 'unknown', query: 'q', documents: [] }),
        })
      ).status,
    ).toBe(404);
  } finally {
    await host.close();
  }
  try {
    expect(dispose).toHaveBeenCalledTimes(1);
    const { readdir } = await import('node:fs/promises');
    const logs = (await Promise.all((await readdir(logDir)).map((f) => readFile(join(logDir, f), 'utf8')))).join('\n');
    expect(logs).not.toContain('SECRET_QUERY');
    expect(logs).not.toContain('SECRET_DOCUMENT');
    expect(logs).toContain('rerankMetrics');
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});
