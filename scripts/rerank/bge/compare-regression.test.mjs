import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import { mkdtemp, readFile, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { test } from 'node:test';
import { fileURLToPath } from 'node:url';

const manifest = JSON.parse(await readFile(new URL('../../../fixtures/rerank/bge/manifest.json', import.meta.url)));
const digest = 'a'.repeat(64);
function fixture() {
  return {
    workload: 'unified-memory-chat8-rerank50-v1',
    chatPath: '/chat',
    jinaPath: '/jina',
    hardware: { cpu: 'test' },
    addonSha256: digest,
    identity: {
      node: 'v24',
      execArgv: [],
      sourceRevision: manifest.baselineRevision,
      ...Object.fromEntries(
        ['executionEnvironment', 'script', 'manifest', 'chat', 'jina', 'source', 'lock', 'metal'].map((k) => [
          k + 'Sha256',
          digest,
        ]),
      ),
    },
    afterClose: { activeBytes: 1024 },
    rounds: Array.from({ length: 3 }, () =>
      Object.fromEntries(
        ['chat', 'jina'].map((family) => [
          family,
          {
            values: Array(10).fill(100),
            medianMs: 100,
            outputTokens: family === 'chat' ? Array(10).fill(8) : [],
          },
        ]),
      ),
    ),
  };
}
async function compare(change) {
  const root = await mkdtemp(join(tmpdir(), 'bge-regression-contract-'));
  try {
    const baseline = fixture();
    const current = fixture();
    change(current);
    await writeFile(join(root, 'baseline.json'), JSON.stringify(baseline));
    await writeFile(join(root, 'current.json'), JSON.stringify(current));
    return spawnSync(
      process.execPath,
      [
        fileURLToPath(new URL('./compare-regression.mjs', import.meta.url)),
        join(root, 'baseline.json'),
        join(root, 'current.json'),
        join(root, 'report.json'),
      ],
      { encoding: 'utf8' },
    );
  } finally {
    await rm(root, { recursive: true, force: true });
  }
}
test('accepts unchanged workloads and active-memory baseline', async () => {
  const result = await compare(() => {});
  assert.equal(result.status, 0, result.stderr);
});
test('blocks three consecutive latency regressions without widening the threshold', async () => {
  const result = await compare((r) => {
    for (const round of r.rounds) {
      round.jina.values.fill(106);
      round.jina.medianMs = 106;
    }
  });
  assert.equal(result.status, 1);
  assert.equal(JSON.parse(result.stdout).passed, false);
});
for (const [name, change] of [
  [
    'active resources retained after close',
    (r) => {
      r.afterClose.activeBytes++;
    },
  ],
  [
    'missing close evidence',
    (r) => {
      delete r.afterClose;
    },
  ],
  [
    'different output workload',
    (r) => {
      r.rounds[0].chat.outputTokens[0]++;
    },
  ],
  [
    'missing identity',
    (r) => {
      delete r.identity.scriptSha256;
    },
  ],
])
  test(`rejects ${name}`, async () => assert.notEqual((await compare(change)).status, 0));
