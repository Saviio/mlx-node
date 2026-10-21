/** Compare separately collected, quiet-machine control runs. */
import assert from 'node:assert/strict';
import { readFile, writeFile } from 'node:fs/promises';

const [baselinePath, currentPath, output] = process.argv.slice(2);
if (!output) throw Error('Usage: node compare-regression.mjs BASELINE CURRENT REPORT');
const read = async (path) => JSON.parse(await readFile(path, 'utf8'));
const [baseline, current, manifest] = await Promise.all([
  read(baselinePath),
  read(currentPath),
  read(new URL('../../../fixtures/rerank/bge/manifest.json', import.meta.url)),
]);
assert.equal(baseline.workload, 'unified-memory-chat8-rerank50-v1');
assert.equal(current.workload, baseline.workload);
assert.equal(current.chatPath, baseline.chatPath);
assert.equal(current.jinaPath, baseline.jinaPath);
assert.deepEqual(current.hardware, baseline.hardware);
for (const report of [baseline, current]) {
  assert.match(report.addonSha256, /^[a-f0-9]{64}$/);
  assert.equal(report.rounds.length, 3);
  for (const round of report.rounds) {
    for (const family of ['chat', 'jina']) {
      const values = round[family].values;
      assert.equal(values.length, manifest.performance.iterations);
      assert.ok(values.every((v) => Number.isFinite(v) && v > 0));
      const sorted = values.toSorted((a, b) => a - b);
      assert.equal(round[family].medianMs, (sorted[4] + sorted[5]) / 2);
    }
  }
}
const threshold = manifest.thresholds.baselineMaxMedianLatencyRegression;
const families = ['chat', 'jina'].map((family) => {
  const medianLatencyChanges = current.rounds.map(
    (r, i) => r[family].medianMs / baseline.rounds[i][family].medianMs - 1,
  );
  return { family, medianLatencyChanges, passed: !medianLatencyChanges.every((v) => v > threshold) };
});
const report = {
  baselinePath,
  currentPath,
  baselineAddonSha256: baseline.addonSha256,
  currentAddonSha256: current.addonSha256,
  threshold,
  families,
  passed: families.every((f) => f.passed),
};
await writeFile(output, JSON.stringify(report, null, 2) + '\n');
console.log(JSON.stringify(report));
if (!report.passed) process.exitCode = 1;
