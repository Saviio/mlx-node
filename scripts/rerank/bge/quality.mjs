import { createHash } from 'node:crypto';
import { createReadStream } from 'node:fs';
import { readFile, writeFile, appendFile, readdir } from 'node:fs/promises';
import { join, resolve } from 'node:path';

import { XlmRobertaRerankerModel } from '../../../packages/core/index.cjs';
const [modelPath, inputPath, outputPath, baselinePath] = process.argv.slice(2);
if (!outputPath) throw Error('Usage: node quality.mjs MODEL INPUT OUTPUT [BASELINE]');
const manifest = JSON.parse(
  await readFile(new URL('../../../fixtures/rerank/bge/manifest.json', import.meta.url), 'utf8'),
);
const hashFile = async (path) => {
  const hash = createHash('sha256');
  for await (const chunk of createReadStream(path)) hash.update(chunk);
  return hash.digest('hex');
};
const input = JSON.parse(await readFile(inputPath, 'utf8'));
const baseline = baselinePath ? JSON.parse(await readFile(baselinePath, 'utf8')) : undefined;
const checksums = {};
for (const name of (await readdir(modelPath)).sort()) {
  if (name.endsWith('.json') || name.endsWith('.safetensors')) checksums[name] = await hashFile(join(modelPath, name));
}
const identity = {
  model: resolve(modelPath),
  checksums,
  inputSha256: await hashFile(inputPath),
  addonSha256: await hashFile(
    process.env.NAPI_RS_NATIVE_LIBRARY_PATH ??
      new URL('../../../packages/core/mlx-core.darwin-arm64.node', import.meta.url),
  ),
};
if (baseline && baseline.inputSha256 !== identity.inputSha256)
  throw Error('Quality baseline input identity differs (queries, candidates, labels, or dataset revision)');
const progressPath = outputPath + '.progress.jsonl';
let completed = [];
try {
  const [savedIdentity, ...rows] = (await readFile(progressPath, 'utf8')).trimEnd().split('\n').map(JSON.parse);
  if (JSON.stringify(savedIdentity) !== JSON.stringify(identity))
    throw Error('Progress identity differs; use a new output path');
  completed = rows;
} catch (error) {
  if (error.code !== 'ENOENT') throw error;
  await writeFile(progressPath, JSON.stringify(identity) + '\n');
}
const model = await XlmRobertaRerankerModel.load(modelPath);
async function rerank(request) {
  const task = model.startRerank(JSON.stringify(request));
  try {
    return JSON.parse(await task.result());
  } finally {
    task.cancel();
  } // Retain task ownership until completion, including GC during long runs.
}
const report = { ...identity, info: JSON.parse(model.infoJson), datasets: [] };
const dcg = (labels) => labels.reduce((sum, rel, i) => sum + rel / Math.log2(i + 2), 0);
try {
  for (const dataset of input.datasets) {
    const rows = [];
    for (const query of dataset.queries) {
      const saved = completed.find((r) => r.repository === dataset.repository && r.sha256 === query.sha256);
      if (saved) {
        const { repository: _repository, ...row } = saved;
        rows.push(row);
        continue;
      }
      const started = performance.now();
      const r = await rerank({ query: query.query, documents: query.documents, truncate: true });
      const ranking = r.results.map((v) => v.index);
      const ideal = dcg(query.labels.toSorted((a, b) => b - a).slice(0, 10));
      const ndcg10 = dcg(ranking.slice(0, 10).map((i) => query.labels[i])) / ideal;
      const first = ranking.slice(0, 10).findIndex((i) => query.labels[i] > 0);
      const mrr10 = first < 0 ? 0 : 1 / (first + 1);
      const row = {
        sha256: query.sha256,
        ranking,
        ndcg10,
        mrr10,
        milliseconds: performance.now() - started,
        usage: r.usage,
        meta: r.meta,
      };
      rows.push(row);
      await appendFile(progressPath, JSON.stringify({ repository: dataset.repository, ...row }) + '\n');
      if (rows.length % 32 === 0) console.log(JSON.stringify({ dataset: dataset.repository, completed: rows.length }));
    }
    const result = {
      repository: dataset.repository,
      revision: dataset.revision,
      fileSha256: dataset.fileSha256,
      skipped: dataset.skipped,
      queries: rows,
      ndcg10: rows.reduce((n, r) => n + r.ndcg10, 0) / rows.length,
      mrr10: rows.reduce((n, r) => n + r.mrr10, 0) / rows.length,
    };
    if (baseline) {
      const ref = baseline.datasets.find((d) => d.repository === dataset.repository);
      if (!ref || ref.queries.length !== rows.length || ref.queries.some((r, i) => r.sha256 !== rows[i].sha256))
        throw Error('Quality subset mismatch');
      result.ndcg10Drop = ref.ndcg10 - result.ndcg10;
      result.mrr10Change = result.mrr10 - ref.mrr10;
      result.changedRankings = rows.filter(
        (r, i) => JSON.stringify(r.ranking) !== JSON.stringify(ref.queries[i].ranking),
      ).length;
      result.passed = result.ndcg10Drop <= manifest.thresholds.quantizationNdcg10MaxDrop;
    }
    report.datasets.push(result);
    await writeFile(outputPath, JSON.stringify(report, null, 2) + '\n');
    console.log(
      JSON.stringify({
        repository: result.repository,
        ndcg10: result.ndcg10,
        mrr10: result.mrr10,
        ndcg10Drop: result.ndcg10Drop,
        passed: result.passed,
      }),
    );
  }
} finally {
  await model.dispose();
}
if (report.datasets.some((d) => d.passed === false)) process.exitCode = 1;
