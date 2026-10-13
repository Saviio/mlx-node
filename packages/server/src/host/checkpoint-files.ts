import { readdir, readFile, stat } from 'node:fs/promises';
import { join } from 'node:path';

/** Cheap installation checks only; tensor/tokenizer contents are validated on load. */
async function fileState(path: string): Promise<boolean | undefined> {
  try {
    const info = await stat(path);
    return info.isFile() && info.size > 0;
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code === 'ENOENT') return undefined;
    throw error;
  }
}

async function hasFile(path: string): Promise<boolean> {
  return (await fileState(path)) === true;
}

async function hasWeights(path: string, requireAsrIndex = false): Promise<boolean> {
  // These are the single-file names and shard layouts read by the native
  // persistence loader. ASR additionally requires model.safetensors or its index.
  const weights = await fileState(join(path, 'weights.safetensors'));
  const model = await fileState(join(path, 'model.safetensors'));
  const index = await hasFile(join(path, 'model.safetensors.index.json'));
  if (requireAsrIndex && !model && !index) return false;
  // Existence selects the native path: an empty/invalid preferred file must
  // not fall through to another single file or an otherwise complete shard set.
  if (weights !== undefined) return weights;
  if (model !== undefined) return model;
  const files = await readdir(path);
  const shards = files.filter((name) => /^(?:model-|model\.safetensors-).+-of-.+\.safetensors$/.test(name));
  if (!shards.length) return false;
  const groups = new Map<string, { total: number; indices: Set<number> }>();
  for (const name of shards) {
    const match = /^(model(?:\.safetensors)?)-(\d+)-of-(\d+)\.safetensors$/.exec(name);
    if (!match || !(await hasFile(join(path, name)))) return false;
    const n = Number(match[2]);
    const total = Number(match[3]);
    if (!Number.isSafeInteger(total) || n < 1 || n > total || total > shards.length) return false;
    const group = groups.get(match[1]) ?? { total, indices: new Set<number>() };
    if (group.total !== total || group.indices.has(n)) return false;
    group.indices.add(n);
    groups.set(match[1], group);
  }
  if ([...groups.values()].some((g) => g.indices.size !== g.total)) return false;
  if (index) {
    const manifest: unknown = JSON.parse(await readFile(join(path, 'model.safetensors.index.json'), 'utf8'));
    const map = manifest && typeof manifest === 'object' && 'weight_map' in manifest ? manifest.weight_map : undefined;
    if (!map || typeof map !== 'object' || Array.isArray(map)) return false;
    const names = Object.values(map);
    if (!names.length || names.some((name) => typeof name !== 'string' || !shards.includes(name))) return false;
  }
  return true;
}

export async function hasAudioCheckpointFiles(path: string, kind: 'asr' | 'tts'): Promise<boolean> {
  if (!(await hasWeights(path, kind === 'asr'))) return false;
  const tokenizer = await hasFile(join(path, 'tokenizer.json'));
  if (kind === 'asr') return tokenizer;
  if (!tokenizer) {
    for (const name of ['vocab.json', 'merges.txt', 'tokenizer_config.json'])
      if (!(await hasFile(join(path, name)))) return false;
  }
  return (await hasFile(join(path, 'generation_config.json'))) && (await hasWeights(join(path, 'speech_tokenizer')));
}
