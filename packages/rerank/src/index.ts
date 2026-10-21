import { JinaRerankerModel } from '@mlx-node/core';

import { rerankError, validateRerankerPolicy } from './protocol.js';
import { wrapReranker } from './runtime.js';
import type { LoadRerankerOptions, RerankerModel } from './types.js';
export type * from './types.js';
export { RerankError } from './protocol.js';
export async function loadReranker(modelPath: string, options: LoadRerankerOptions = {}): Promise<RerankerModel> {
  const policy = validateRerankerPolicy(options.policy);
  try {
    return wrapReranker(await JinaRerankerModel.load(modelPath, JSON.stringify(policy)));
  } catch (error) {
    throw rerankError(error);
  }
}
