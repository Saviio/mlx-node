import { normalizeRequest, RerankError, rerankError } from './protocol.js';
import type { RerankerInfo, RerankerModel, RerankRequest, RerankResponse } from './types.js';
export interface NativeReranker {
  readonly infoJson: string;
  startRerank(request: string): { cancel(): void; result(): Promise<string> };
  dispose(): Promise<void>;
}
/** Abort is a request to the worker; completion remains the lifetime boundary. */
export function wrapReranker(native: NativeReranker): RerankerModel {
  const info = JSON.parse(native.infoJson) as RerankerInfo;
  Object.freeze(info.policy);
  Object.freeze(info.weightFormats);
  Object.freeze(info);
  let disposed = false;
  let disposal: Promise<void> | undefined;
  return {
    info,
    async rerank(request: RerankRequest): Promise<RerankResponse> {
      if (disposed) throw new RerankError('closed', 'Reranker is disposed', 503);
      const input = normalizeRequest(request);
      request.signal?.throwIfAborted();
      let task: ReturnType<NativeReranker['startRerank']>;
      try {
        task = native.startRerank(JSON.stringify(input));
      } catch (error) {
        throw rerankError(error);
      }
      const abort = () => task.cancel();
      request.signal?.addEventListener('abort', abort, { once: true });
      if (request.signal?.aborted) abort();
      try {
        const result = JSON.parse(await task.result()) as RerankResponse;
        request.signal?.throwIfAborted();
        result.meta.maxDocumentsPerBlock = info.policy.maxDocumentsPerBlock;
        if (request.returnDocuments)
          for (const item of result.results) item.document = { text: input.documents[item.index] };
        return result;
      } catch (error) {
        request.signal?.throwIfAborted();
        throw rerankError(error);
      } finally {
        request.signal?.removeEventListener('abort', abort);
      }
    },
    dispose() {
      disposed = true;
      return (disposal ??= native.dispose());
    },
  };
}
