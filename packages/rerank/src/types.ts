export interface RerankerPolicy {
  maxBlockTokens: number;
  maxDocumentsPerBlock: number;
  prefillChunkSize: number;
  maxQueryTokens: number;
  maxDocumentTokens: number;
  maxDocuments: number;
  maxInputTokens: number;
}
export interface RerankerInfo {
  family: 'jina_reranker';
  profile: 'jina-v3' | 'jina-v3.5';
  contextWindow: number;
  policy: Readonly<RerankerPolicy>;
  quantized: boolean;
  weightFormats: readonly ('dense' | 'affine-4bit' | 'mxfp4')[];
}
export type RerankDocument = string | { text: string };
export interface RerankRequest {
  query: string;
  documents: readonly RerankDocument[];
  topN?: number;
  returnDocuments?: boolean;
  truncate?: boolean;
  signal?: AbortSignal;
}
export interface RerankResponse {
  results: Array<{ index: number; relevanceScore: number; document?: { text: string } }>;
  usage: { inputTokens: number; promptTokens: number };
  meta: {
    profile: RerankerInfo['profile'];
    blocks: number;
    processedDocuments: number;
    maxBlockTokens: number;
    maxDocumentsPerBlock: number;
    truncatedQuery: boolean;
    truncatedDocumentIndices: number[];
  };
}
export interface RerankerModel {
  readonly info: RerankerInfo;
  rerank(request: RerankRequest): Promise<RerankResponse>;
  dispose(): Promise<void>;
}
export interface LoadRerankerOptions {
  policy?: Partial<RerankerPolicy>;
}
