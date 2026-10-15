import type { ServerResponse } from 'node:http';

/** Small request metadata only. Never retain text, PCM, WAV or SSE payloads. */
export interface AudioRequestMetrics {
  task: 'asr' | 'tts' | 'voice';
  model?: string;
  state: 'running' | 'completed' | 'error' | 'cancelled';
  errorCode?: string;
  uploadMs?: number;
  /** Peak bytes reserved by this request's upload or WAV response file. */
  tempBytes?: number;
  modelLoadMs?: number;
  queueWaitMs?: number;
  /** Opening the uploaded WAV and decoding its samples. */
  wavReadMs?: number;
  computeMs?: number;
  featureMs?: number;
  /** Audio encoder plus decoder prefill, up to the first generated token read. */
  prefillMs?: number;
  /** First generated token read to the last one. */
  decodeMs?: number;
  outputTokens?: number;
  firstPcmMs?: number;
  /** PCM frames per channel sent to the client. */
  pcmFrames?: number;
  audioSeconds?: number;
  realtimeFactor?: number;
  inputWaitMs?: number;
  consumerWaitMs?: number;
  finishReason?: string;
  /** Handler entry to handler exit, or to the response flush when that comes first. */
  totalMs?: number;
}
const requests = new WeakMap<ServerResponse, AudioRequestMetrics>();
export function beginAudioMetrics(res: ServerResponse, task: AudioRequestMetrics['task']): AudioRequestMetrics {
  const metrics: AudioRequestMetrics = { task, state: 'running' };
  requests.set(res, metrics);
  return metrics;
}
export function audioMetrics(res: ServerResponse): AudioRequestMetrics | undefined {
  return requests.get(res);
}
