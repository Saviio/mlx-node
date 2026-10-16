import { readFile } from 'node:fs/promises';
import { join } from 'node:path';

import { createTtsModel } from './model.js';
export { NON_STREAMING_CHUNK_DURATION_MS } from './model.js';
import { loadQwen3Backend } from './qwen3.js';
import type { TtsModel, TtsLoadOptions } from './types.js';
export type * from './types.js';
export { InvalidReferenceAudioError } from './errors.js';

export async function loadTtsModel(path: string, options: TtsLoadOptions = {}): Promise<TtsModel> {
  const config = JSON.parse(await readFile(join(path, 'config.json'), 'utf8')) as { model_type?: string };
  if (config.model_type !== 'qwen3_tts') throw new Error(`Unsupported TTS model family: ${config.model_type}`);
  return createTtsModel(await loadQwen3Backend(path, options));
}
export { readWav, WavReader, WavWriter, InvalidWavError, createAudioPlayer } from './audio.js';
export type { AudioPlayer, PlaybackStats } from './audio.js';
export { changeAudioSpeed } from './speed.js';
export type { AudioSpeedStream, AudioProcessingStats } from './speed.js';
