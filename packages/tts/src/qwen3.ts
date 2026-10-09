import type { TtsBackend } from './model.js';
import type { TtsCapabilities, TtsLoadOptions } from './types.js';

/** Codec frame scheduling belongs to this adapter, outside generic text sessions. */
export async function loadQwen3Backend(path: string, options: TtsLoadOptions = {}): Promise<TtsBackend> {
  for (const [name, value] of Object.entries({
    maxBytes: options.instructionCache?.maxBytes,
    maxEntries: options.instructionCache?.maxEntries,
  }))
    if (value !== undefined && (!Number.isSafeInteger(value) || value < 0))
      throw new RangeError(`Invalid instruction cache ${name}`);
  if (options.instructionCache?.enabled !== undefined && typeof options.instructionCache.enabled !== 'boolean')
    throw new TypeError('Instruction cache enabled must be boolean');
  const { TtsNativeModel } = await import('@mlx-node/core');
  const native = await TtsNativeModel.load(
    path,
    JSON.stringify({
      instruction_cache: {
        enabled: options.instructionCache?.enabled,
        max_bytes: options.instructionCache?.maxBytes,
        max_entries: options.instructionCache?.maxEntries,
      },
    }),
  );
  const { samplesPerFrame, ...capabilities } = JSON.parse(native.metadata) as TtsCapabilities & {
    samplesPerFrame: number;
  };
  return {
    capabilities,
    start(text, options) {
      const frameMs = (samplesPerFrame * 1000) / capabilities.sampleRate;
      const chunkFrames = Math.ceil(options.chunkDurationMs / frameMs);
      const bufferChunks = Math.max(1, Math.floor((options.audioBufferSeconds * 1000) / (chunkFrames * frameMs)));
      if (bufferChunks > 4096)
        throw new RangeError('audioBufferSeconds/chunkDurationMs exceeds the native buffer capacity (4096 chunks)');
      return native.start(
        text,
        JSON.stringify({
          voice: options.voice,
          voice_description: options.voiceDescription,
          instruct: options.instruct,
          prepared_voice_id: options.preparedVoiceId,
          language: options.language,
          temperature: options.temperature,
          top_k: options.topK,
          top_p: options.topP,
          repetition_penalty: options.repetitionPenalty,
          seed: options.seed,
          chunk_frames: chunkFrames,
          buffer_chunks: bufferChunks,
          max_frames:
            options.maxDurationSeconds === undefined
              ? undefined
              : Math.ceil((options.maxDurationSeconds * 1000) / frameMs),
        }),
      );
    },
    prepareVoice: (audio, sampleRate, transcript) => native.prepareVoice(audio, sampleRate, transcript),
    releaseVoice: (id) => native.releaseVoice(id),
    dispose: () => native.dispose(),
  };
}
