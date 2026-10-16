/** Native-free resource metadata, also consumed by download/conversion tooling. */
import type { TtsCapabilities } from './types.js';

export const ttsFamilies = {
  qwen3_tts: {
    variants: ['custom_voice', 'base', 'voice_design'],
    components: ['speech_tokenizer'],
    textStreaming: 'segmented',
  },
} as const;

export function ttsComponents(config: { model_type?: unknown }): readonly string[] {
  return config.model_type === 'qwen3_tts' ? ttsFamilies.qwen3_tts.components : [];
}

function record(value: unknown, name: string): Record<string, unknown> {
  if (!value || typeof value !== 'object' || Array.isArray(value)) throw new Error(`Invalid TTS ${name}`);
  return value as Record<string, unknown>;
}

/** Inspect configuration without importing the addon or loading model weights.
 * Native loading remains authoritative; hosts must compare the loaded capabilities. */
export function inspectTtsConfig(config: unknown, codec: unknown): TtsCapabilities {
  const model = record(config, 'configuration');
  const tokenizer = record(codec, 'codec configuration');
  if (model.model_type !== 'qwen3_tts') throw new Error('Unsupported TTS model family');
  const variant = model.tts_model_type;
  if (variant !== 'base' && variant !== 'custom_voice' && variant !== 'voice_design')
    throw new Error(`Unsupported TTS variant: ${String(variant)}`);
  const rate = tokenizer.output_sample_rate;
  if (typeof rate !== 'number' || !Number.isSafeInteger(rate) || rate < 1 || rate > 0xffffffff)
    throw new Error('Invalid TTS sample rate');
  const voice = variant === 'base' ? 'reference' : variant === 'voice_design' ? 'description' : 'preset';
  const instruct = model.tts_model_size === '1b7' && voice !== 'reference' ? 'supported' : 'unsupported';
  // Match the SDK backend's released Qwen3 conditioning profile. Unknown
  // preset sizes fail closed for instructions; unknown design sizes cannot load.
  if (voice === 'description' && instruct === 'unsupported') throw new Error('Unknown VoiceDesign instruction profile');
  const talker = record(model.talker_config, 'talker configuration');
  const ids = (value: unknown, name: string): string[] => {
    const values = record(value ?? {}, name);
    if (Object.values(values).some((id) => typeof id !== 'number' || !Number.isSafeInteger(id) || id < 0))
      throw new Error(`Invalid TTS ${name}`);
    return Object.keys(values).sort();
  };
  return {
    family: 'qwen3_tts',
    variant,
    sampleRate: rate,
    channels: 1,
    voices: ids(talker.spk_id, 'speaker IDs'),
    languages: ids(talker.codec_language_id, 'language IDs'),
    conditioning: [{ voice, instruct }],
    voiceCloning: variant === 'base',
    textStreaming: ttsFamilies.qwen3_tts.textStreaming,
    audioStreaming: true,
  };
}
