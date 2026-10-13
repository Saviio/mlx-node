import { expect, it } from 'vite-plus/test';

import { inspectTtsConfig } from '../src/catalog.js';

it.each([
  ['custom_voice', '0b6', 'preset', 'unsupported'],
  ['custom_voice', '1b7', 'preset', 'supported'],
  ['custom_voice', 'unknown', 'preset', 'unsupported'],
  ['base', '1b7', 'reference', 'unsupported'],
  ['voice_design', '1b7', 'description', 'supported'],
])('inspects the released %s/%s profile without weights', (variant, size, voice, instruct) => {
  expect(
    inspectTtsConfig(
      {
        model_type: 'qwen3_tts',
        tts_model_type: variant,
        tts_model_size: size,
        talker_config: { spk_id: { z: 2, a: 1 }, codec_language_id: { english: 3 } },
      },
      { output_sample_rate: 44100 },
    ),
  ).toMatchObject({
    variant,
    sampleRate: 44100,
    channels: 1,
    voices: ['a', 'z'],
    languages: ['english'],
    conditioning: [{ voice, instruct }],
    voiceCloning: variant === 'base',
  });
});

it.each([undefined, '0b6', 'unknown'])('rejects an unsupported VoiceDesign size: %s', (size) => {
  expect(() =>
    inspectTtsConfig(
      {
        model_type: 'qwen3_tts',
        tts_model_type: 'voice_design',
        tts_model_size: size,
        talker_config: {},
      },
      { output_sample_rate: 24000 },
    ),
  ).toThrow('Unknown VoiceDesign instruction profile');
});
