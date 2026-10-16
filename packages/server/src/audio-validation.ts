import type { Qwen3AsrStreamOptions, Qwen3AsrTranscribeOptions } from '@mlx-node/core';
import type { TtsInputEvent, TtsModel, TtsOptions } from '@mlx-node/tts';

import { AudioError, bad, object, string } from './audio-io.js';

function numeric(value: unknown, key: string, min: number, max: number, integer = false): number {
  if (
    typeof value !== 'number' ||
    !Number.isFinite(value) ||
    value < min ||
    value > max ||
    (integer && !Number.isSafeInteger(value))
  )
    bad(`${key} must be ${integer ? 'an integer' : 'a number'} in ${min}..${max}`, key);
  return value as number;
}
export function textLength(text: string): number {
  let count = 0;
  for (const _codePoint of text) count++;
  return count;
}
export interface SpeechRequest {
  model: string;
  input?: string;
  format: 'wav' | 'pcm';
  sse: boolean;
  speed: number;
  options: Omit<TtsOptions, 'voice'> & {
    voice: string | { type: 'description'; description: string } | { id: string };
  };
}
export function speechRequest(body: Record<string, unknown>, inputStream: boolean): SpeechRequest {
  const model = string(body.model, 'model')!;
  const format = body.response_format ?? (inputStream ? 'pcm' : 'wav');
  if (format !== 'wav' && format !== 'pcm')
    throw new AudioError(400, 'unsupported_response_format', 'Only wav and pcm are supported', 'response_format');
  const streamFormat = body.stream_format ?? 'audio';
  if (streamFormat !== 'audio' && streamFormat !== 'sse') bad('stream_format must be audio or sse', 'stream_format');
  if ((streamFormat === 'sse' && format !== 'pcm') || (inputStream && (format !== 'pcm' || streamFormat !== 'audio')))
    bad('This streaming mode requires PCM audio', 'response_format');
  let input: string | undefined;
  if (!inputStream) {
    input = string(body.input, 'input')!;
    if (textLength(input) > 4096) bad('input exceeds 4096 characters', 'input');
  } else if (body.input !== undefined) bad('Send text events instead of input', 'input');
  const voice = typeof body.voice === 'string' ? string(body.voice, 'voice')! : object(body.voice, 'voice');
  let options: SpeechRequest['options'];
  if (typeof voice === 'string') options = { voice };
  else if ('id' in voice) {
    if (Object.keys(voice).some((key) => key !== 'id')) bad('Voice references accept only id', 'voice');
    options = { voice: { id: string(voice.id, 'voice.id')! } };
  } else {
    if (voice.type !== 'description') bad('Expected a preset voice or a description object', 'voice');
    const description = string(voice.description, 'voice.description')!;
    if (textLength(description) > 4096) bad('Voice description exceeds 4096 characters', 'voice');
    options = { voice: { type: 'description', description } };
  }
  if (body.instructions !== undefined) {
    if (typeof body.instructions !== 'string' || textLength(body.instructions) > 4096)
      bad('Invalid instructions', 'instructions');
    options.instruct = body.instructions as string;
  }
  if (body.language !== undefined) options.language = string(body.language, 'language');
  if (body.suppress_bootstrap_silence !== undefined) {
    if (typeof body.suppress_bootstrap_silence !== 'boolean')
      bad('suppress_bootstrap_silence must be a boolean', 'suppress_bootstrap_silence');
    options.suppressBootstrapSilence = body.suppress_bootstrap_silence as boolean;
  }
  const sampling = body.sampling === undefined ? {} : object(body.sampling, 'sampling');
  const stream = body.stream === undefined ? {} : object(body.stream, 'stream');
  const specs: [Record<string, unknown>, string, keyof TtsOptions, number, number, boolean?][] = [
    [sampling, 'seed', 'seed', 0, 0xffffffff, true],
    [sampling, 'temperature', 'temperature', 0, 100],
    [sampling, 'top_k', 'topK', 0, 0x7fffffff, true],
    [sampling, 'top_p', 'topP', 0, 1],
    [sampling, 'repetition_penalty', 'repetitionPenalty', 0.001, 100],
    [sampling, 'max_duration_seconds', 'maxDurationSeconds', 0.1, 163.84],
    [stream, 'chunk_duration_ms', 'chunkDurationMs', 1, 10000],
    [stream, 'audio_buffer_seconds', 'audioBufferSeconds', 0.01, 10],
    [stream, 'max_segment_graphemes', 'maxSegmentGraphemes', 1, 4096, true],
    [stream, 'queued_segments', 'queuedSegments', 1, 16, true],
  ];
  for (const [source, key, target, min, max, integer] of specs)
    if (source[key] !== undefined) Object.assign(options, { [target]: numeric(source[key], key, min, max, integer) });
  for (const [source, label] of [
    [sampling, 'sampling'],
    [stream, 'stream'],
  ] as const)
    for (const key of Object.keys(source))
      if (!specs.some((s) => s[0] === source && s[1] === key)) bad(`Unsupported ${label}.${key}`, `${label}.${key}`);
  return {
    model,
    input,
    format,
    sse: streamFormat === 'sse',
    speed: body.speed === undefined ? 1 : numeric(body.speed, 'speed', 0.25, 4),
    options,
  };
}

export function validateVoice(model: TtsModel, options: TtsOptions): void {
  const caps = model.capabilities;
  const mode = typeof options.voice === 'string' ? 'preset' : 'type' in options.voice ? 'description' : 'reference';
  const condition = caps.conditioning.find((c) => c.voice === mode);
  if (!condition) bad('Voice condition is not supported by this model', 'voice');
  if (typeof options.voice === 'string' && !caps.voices.includes(options.voice.toLowerCase()))
    bad('Unknown preset voice', 'voice');
  if (options.instruct?.trim() && condition!.instruct === 'unsupported')
    bad('Instructions are unsupported by this checkpoint', 'instructions');
  if (
    options.language &&
    options.language.toLowerCase() !== 'auto' &&
    !caps.languages.includes(options.language.toLowerCase())
  )
    bad('Unsupported language', 'language');
}

export async function* textEvents(
  source: AsyncIterable<unknown>,
  model: TtsModel,
  options: TtsOptions,
): AsyncGenerator<TtsInputEvent> {
  let count = 0;
  let trailingHighSurrogate = false;
  let nonempty = false;
  for await (const value of source) {
    const record = typeof value === 'string' ? { type: 'text', text: value } : object(value, 'event');
    if (record.type === 'text') {
      if (typeof record.text !== 'string') bad('text event requires a string');
      const text = record.text as string;
      if (text.length) {
        count += textLength(text) - (trailingHighSurrogate && /^[\uDC00-\uDFFF]/u.test(text) ? 1 : 0);
        trailingHighSurrogate = /[\uD800-\uDBFF]$/u.test(text);
      }
      if (count > 4096) bad('Cumulative text exceeds 4096 characters', 'input');
      nonempty ||= !!(record.text as string).trim();
      yield { type: 'text', text: record.text as string };
    } else if (record.type === 'flush') yield { type: 'flush' };
    else if (record.type === 'instruct') {
      if (record.value !== null && (typeof record.value !== 'string' || textLength(record.value) > 4096))
        bad('Invalid instruct event');
      const value = record.value as string | null;
      validateVoice(model, { ...options, instruct: value ?? undefined });
      yield { type: 'instruct', value };
    } else bad('Expected a text, flush, or instruct event');
  }
  if (!nonempty) bad('At least one nonempty text event is required', 'input');
}

function extension(value: unknown, key: string): Record<string, unknown> {
  if (value === undefined) return {};
  if (typeof value === 'string') {
    try {
      value = JSON.parse(value);
    } catch {
      bad(`Invalid ${key} JSON`, key);
    }
  }
  return object(value, key);
}
export function transcriptionRequest(body: Record<string, unknown>): {
  model: string;
  stream: boolean;
  format: 'json' | 'text';
  options: Qwen3AsrTranscribeOptions & Qwen3AsrStreamOptions;
} {
  const model = string(body.model, 'model')!;
  const streaming = body.stream ?? false;
  if (![true, false, 'true', 'false'].includes(streaming as boolean)) bad('stream must be true or false', 'stream');
  const stream = streaming === true || streaming === 'true';
  const format = body.response_format ?? 'json';
  if (format !== 'json' && format !== 'text')
    throw new AudioError(
      400,
      'unsupported_response_format',
      'Only json and text transcription formats are supported',
      'response_format',
    );
  if (stream && format !== 'json') bad('Streaming transcription uses JSON revision events', 'response_format');
  for (const key of Object.keys(body)) {
    if (!['model', 'language', 'prompt', 'stream', 'response_format', 'sampling', 'stream_options'].includes(key))
      bad(`Unsupported transcription parameter: ${key}`, key);
  }
  const sampling = extension(body.sampling, 'sampling');
  const rolling = extension(body.stream_options, 'stream_options');
  for (const key of Object.keys(sampling))
    if (key !== 'max_tokens') bad(`Unsupported sampling.${key}`, `sampling.${key}`);
  for (const key of Object.keys(rolling))
    if (!['chunk_seconds', 'provisional_tokens', 'unfixed_chunks'].includes(key))
      bad(`Unsupported stream_options.${key}`, `stream_options.${key}`);
  if (!stream && Object.keys(rolling).length) bad('stream_options requires stream:true', 'stream_options');
  const options: Qwen3AsrTranscribeOptions & Qwen3AsrStreamOptions = {};
  if (sampling.max_tokens !== undefined)
    options.maxTokens = numeric(sampling.max_tokens, 'sampling.max_tokens', 1, 16384, true);
  else if (stream) options.maxTokens = 32;
  if (body.language !== undefined) options.language = string(body.language, 'language');
  if (body.prompt !== undefined) {
    if (typeof body.prompt !== 'string' || textLength(body.prompt) > 4096) bad('Invalid prompt', 'prompt');
    options.prompt = body.prompt as string;
  }
  for (const [key, target, min, max, integer] of [
    ['chunk_seconds', 'chunkSeconds', 0.1, 10, false],
    ['provisional_tokens', 'provisionalTokens', 0, 128, true],
    ['unfixed_chunks', 'unfixedChunks', 0, 32, true],
  ] as const)
    if (rolling[key] !== undefined) options[target] = numeric(rolling[key], `stream_options.${key}`, min, max, integer);
  return { model, stream, format, options };
}
