import { randomUUID } from 'node:crypto';
import type { EventEmitter } from 'node:events';
import { mkdtemp, open, rm } from 'node:fs/promises';
import type { FileHandle } from 'node:fs/promises';
import type { IncomingMessage, ServerResponse } from 'node:http';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import type { Readable } from 'node:stream';
import { TextDecoder } from 'node:util';

import { Busboy, type BusboyInstance } from '@fastify/busboy';

export class AudioError extends Error {
  constructor(
    readonly status: number,
    readonly code: string,
    message: string,
    readonly param: string | null = null,
  ) {
    super(message);
  }
}
export const bad = (message: string, param: string | null = null): never => {
  throw new AudioError(400, 'invalid_request', message, param);
};
export function object(value: unknown, param = 'body'): Record<string, unknown> {
  if (!value || typeof value !== 'object' || Array.isArray(value)) bad(`${param} must be an object`, param);
  return value as Record<string, unknown>;
}
export function string(value: unknown, param: string, required = true): string | undefined {
  if (value === undefined && !required) return;
  if (typeof value !== 'string' || !value.trim()) bad(`${param} must be a nonempty string`, param);
  return value as string;
}

/** Shared disk accounting, including WAV responses still being downloaded. */
export class AudioTempStore {
  private root?: Promise<string>;
  private bytes = 0;
  constructor(readonly limit = 512 * 1024 * 1024) {}
  async create(): Promise<AudioTempFile> {
    if (!this.root) {
      const creating = mkdtemp(join(tmpdir(), `mlx-audio-${process.pid}-`)).catch((error) => {
        // Share an in-flight creation, but allow recovery after transient I/O errors.
        if (this.root === creating) this.root = undefined;
        throw error;
      });
      this.root = creating;
    }
    const path = join(await this.root, randomUUID());
    return new AudioTempFile(path, await open(path, 'wx', 0o600), (delta) => {
      if (delta > 0 && this.bytes + delta > this.limit)
        throw new AudioError(503, 'audio_storage_full', 'Audio temporary storage is full');
      this.bytes += delta;
    });
  }
  async close(): Promise<void> {
    if (this.root) await rm(await this.root, { recursive: true, force: true });
  }
}
export class AudioTempFile {
  private bytes = 0;
  private closed = false;
  get bytesReserved(): number {
    return this.bytes;
  }
  constructor(
    readonly path: string,
    readonly file: FileHandle,
    private readonly account: (bytes: number) => void,
  ) {}
  reserve(bytes: number): void {
    this.account(bytes);
    this.bytes += bytes;
  }
  async write(buffer: Uint8Array): Promise<void> {
    this.reserve(buffer.byteLength);
    let at = 0;
    while (at < buffer.byteLength) {
      const { bytesWritten } = await this.file.write(buffer, at, buffer.byteLength - at);
      if (!bytesWritten) throw new Error('Audio file write made no progress');
      at += bytesWritten;
    }
  }
  async dispose(): Promise<void> {
    if (this.closed) return;
    this.closed = true;
    try {
      await this.file.close();
    } finally {
      try {
        await rm(this.path, { force: true });
      } finally {
        this.account(-this.bytes);
      }
    }
  }
}

/** Wait only while the consumer is asking for bytes: sink backpressure does
 * not count as input idleness. No data listener may put the request in flowing mode. */
export async function* inputBytes(
  req: IncomingMessage,
  signal: AbortSignal,
  limit: number,
  idleMs = 30_000,
): AsyncGenerator<Buffer> {
  let total = 0;
  while (true) {
    signal.throwIfAborted();
    const chunk = req.read() as Buffer | null;
    if (chunk !== null) {
      total += chunk.length;
      if (total > limit) throw new AudioError(413, 'request_too_large', 'Audio request exceeds byte limit');
      yield chunk;
      continue;
    }
    if (req.readableEnded) return;
    if (req.destroyed) throw new Error('Audio upload disconnected');
    await new Promise<void>((resolve, reject) => {
      const finish = (error?: unknown) => {
        clearTimeout(timer);
        req.off('readable', ready);
        req.off('end', ready);
        req.off('error', failed);
        req.off('close', closed);
        signal.removeEventListener('abort', aborted);
        if (error) reject(error);
        else resolve();
      };
      const ready = () => finish();
      const failed = (error: Error) => finish(error);
      const closed = () => finish(req.complete ? undefined : new Error('Audio upload disconnected'));
      const aborted = () => finish(signal.reason ?? new Error('Audio request cancelled'));
      const timer = setTimeout(() => finish(new AudioError(408, 'input_timeout', 'Audio input timed out')), idleMs);
      req.once('readable', ready);
      req.once('end', ready);
      req.once('error', failed);
      req.once('close', closed);
      signal.addEventListener('abort', aborted, { once: true });
      if (signal.aborted) aborted();
      else if (req.readableLength || req.readableEnded) ready();
    });
  }
}

export async function readJson(
  req: IncomingMessage,
  signal: AbortSignal,
  idleMs: number,
): Promise<Record<string, unknown>> {
  const chunks: Buffer[] = [];
  for await (const chunk of inputBytes(req, signal, 64 * 1024, idleMs)) chunks.push(chunk);
  try {
    return object(JSON.parse(new TextDecoder('utf-8', { fatal: true }).decode(Buffer.concat(chunks))));
  } catch (error) {
    if (error instanceof AudioError) throw error;
    return bad('Invalid JSON request');
  }
}

export async function* ndjson(req: IncomingMessage, signal: AbortSignal, idleMs: number): AsyncGenerator<unknown> {
  const decoder = new TextDecoder('utf-8', { fatal: true });
  let pending = '';
  const parse = (line: string): unknown => {
    if (Buffer.byteLength(line) > 64 * 1024) bad('NDJSON record exceeds 64 KiB');
    try {
      return JSON.parse(line);
    } catch {
      return bad('Invalid NDJSON record');
    }
  };
  try {
    for await (const bytes of inputBytes(req, signal, 1024 * 1024, idleMs)) {
      pending += decoder.decode(bytes, { stream: true });
      let end: number;
      while ((end = pending.indexOf('\n')) >= 0) {
        const line = pending.slice(0, end);
        pending = pending.slice(end + 1);
        if (line.trim()) yield parse(line);
      }
      if (Buffer.byteLength(pending) > 64 * 1024) bad('NDJSON record exceeds 64 KiB');
    }
    pending += decoder.decode();
    if (pending.trim()) yield parse(pending);
  } catch (error) {
    if (error instanceof TypeError) bad('Invalid UTF-8 input');
    throw error;
  }
}

export async function uploadWav(
  req: IncomingMessage,
  signal: AbortSignal,
  file: AudioTempFile,
  maxBytes: number,
  idleMs: number,
  fileField = 'file',
): Promise<Record<string, unknown>> {
  const type = req.headers['content-type'] ?? '';
  if (/^audio\/wav(?:;|$)/i.test(type)) {
    for await (const chunk of inputBytes(req, signal, maxBytes, idleMs)) await file.write(chunk);
    return Object.fromEntries(new URL(req.url ?? '/', 'http://localhost').searchParams);
  }
  if (!/^multipart\/form-data(?:;|$)/i.test(type))
    throw new AudioError(415, 'unsupported_media_type', 'Expected audio/wav or multipart/form-data');
  let parser: BusboyInstance;
  try {
    parser = new Busboy({
      headers: { ...req.headers, 'content-type': type },
      limits: { files: 1, fields: 16, parts: 17, fileSize: maxBytes, fieldSize: 64 * 1024 },
    });
  } catch {
    return bad('Invalid multipart boundary');
  }
  const fields: Record<string, unknown> = Object.create(null);
  let seen = false;
  let failure: unknown;
  let writing: Promise<void> = Promise.resolve();
  let part: Readable | undefined;
  const inputAbort = new AbortController();
  const inputSignal = AbortSignal.any([signal, inputAbort.signal]);
  const stopInput = (error: unknown) => {
    failure ??= error;
    // A file write can fail while the request is waiting for its next chunk.
    // Wake that wait without aborting the HTTP response carrying the error.
    inputAbort.abort(failure);
    part?.destroy(failure as Error);
  };
  const fail = (error: unknown) => {
    stopInput(error);
    parser.destroy(failure as Error);
  };
  const complete = new Promise<void>((resolve, reject) => {
    parser.once('finish', resolve);
    // Parser errors can be followed by another error during stream teardown.
    parser.on('error', (error: Error) => {
      stopInput(error instanceof AudioError ? error : new AudioError(400, 'invalid_request', error.message, fileField));
      reject(failure);
    });
  });
  void complete.catch(() => {});
  parser.on('field', (name, value, nameTruncated, valueTruncated) => {
    if (nameTruncated || valueTruncated || Object.hasOwn(fields, name))
      fail(new AudioError(400, 'invalid_request', 'Duplicate or oversized multipart field', name));
    else fields[name] = value;
  });
  for (const event of ['filesLimit', 'fieldsLimit', 'partsLimit'] as const)
    parser.on(event, () => fail(new AudioError(400, 'invalid_request', 'Too many multipart parts')));
  parser.on('file', (name, stream) => {
    if (name !== fileField || seen) {
      stream.resume();
      fail(new AudioError(400, 'invalid_request', `Expected one ${fileField} part`, fileField));
      return;
    }
    seen = true;
    part = stream;
    stream.on('limit', () =>
      fail(new AudioError(413, 'request_too_large', 'WAV upload exceeds byte limit', fileField)),
    );
    writing = (async () => {
      for await (const chunk of stream) {
        inputSignal.throwIfAborted();
        await file.write(chunk);
      }
      if (stream.truncated) throw new AudioError(413, 'request_too_large', 'WAV upload was truncated', fileField);
    })();
    void writing.catch(fail);
  });
  try {
    for await (const chunk of inputBytes(req, inputSignal, maxBytes + 1024 * 1024, idleMs)) {
      if (failure) throw failure;
      const accepted = parser.write(chunk);
      if (failure) throw failure;
      if (!accepted) await drain(parser, inputSignal, idleMs);
    }
    parser.end();
    await complete;
    await writing;
    if (failure) throw failure;
    if (!seen) bad(`${fileField} is required`, fileField);
    return fields;
  } catch (error) {
    throw failure ?? error;
  } finally {
    parser.destroy();
    part?.destroy();
    await writing.catch(() => {});
  }
}

export async function drain(
  stream: Pick<EventEmitter, 'on' | 'off'> & { writableNeedDrain: boolean; destroyed: boolean },
  signal: AbortSignal,
  timeoutMs = 30_000,
): Promise<void> {
  signal.throwIfAborted();
  if (stream.destroyed) throw new Error('Audio output disconnected');
  if (!stream.writableNeedDrain) return;
  await new Promise<void>((resolve, reject) => {
    const finish = (error?: unknown) => {
      clearTimeout(timer);
      stream.off('drain', ready);
      stream.off('close', closed);
      stream.off('error', failed);
      signal.removeEventListener('abort', aborted);
      if (error) reject(error);
      else resolve();
    };
    const ready = () => finish();
    const closed = () => finish(new Error('Audio output disconnected'));
    const failed = (error: Error) => finish(error);
    const aborted = () => finish(signal.reason ?? new Error('Audio request cancelled'));
    const timer = setTimeout(() => finish(new AudioError(408, 'output_timeout', 'Audio output timed out')), timeoutMs);
    stream.on('drain', ready);
    stream.on('close', closed);
    stream.on('error', failed);
    signal.addEventListener('abort', aborted, { once: true });
    if (signal.aborted) aborted();
    else if (stream.destroyed) closed();
    else if (!stream.writableNeedDrain) ready();
  });
}
export async function write(
  res: ServerResponse,
  data: string | Uint8Array,
  signal: AbortSignal,
  timeoutMs: number,
): Promise<void> {
  signal.throwIfAborted();
  if (res.destroyed) throw new Error('Audio output disconnected');
  if (!res.write(data)) await drain(res, signal, timeoutMs);
}
export function event(type: string, data: Record<string, unknown>): string {
  return `event: ${type}\ndata: ${JSON.stringify({ type, ...data })}\n\n`;
}
