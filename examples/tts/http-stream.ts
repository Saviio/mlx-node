import { once } from 'node:events';
/** Run: node examples/tts/http-stream.ts MODEL VOICE [output.pcm]
 * Node 22.18+ supports this TypeScript file directly. The input producer and
 * response consumer run concurrently; fetch in browsers does not promise that.
 */
import { createWriteStream } from 'node:fs';
import { request } from 'node:http';
import { request as secureRequest } from 'node:https';
import { pipeline } from 'node:stream/promises';

const model = process.argv[2];
const voice = process.argv[3];
if (!model || !voice) throw new Error('Usage: http-stream.ts MODEL VOICE [output.pcm]');
const url = new URL('/v1/audio/speech/stream', process.env.MLX_BASE_URL ?? 'http://127.0.0.1:8000');
const token = process.env.MLX_API_KEY;
const req = (url.protocol === 'https:' ? secureRequest : request)(url, {
  method: 'POST',
  headers: {
    'Content-Type': 'application/x-ndjson',
    ...(token ? { Authorization: `Bearer ${token}` } : {}),
  },
});
const received = new Promise<void>((resolve, reject) => {
  req.on('error', reject);
  req.on('response', (res) => {
    if (res.statusCode !== 200) {
      let message = '';
      res.setEncoding('utf8');
      res.on('data', (chunk) => {
        message += chunk;
      });
      res.on('error', reject);
      res.on('end', () => {
        reject(new Error(`${res.statusCode}: ${message}`));
        req.destroy();
      });
      return;
    }
    console.log('PCM16 little-endian:', res.headers['x-sample-rate'], 'Hz,', res.headers['x-channels'], 'channel(s)');
    // pipeline applies output backpressure; it rejects an incomplete response.
    pipeline(res, createWriteStream(process.argv[4] ?? 'output.pcm')).then(() => {
      console.log('Final statistics:', res.trailers);
      resolve();
    }, reject);
  });
});
async function send(value: unknown) {
  if (!req.write(JSON.stringify(value) + '\n')) await once(req, 'drain');
}
async function produce() {
  await send({ type: 'config', model, voice, response_format: 'pcm' });
  await send({ type: 'text', text: 'Hello. ' });
  await send({ type: 'flush' });
  // Replace this loop with tokens from your language model. Audio can arrive
  // while this producer is still awaiting its next text fragment.
  for (const text of ['We can receive ', 'speech while sending ', 'the remaining text.']) {
    await new Promise((resolve) => setTimeout(resolve, 500));
    await send({ type: 'text', text });
  }
  req.end(); // A normal request EOF flushes the remaining text.
}
try {
  await Promise.all([produce(), received]);
} catch (error) {
  req.destroy();
  throw error;
}
