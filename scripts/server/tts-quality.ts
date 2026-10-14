/**
 * Seeded Qwen3-TTS output, transcribed by in-process Qwen3-ASR and scored with the sglang-omni TTS WER rules.
 * Run with oxnode, no server.
 *
 *   oxnode scripts/server/tts-quality.ts run --cv-model <dir> --base-model <dir> --asr-model <dir> --clips <clip-dir>
 *     --out <dir> --seed <n> [--voices cv,base-0,base-1,base-2] [--en 100] [--zh 50] [--refs 3] [--voice ryan]
 *   oxnode scripts/server/tts-quality.ts report --out <dir> --whisper-json <english.json> [--baseline <dir>]
 *
 * Texts: the first `en` rows of split en and the first `zh` rows of split zh with a distinct target_text, in the bench
 * client's order (sglang-omni benchmarks/dataset/seedtts.py:164-165). A repeated text at a fixed seed is a byte copy.
 * <clip-dir> is the export asr-golden-ids.ts reads; its en200/zh200 fingerprints are checked. The id list goes to
 * <out>/texts.json and every later call must reproduce it.
 * Voices: `cv` = CustomVoice preset `--voice`; `base-N` = Base clone of the N-th EN row with a distinct reference wav,
 * prepared like POST /v1/audio/voices (tts-golden.ts rule). Every text is synthesized with { voice, seed } only.
 * Each PCM goes to startTranscription(pcm, { sampleRate, language: en|zh, maxTokens: 4096 }), the server's non-stream
 * options; the native resampler takes it to 16 kHz. The bench sends int16 WAV; this sends the float PCM.
 * `run` writes seed-<n>/<voice>.jsonl + .meta.json and raw float32 PCM in seed-<n>/<voice>/. An existing cell is
 * refused, so a seed can be split over several calls by --voices.
 *
 * `report` scores every row like sglang-omni: normalize_text + apply_wer (benchmarks/tasks/asr.py:74-94, 283-298:
 * jiwer 4.0.0 process_words with wer_default, rapidfuzz 3.14.6 Levenshtein editops) and calculate_wer_metrics
 * (benchmarks/metrics/wer.py:40-115). EN = openai-whisper 20250625 EnglishTextNormalizer (normalizers/english.py,
 * basic.py), whose spelling map english.json is --whisper-json (sha256 pinned). ZH = remove zhon 2.1.1
 * hanzi.punctuation and string.punctuation except "'", drop spaces, split into characters. A row with empty PCM is a
 * generation failure and is left out of WER, as in benchmarks/tasks/tts.py:1315-1320 and :554-557.
 * Groups cv-en, cv-zh, base-en, base-zh. Per seed and group: wer_corpus, wer_below_50_corpus and n_above_50_pct_wer
 * (their summary rows "WER (corpus)", "WER corpus (excl >50%)", ">50% WER samples").
 * --baseline: the call is refused unless both runs have the same text set, the same seeds and, per cell, the same TTS
 * and ASR model paths (meta.json). For every group and value, the mean over the seeds must not exceed the baseline
 * max, and the candidate may have no failed rows; exit 1 otherwise. Each seed also prints PCM / ASR-text identity.
 * Speaker similarity is not reported: the native TTS API exposes no speaker embedding.
 */
import { execFileSync } from 'node:child_process';
import { createHash } from 'node:crypto';
import { existsSync, mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join, resolve } from 'node:path';
import { parseArgs } from 'node:util';

import { Qwen3AsrModel } from '@mlx-node/asr';
import { WavReader, loadTtsModel, type PreparedVoice } from '@mlx-node/tts';

const WHISPER_JSON_SHA256 = '6607f948be9824d2e1b2fa2223cd94c06c45afa4e05ea0e3d5e1f2bdffde2465';
const ASR_MAX_TOKENS = 4096;
const usage = `Usage:
  oxnode scripts/server/tts-quality.ts run --cv-model <dir> --base-model <dir> --asr-model <dir> --clips <dir> --out <dir> --seed <n> [--voices cv,base-0,...] [--en 100] [--zh 50] [--refs 3] [--voice ryan]
  oxnode scripts/server/tts-quality.ts report --out <dir> --whisper-json <english.json> [--baseline <dir>]`;

const { values, positionals } = parseArgs({
  allowPositionals: true,
  options: {
    'cv-model': { type: 'string' },
    'base-model': { type: 'string' },
    'asr-model': { type: 'string' },
    clips: { type: 'string' },
    out: { type: 'string' },
    seed: { type: 'string' },
    voices: { type: 'string' },
    en: { type: 'string', default: '100' },
    zh: { type: 'string', default: '50' },
    refs: { type: 'string', default: '3' },
    voice: { type: 'string', default: 'ryan' },
    'whisper-json': { type: 'string' },
    baseline: { type: 'string' },
  },
});
const int = (raw: string | undefined, name: string, min: number, max = Number.MAX_SAFE_INTEGER): number => {
  const value = Number(raw);
  if (raw === undefined || !Number.isSafeInteger(value) || value < min || value > max)
    throw new Error(`--${name} must be an integer in ${min}..${max}\n${usage}`);
  return value;
};
const readJsonl = <T>(path: string): T[] =>
  readFileSync(path, 'utf8')
    .split('\n')
    .filter((line) => line.trim())
    .map((line) => JSON.parse(line) as T);
const sha256 = (data: string | Buffer) => createHash('sha256').update(data).digest('hex');
const git = (args: string[]) => {
  try {
    return execFileSync('git', args, { encoding: 'utf8' }).trim();
  } catch {
    return null;
  }
};
const mapped = () => {
  try {
    return execFileSync('lsof', ['-p', String(process.pid)], { encoding: 'utf8' })
      .split('\n')
      .filter((line) => /\.node$|\.metallib$/.test(line))
      .map((line) => line.split(/\s+/).slice(8).join(' '));
  } catch {
    return [];
  }
};

type Lang = 'en' | 'zh';
interface ClipRow {
  index: number;
  id: string;
  ref_text: string;
  target_text: string;
  ref_audio: string;
}
interface TextItem {
  lang: Lang;
  index: number;
  clip: string;
  text: string;
}
interface RefItem {
  index: number;
  clip: string;
  refAudio: string;
  refText: string;
  wavSha256: string;
}
interface TextSet {
  rule: string;
  clipsDir: string;
  dataset: unknown;
  voice: string;
  voices: string[];
  texts: TextItem[];
  refs: RefItem[];
  asr: { maxTokens: number; language: string };
  digest: string;
}
interface Row {
  id: string;
  voice: string;
  lang: Lang;
  clip: string;
  seed: number;
  text: string;
  samples: number;
  sampleRate: number;
  audioSeconds: number;
  finishReason: string;
  pcmSha256: string;
  synthesisMs: number;
  firstPcmMs: number | null;
  asrText: string | null;
  asrTokens: number | null;
  asrReachedMaxTokens: boolean | null;
  asrTotalMs: number | null;
  error: string | null;
}
const itemId = (voice: string, item: TextItem) => `${voice}-${item.lang}-${String(item.index).padStart(3, '0')}`;

// Mirrors audio-service.ts createVoice(): WavReader + mono chunks of 16384 frames.
async function readClip(path: string): Promise<{ samples: Float32Array; sampleRate: number }> {
  const reader = await WavReader.open(path);
  try {
    const samples = new Float32Array(reader.frames);
    let offset = 0;
    for await (const chunk of reader.chunks(16384, true)) {
      samples.set(chunk, offset);
      offset += chunk.length;
    }
    return { samples, sampleRate: reader.sampleRate };
  } finally {
    await reader.close();
  }
}

function buildTextSet(clipsDir: string, enCount: number, zhCount: number, refCount: number, voice: string): TextSet {
  const sets = JSON.parse(readFileSync(join(clipsDir, 'sets.json'), 'utf8')) as {
    dataset: string;
    revision: string;
    sets: Record<string, { split: string; count: number; fingerprint: string }>;
  };
  const split = (lang: Lang, count: number): ClipRow[] => {
    const info = sets.sets[`${lang}200`];
    if (!info || info.split !== lang || count > info.count)
      throw new Error(`--${lang} must be <= ${info?.count ?? 0} (fingerprinted ${lang}200 rows)`);
    const rows = readJsonl<ClipRow>(join(clipsDir, `${lang}.jsonl`)).slice(0, info.count);
    // Same digest as sglang-omni benchmarks/eval/benchmark_asr_seedtts.py::_evaluation_input_sha256.
    const digest = createHash('sha256').update('seedtts-evaluation-input-v1\0');
    const field = (bytes: Buffer) => {
      const size = Buffer.alloc(8);
      size.writeBigUInt64BE(BigInt(bytes.length));
      digest.update(size).update(bytes);
    };
    for (const row of rows) {
      for (const value of [row.id, row.ref_text, row.target_text]) field(Buffer.from(value, 'utf8'));
      field(readFileSync(join(clipsDir, row.ref_audio)));
    }
    const fingerprint = digest.digest('hex');
    if (rows.length !== info.count || fingerprint !== info.fingerprint)
      throw new Error(`${lang}200 fingerprint ${fingerprint} (${rows.length} rows) != sets.json ${info.fingerprint}`);
    const picked: ClipRow[] = [];
    for (const row of rows)
      if (picked.length < count && !picked.some((p) => p.target_text === row.target_text)) picked.push(row);
    if (picked.length !== count) throw new Error(`${lang}200 has ${picked.length} distinct texts, need ${count}`);
    return picked;
  };
  const en = split('en', enCount);
  const zh = split('zh', zhCount);
  const texts: TextItem[] = [
    ...en.map((row) => ({ lang: 'en' as const, index: row.index, clip: row.id, text: row.target_text })),
    ...zh.map((row) => ({ lang: 'zh' as const, index: row.index, clip: row.id, text: row.target_text })),
  ];
  const refs: RefItem[] = [];
  for (const row of readJsonl<ClipRow>(join(clipsDir, 'en.jsonl'))) {
    if (refs.length === refCount) break;
    if (!refs.some((ref) => ref.refAudio === row.ref_audio))
      refs.push({
        index: row.index,
        clip: row.id,
        refAudio: row.ref_audio,
        refText: row.ref_text,
        wavSha256: sha256(readFileSync(join(clipsDir, row.ref_audio))),
      });
  }
  if (refs.length !== refCount) throw new Error(`Need ${refCount} distinct references, found ${refs.length}`);
  const voices = ['cv', ...refs.map((_, i) => `base-${i}`)];
  const asr = { maxTokens: ASR_MAX_TOKENS, language: 'the text language (en or zh)' };
  const rule =
    `first ${enCount} distinct target_text rows of split en and first ${zhCount} of split zh in dataset order ` +
    `(within the bench client's --max-samples 200 rows); base-N = the N-th EN row with a distinct ref_audio`;
  const digest = sha256(JSON.stringify({ rule, voice, voices, texts, refs, asr }));
  return {
    rule,
    clipsDir,
    dataset: { repo: sets.dataset, revision: sets.revision },
    voice,
    voices,
    texts,
    refs,
    asr,
    digest,
  };
}

async function run() {
  if (!values.out || !values.clips || !values['asr-model']) throw new Error(usage);
  const seed = int(values.seed, 'seed', 0, 0xffffffff);
  const outDir = resolve(values.out);
  const set = buildTextSet(
    resolve(values.clips),
    int(values.en, 'en', 0),
    int(values.zh, 'zh', 0),
    int(values.refs, 'refs', 0),
    values.voice,
  );
  if (!set.texts.length) throw new Error('The text set is empty');
  const textsPath = join(outDir, 'texts.json');
  if (existsSync(textsPath)) {
    const stored = JSON.parse(readFileSync(textsPath, 'utf8')) as TextSet;
    if (stored.digest !== set.digest)
      throw new Error(`${textsPath} has digest ${stored.digest}, this call ${set.digest}`);
  } else {
    mkdirSync(outDir, { recursive: true });
    writeFileSync(textsPath, JSON.stringify(set, null, 1) + '\n');
  }
  const voices = values.voices ? values.voices.split(',') : set.voices;
  for (const voice of voices)
    if (!set.voices.includes(voice)) throw new Error(`Unknown voice ${voice}; ${set.voices.join(',')}`);
  const seedDir = join(outDir, `seed-${seed}`);
  for (const voice of voices)
    if (existsSync(join(seedDir, `${voice}.jsonl`))) throw new Error(`${join(seedDir, voice)}.jsonl exists`);
  const stages = [
    { path: values['cv-model'], name: '--cv-model', cells: voices.filter((voice) => voice === 'cv') },
    { path: values['base-model'], name: '--base-model', cells: voices.filter((voice) => voice !== 'cv') },
  ].filter((stage) => stage.cells.length);
  for (const stage of stages)
    if (!stage.path) throw new Error(`${stage.name} is required for ${stage.cells.join(',')}`);
  mkdirSync(seedDir, { recursive: true });

  const asrPath = resolve(values['asr-model']);
  const asr = await Qwen3AsrModel.load(asrPath);
  try {
    for (const stage of stages) {
      const modelPath = resolve(stage.path!);
      const model = await loadTtsModel(modelPath);
      try {
        const preset = model.capabilities.voices.find((v) => v.toLowerCase() === set.voice.toLowerCase());
        if (stage.cells.includes('cv') && !preset)
          throw new Error(`Voice ${set.voice} not in ${JSON.stringify(model.capabilities.voices)}`);
        let warm = false;
        for (const cell of stage.cells) {
          const started = performance.now();
          let voice: string | PreparedVoice;
          let prepareMs = 0;
          if (cell === 'cv') voice = preset!;
          else {
            if (!model.capabilities.voiceCloning) throw new Error(`${modelPath} does not support voice cloning`);
            const ref = set.refs[Number(cell.slice('base-'.length))];
            const audio = await readClip(join(set.clipsDir, ref.refAudio));
            const prepareStart = performance.now();
            voice = await model.prepareVoice({
              audio: { samples: audio.samples, sampleRate: audio.sampleRate, channels: 1 },
              transcript: ref.refText,
            });
            prepareMs = performance.now() - prepareStart;
          }
          const rows: Row[] = [];
          const pcm: Float32Array[] = [];
          const pcmDir = join(seedDir, cell);
          mkdirSync(pcmDir, { recursive: true });
          try {
            // Unrecorded, as the bench client warms up before timing.
            if (!warm) await model.synthesize(set.texts[0].text, { voice, seed });
            warm = true;
            for (const item of set.texts) {
              const id = itemId(cell, item);
              const audio = await model.synthesize(item.text, { voice, seed });
              if (audio.channels !== 1) throw new Error(`${id}: ${audio.channels} channels`);
              const bytes = Buffer.from(audio.samples.buffer, audio.samples.byteOffset, audio.samples.byteLength);
              writeFileSync(join(pcmDir, `${id}.f32`), bytes);
              pcm.push(audio.samples);
              rows.push({
                id,
                voice: cell,
                lang: item.lang,
                clip: item.clip,
                seed,
                text: item.text,
                samples: audio.samples.length,
                sampleRate: audio.sampleRate,
                audioSeconds: audio.stats.audioSeconds,
                finishReason: audio.stats.finishReason,
                pcmSha256: sha256(bytes),
                synthesisMs: audio.stats.synthesisMs,
                firstPcmMs: audio.stats.firstPcmMs,
                asrText: null,
                asrTokens: null,
                asrReachedMaxTokens: null,
                asrTotalMs: null,
                error: audio.samples.length ? null : 'Generation failed: empty audio',
              });
              if (rows.length % 25 === 0)
                process.stderr.write(`[tts-quality] seed ${seed} ${cell} tts ${rows.length}\n`);
            }
          } finally {
            if (typeof voice !== 'string') await voice.dispose();
          }
          const synthWallMs = performance.now() - started;
          const asrStart = performance.now();
          for (const [i, row] of rows.entries()) {
            if (row.error) continue;
            const result = await asr
              .startTranscription(pcm[i], { sampleRate: row.sampleRate, language: row.lang, maxTokens: ASR_MAX_TOKENS })
              .result();
            row.asrText = result.text;
            row.asrTokens = result.tokenIds.length;
            row.asrReachedMaxTokens = result.reachedMaxTokens;
            row.asrTotalMs = result.totalMs;
          }
          const asrWallMs = performance.now() - asrStart;
          if (rows.length !== set.texts.length)
            throw new Error(`${cell}: ${rows.length} rows, ${set.texts.length} texts`);
          writeFileSync(join(seedDir, `${cell}.jsonl`), rows.map((row) => JSON.stringify(row)).join('\n') + '\n');
          const summary = {
            seed,
            voice: cell,
            rows: rows.length,
            failed: rows.filter((row) => row.error).length,
            length: rows.filter((row) => row.finishReason === 'length').length,
            audioSeconds: rows.reduce((sum, row) => sum + row.audioSeconds, 0),
            synthesisMs: rows.reduce((sum, row) => sum + row.synthesisMs, 0),
            prepareMs,
            synthWallMs,
            asrWallMs,
          };
          writeFileSync(
            join(seedDir, `${cell}.meta.json`),
            JSON.stringify(
              {
                ...summary,
                textsDigest: set.digest,
                model: modelPath,
                asrModel: asrPath,
                cwd: process.cwd(),
                gitHead: git(['rev-parse', 'HEAD']),
                gitDirtyFiles: git(['status', '--porcelain'])?.split('\n').filter(Boolean).length ?? null,
                addonDir: process.env.MLX_ADDON_DIR ?? null,
                mapped: mapped(),
                node: process.version,
              },
              null,
              1,
            ) + '\n',
          );
          console.log(JSON.stringify(summary));
        }
      } finally {
        await model.dispose();
      }
    }
  } finally {
    await asr.dispose();
  }
}

// Python `re` str-pattern semantics: \s = str.isspace(), \d = Unicode decimal, \w = alphanumeric or '_'.
const PY_SPACE = '[\\t-\\r\\x1c-\\x20\\x85\\xa0\\u1680\\u2000-\\u200a\\u2028\\u2029\\u202f\\u205f\\u3000]';
const PY_WORD = '[\\p{L}\\p{N}_]';
const PY_BOUNDARY = `(?:(?<=${PY_WORD})(?!${PY_WORD})|(?<!${PY_WORD})(?=${PY_WORD}))`;
const py = (pattern: string, flags = 'gu') =>
  new RegExp(pattern.replaceAll('\\b', PY_BOUNDARY).replaceAll('\\s', PY_SPACE).replaceAll('\\d', '\\p{Nd}'), flags);
const SPACE_RUN = py('\\s+', 'u');
const STRIP = py('^\\s+|\\s+$');
const pyStrip = (s: string) => s.replace(STRIP, '');
const pySplit = (s: string) => s.split(SPACE_RUN).filter((word) => word !== '');
const ND = /\p{Nd}/u;
const asciiDigits = (s: string) =>
  s.replace(/\p{Nd}/gu, (digit) => {
    const point = digit.codePointAt(0)!;
    let zero = point;
    while (ND.test(String.fromCodePoint(zero - 1))) zero--;
    return String((point - zero) % 10);
  });

// whisper/normalizers/basic.py:7-47
const ADDITIONAL_DIACRITICS = new Map(
  Object.entries({
    œ: 'oe',
    Œ: 'OE',
    ø: 'o',
    Ø: 'O',
    æ: 'ae',
    Æ: 'AE',
    ß: 'ss',
    ẞ: 'SS',
    đ: 'd',
    Đ: 'D',
    ð: 'd',
    Ð: 'D',
    þ: 'th',
    Þ: 'th',
    ł: 'l',
    Ł: 'L',
  }),
);
const MN = /^\p{Mn}$/u;
const MSP = /^[\p{M}\p{S}\p{P}]$/u;
function removeSymbolsAndDiacritics(s: string, keep: string): string {
  let out = '';
  for (const c of s.normalize('NFKD')) {
    if (keep.includes(c)) out += c;
    else if (ADDITIONAL_DIACRITICS.has(c)) out += ADDITIONAL_DIACRITICS.get(c);
    else if (MN.test(c)) continue;
    else out += MSP.test(c) ? ' ' : c;
  }
  return out;
}

// whisper/normalizers/english.py:12-447 EnglishNumberNormalizer
const ONES = new Map(
  [
    'one',
    'two',
    'three',
    'four',
    'five',
    'six',
    'seven',
    'eight',
    'nine',
    'ten',
    'eleven',
    'twelve',
    'thirteen',
    'fourteen',
    'fifteen',
    'sixteen',
    'seventeen',
    'eighteen',
    'nineteen',
  ].map((name, i) => [name, BigInt(i + 1)]),
);
const ZEROS = new Set(['o', 'oh', 'zero']);
const ONES_SUFFIXED = new Map<string, [bigint, string]>([
  ...[...ONES].map(([name, value]): [string, [bigint, string]] => [
    name === 'six' ? 'sixes' : `${name}s`,
    [value, 's'],
  ]),
  ['zeroth', [0n, 'th']],
  ['first', [1n, 'st']],
  ['second', [2n, 'nd']],
  ['third', [3n, 'rd']],
  ['fifth', [5n, 'th']],
  ['twelfth', [12n, 'th']],
  ...[...ONES]
    .filter(([, value]) => value > 3n && value !== 5n && value !== 12n)
    .map(([name, value]): [string, [bigint, string]] => [name + (name.endsWith('t') ? 'h' : 'th'), [value, 'th']]),
]);
const TENS = new Map(
  ['twenty', 'thirty', 'forty', 'fifty', 'sixty', 'seventy', 'eighty', 'ninety'].map((name, i) => [
    name,
    BigInt((i + 2) * 10),
  ]),
);
const TENS_SUFFIXED = new Map<string, [bigint, string]>([
  ...[...TENS].map(([name, value]): [string, [bigint, string]] => [name.replaceAll('y', 'ies'), [value, 's']]),
  ...[...TENS].map(([name, value]): [string, [bigint, string]] => [name.replaceAll('y', 'ieth'), [value, 'th']]),
]);
const MULTIPLIERS = new Map(
  [
    'hundred',
    'thousand',
    'million',
    'billion',
    'trillion',
    'quadrillion',
    'quintillion',
    'sextillion',
    'septillion',
    'octillion',
    'nonillion',
    'decillion',
  ].map((name, i) => [name, 10n ** BigInt(i ? 3 * i : 2)]),
);
const MULTIPLIERS_SUFFIXED = new Map<string, [bigint, string]>([
  ...[...MULTIPLIERS].map(([name, value]): [string, [bigint, string]] => [`${name}s`, [value, 's']]),
  ...[...MULTIPLIERS].map(([name, value]): [string, [bigint, string]] => [`${name}th`, [value, 'th']]),
]);
const DECIMALS = new Set([...ONES.keys(), ...TENS.keys(), ...ZEROS]);
const PRECEDING_PREFIXERS = new Map(Object.entries({ minus: '-', negative: '-', plus: '+', positive: '+' }));
const FOLLOWING_PREFIXERS = new Map(
  Object.entries({ pound: '£', pounds: '£', euro: '€', euros: '€', dollar: '$', dollars: '$', cent: '¢', cents: '¢' }),
);
const PREFIXES = new Set([...PRECEDING_PREFIXERS.values(), ...FOLLOWING_PREFIXERS.values()]);
const SUFFIXERS = new Map<string, string | Map<string, string>>([
  ['per', new Map([['cent', '%']])],
  ['percent', '%'],
]);
const SPECIALS = new Set(['and', 'double', 'triple', 'point']);
const WORDS = new Set([
  ...ZEROS,
  ...ONES.keys(),
  ...ONES_SUFFIXED.keys(),
  ...TENS.keys(),
  ...TENS_SUFFIXED.keys(),
  ...MULTIPLIERS.keys(),
  ...MULTIPLIERS_SUFFIXED.keys(),
  ...PRECEDING_PREFIXERS.keys(),
  ...FOLLOWING_PREFIXERS.keys(),
  ...SUFFIXERS.keys(),
  ...SPECIALS,
]);
const NUMERIC = py('^\\d+(\\.\\d+)?$', 'u');
const FRACTION = py(
  '^\\s*([-+]?)(?=\\d|\\.\\d)(\\d*|\\d+(?:_\\d+)*)(?:(?:\\s*/\\s*(\\d+(?:_\\d+)*))?|(?:\\.(\\d*|\\d+(?:_\\d+)*))?(?:[eE]([-+]?\\d+(?:_\\d+)*))?)\\s*$',
  'u',
);
const gcd = (a: bigint, b: bigint): bigint => (b ? gcd(b, a % b) : a < 0n ? -a : a);
// Python fractions.Fraction(str | int); null where Fraction raises ValueError.
function toFraction(value: string | bigint): { num: bigint; den: bigint } | null {
  if (typeof value === 'bigint') return { num: value, den: 1n };
  const m = FRACTION.exec(value);
  if (!m) return null;
  const digits = (raw: string) => BigInt(asciiDigits(raw.replaceAll('_', '')));
  let num = digits(m[2] || '0');
  let den = 1n;
  if (m[3]) den = digits(m[3]);
  else {
    if (m[4]) {
      const decimal = asciiDigits(m[4].replaceAll('_', ''));
      num = num * 10n ** BigInt(decimal.length) + BigInt(decimal);
      den *= 10n ** BigInt(decimal.length);
    }
    if (m[5]) {
      const exp = BigInt(asciiDigits(m[5].replaceAll('_', '')));
      if (exp >= 0n) num *= 10n ** exp;
      else den *= 10n ** -exp;
    }
  }
  if (m[1] === '-') num = -num;
  if (den === 0n) throw new Error(`Fraction(${num}, 0)`);
  const g = gcd(num, den);
  return { num: num / g, den: den / g };
}

function* processNumberWords(words: string[]): Generator<string> {
  let prefix: string | null = null;
  let value: bigint | string | null = null;
  let skip = false;
  const output = (result: bigint | string) => {
    let text = String(result);
    if (prefix !== null) text = prefix + text;
    value = null;
    prefix = null;
    return text;
  };
  const orEmpty = (v: bigint | string | null) => (v === null || v === 0n || v === '' ? '' : String(v));
  const scaled = (v: bigint | string, multiplier: bigint) => {
    const f = toFraction(v);
    return f && (f.num * multiplier) % f.den === 0n ? (f.num * multiplier) / f.den : null;
  };
  if (!words.length) return;
  const padded = [null, ...words, null];
  for (let i = 1; i <= words.length; i++) {
    const [prev, current, next] = [padded[i - 1], padded[i]!, padded[i + 1]];
    if (skip) {
      skip = false;
      continue;
    }
    const nextIsNumeric = next !== null && NUMERIC.test(next);
    const hasPrefix = PREFIXES.has(current[0]);
    const currentWithoutPrefix = hasPrefix ? current.slice(1) : current;
    if (NUMERIC.test(currentWithoutPrefix)) {
      const f = toFraction(currentWithoutPrefix)!;
      if (value !== null) {
        if (typeof value === 'string' && value.endsWith('.')) {
          value = value + current;
          continue;
        } else yield output(value);
      }
      prefix = hasPrefix ? current[0] : prefix;
      value = f.den === 1n ? f.num : currentWithoutPrefix;
    } else if (!WORDS.has(current)) {
      if (value !== null) yield output(value);
      yield output(current);
    } else if (ZEROS.has(current)) {
      value = orEmpty(value) + '0';
    } else if (ONES.has(current)) {
      const ones = ONES.get(current)!;
      if (value === null) value = ones;
      else if (typeof value === 'string' || (prev !== null && ONES.has(prev))) {
        if (prev !== null && TENS.has(prev) && ones < 10n) {
          const text: string = String(value);
          if (!text.endsWith('0')) throw new Error(`AssertionError: ${text}`);
          value = text.slice(0, -1) + String(ones);
        } else value = String(value) + String(ones);
      } else if (ones < 10n) {
        if (value % 10n === 0n) value += ones;
        else value = String(value) + String(ones);
      } else if (value % 100n === 0n) value += ones;
      else value = String(value) + String(ones);
    } else if (ONES_SUFFIXED.has(current)) {
      const [ones, suffix] = ONES_SUFFIXED.get(current)!;
      if (value === null) yield output(String(ones) + suffix);
      else if (typeof value === 'string' || (prev !== null && ONES.has(prev))) {
        if (prev !== null && TENS.has(prev) && ones < 10n) {
          const text = String(value);
          if (!text.endsWith('0')) throw new Error(`AssertionError: ${text}`);
          yield output(text.slice(0, -1) + String(ones) + suffix);
        } else yield output(String(value) + String(ones) + suffix);
      } else if (ones < 10n) {
        if (value % 10n === 0n) yield output(String(value + ones) + suffix);
        else yield output(String(value) + String(ones) + suffix);
      } else if (value % 100n === 0n) yield output(String(value + ones) + suffix);
      else yield output(String(value) + String(ones) + suffix);
      value = null;
    } else if (TENS.has(current)) {
      const tens = TENS.get(current)!;
      if (value === null) value = tens;
      else if (typeof value === 'string') value = value + String(tens);
      else if (value % 100n === 0n) value += tens;
      else value = String(value) + String(tens);
    } else if (TENS_SUFFIXED.has(current)) {
      const [tens, suffix] = TENS_SUFFIXED.get(current)!;
      if (value === null) yield output(String(tens) + suffix);
      else if (typeof value === 'string') yield output(value + String(tens) + suffix);
      else if (value % 100n === 0n) yield output(String(value + tens) + suffix);
      else yield output(String(value) + String(tens) + suffix);
    } else if (MULTIPLIERS.has(current)) {
      const multiplier = MULTIPLIERS.get(current)!;
      if (value === null) value = multiplier;
      else if (typeof value === 'string' || value === 0n) {
        const product = scaled(value, multiplier);
        if (product !== null) value = product;
        else {
          yield output(value);
          value = multiplier;
        }
      } else value = (value / 1000n) * 1000n + (value % 1000n) * multiplier;
    } else if (MULTIPLIERS_SUFFIXED.has(current)) {
      const [multiplier, suffix] = MULTIPLIERS_SUFFIXED.get(current)!;
      if (value === null) yield output(String(multiplier) + suffix);
      else if (typeof value === 'string') {
        const product = scaled(value, multiplier);
        if (product !== null) yield output(String(product) + suffix);
        else {
          yield output(value);
          yield output(String(multiplier) + suffix);
        }
      } else {
        value = (value / 1000n) * 1000n + (value % 1000n) * multiplier;
        yield output(String(value) + suffix);
      }
      value = null;
    } else if (PRECEDING_PREFIXERS.has(current)) {
      if (value !== null) yield output(value);
      if ((next !== null && WORDS.has(next)) || nextIsNumeric) prefix = PRECEDING_PREFIXERS.get(current)!;
      else yield output(current);
    } else if (FOLLOWING_PREFIXERS.has(current)) {
      if (value !== null) {
        prefix = FOLLOWING_PREFIXERS.get(current)!;
        yield output(value);
      } else yield output(current);
    } else if (SUFFIXERS.has(current)) {
      if (value !== null) {
        const suffix = SUFFIXERS.get(current)!;
        if (typeof suffix !== 'string') {
          if (next !== null && suffix.has(next)) {
            yield output(String(value) + suffix.get(next)!);
            skip = true;
          } else {
            yield output(value);
            yield output(current);
          }
        } else yield output(String(value) + suffix);
      } else yield output(current);
    } else if (SPECIALS.has(current)) {
      if (!(next !== null && WORDS.has(next)) && !nextIsNumeric) {
        if (value !== null) yield output(value);
        yield output(current);
      } else if (current === 'and') {
        if (!(prev !== null && MULTIPLIERS.has(prev))) {
          if (value !== null) yield output(value);
          yield output(current);
        }
      } else if (current === 'double' || current === 'triple') {
        if (next !== null && (ONES.has(next) || ZEROS.has(next))) {
          const repeats = current === 'double' ? 2 : 3;
          value = orEmpty(value) + String(ONES.get(next) ?? 0n).repeat(repeats);
          skip = true;
        } else {
          if (value !== null) yield output(value);
          yield output(current);
        }
      } else if (next !== null && (DECIMALS.has(next) || nextIsNumeric)) value = orEmpty(value) + '.';
    } else throw new Error(`Unexpected token: ${current}`);
  }
  if (value !== null) yield output(value);
}

const AND_A_HALF = py('\\band\\s+a\\s+half\\b');
const LETTER_DIGIT = py('([a-z])([0-9])');
const DIGIT_LETTER = py('([0-9])([a-z])');
const DIGIT_SUFFIX = py('([0-9])\\s+(st|nd|rd|th|s)\\b');
const CENTS_COMBINE = py('([€£$])([0-9]+) (?:and )?¢([0-9]{1,2})\\b');
const CENTS_EXTRACT = py('[€£$]0[^\\n]([0-9]{1,2})\\b');
const LITERAL_ONE = py('\\b1(s?)\\b');
function standardizeNumbers(input: string): string {
  const segments = input.split(AND_A_HALF);
  const results: string[] = [];
  for (const [i, segment] of segments.entries()) {
    if (!pyStrip(segment)) continue;
    results.push(segment);
    if (i === segments.length - 1) continue;
    const lastWord = pySplit(segment).at(-1)!;
    results.push(DECIMALS.has(lastWord) || MULTIPLIERS.has(lastWord) ? 'point five' : 'and a half');
  }
  let s = results.join(' ').replace(LETTER_DIGIT, '$1 $2').replace(DIGIT_LETTER, '$1 $2').replace(DIGIT_SUFFIX, '$1$2');
  s = [...processNumberWords(pySplit(s))].join(' ');
  s = s.replace(
    CENTS_COMBINE,
    (_match, currency: string, integer: string, cents: string) =>
      `${currency}${integer}.${String(Number(cents)).padStart(2, '0')}`,
  );
  s = s.replace(CENTS_EXTRACT, (_match, cents: string) => `¢${Number(cents)}`);
  return s.replace(LITERAL_ONE, 'one$1');
}

// whisper/normalizers/english.py:465-550 EnglishTextNormalizer
const BRACKETS = py('[<\\[][^>\\]]*[>\\]]');
const PARENS = py('\\(([^)]+?)\\)');
const FILLERS = py('\\b(hmm|mm|mhm|mmm|uh|um)\\b');
const SPACE_APOSTROPHE = py("\\s+'");
const REPLACERS: [RegExp, string][] = (
  [
    ["\\bwon't\\b", 'will not'],
    ["\\bcan't\\b", 'can not'],
    ["\\blet's\\b", 'let us'],
    ["\\bain't\\b", 'aint'],
    ["\\by'all\\b", 'you all'],
    ['\\bwanna\\b', 'want to'],
    ['\\bgotta\\b', 'got to'],
    ['\\bgonna\\b', 'going to'],
    ["\\bi'ma\\b", 'i am going to'],
    ['\\bimma\\b', 'i am going to'],
    ['\\bwoulda\\b', 'would have'],
    ['\\bcoulda\\b', 'could have'],
    ['\\bshoulda\\b', 'should have'],
    ["\\bma'am\\b", 'madam'],
    ['\\bmr\\b', 'mister '],
    ['\\bmrs\\b', 'missus '],
    ['\\bst\\b', 'saint '],
    ['\\bdr\\b', 'doctor '],
    ['\\bprof\\b', 'professor '],
    ['\\bcapt\\b', 'captain '],
    ['\\bgov\\b', 'governor '],
    ['\\bald\\b', 'alderman '],
    ['\\bgen\\b', 'general '],
    ['\\bsen\\b', 'senator '],
    ['\\brep\\b', 'representative '],
    ['\\bpres\\b', 'president '],
    ['\\brev\\b', 'reverend '],
    ['\\bhon\\b', 'honorable '],
    ['\\basst\\b', 'assistant '],
    ['\\bassoc\\b', 'associate '],
    ['\\blt\\b', 'lieutenant '],
    ['\\bcol\\b', 'colonel '],
    ['\\bjr\\b', 'junior '],
    ['\\bsr\\b', 'senior '],
    ['\\besq\\b', 'esquire '],
    ["'d been\\b", ' had been'],
    ["'s been\\b", ' has been'],
    ["'d gone\\b", ' had gone'],
    ["'s gone\\b", ' has gone'],
    ["'d done\\b", ' had done'],
    ["'s got\\b", ' has got'],
    ["n't\\b", ' not'],
    ["'re\\b", ' are'],
    ["'s\\b", ' is'],
    ["'d\\b", ' would'],
    ["'ll\\b", ' will'],
    ["'t\\b", ' not'],
    ["'ve\\b", ' have'],
    ["'m\\b", ' am'],
  ] as const
).map(([pattern, replacement]): [RegExp, string] => [py(pattern), replacement]);
const DIGIT_COMMA = py('(\\d),(\\d)');
const PERIOD = py('\\.([^0-9]|$)');
const SYMBOL_PREFIX = py('[.$¢€£]([^0-9])');
const SYMBOL_SUFFIX = py('([^0-9])%');
const SPACES = py('\\s+');
function normalizeEnglish(input: string, spelling: Map<string, string>): string {
  let s = input.toLowerCase().replace(BRACKETS, '').replace(PARENS, '').replace(FILLERS, '');
  s = s.replace(SPACE_APOSTROPHE, "'");
  for (const [pattern, replacement] of REPLACERS) s = s.replace(pattern, replacement);
  s = s.replace(DIGIT_COMMA, '$1$2').replace(PERIOD, ' $1');
  s = standardizeNumbers(removeSymbolsAndDiacritics(s, '.%$¢€£'));
  s = pySplit(s)
    .map((word) => spelling.get(word) ?? word)
    .join(' ');
  return s.replace(SYMBOL_PREFIX, ' $1').replace(SYMBOL_SUFFIX, '$1 ').replace(SPACES, ' ');
}

// zhon 2.1.1 hanzi.punctuation (non_stops + stops) and Python string.punctuation.
const ZH_PUNCTUATION =
  '＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠' +
  '［＼］＾＿｀｛｜｝～｟｠｢｣､　、〃〈' +
  '〉《》「」『』【】〔〕〖〗〘〙〚〛〜〝' +
  '〞〟〰〾〿–—‘’‛“”„‟…‧﹏﹑﹔' +
  '·．！？｡。';
const ASCII_PUNCTUATION = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~';
function normalizeChinese(input: string): string {
  let s = input;
  for (const ch of ZH_PUNCTUATION + ASCII_PUNCTUATION) if (ch !== "'") s = s.replaceAll(ch, '');
  return Array.from(pyStrip(s.replaceAll(' ', '').replaceAll('　', ''))).join(' ');
}

// jiwer wer_default (RemoveMultipleSpaces, Strip, ReduceToListOfListOfWords) and rapidfuzz Levenshtein.editops.
const MULTI_SPACE = py('\\s\\s+');
const jiwerWords = (s: string) =>
  pyStrip(s.replace(MULTI_SPACE, ' '))
    .split(' ')
    .filter((word) => word.length >= 1);
function editCounts(ref: string[], hyp: string[]) {
  let prefix = 0;
  while (prefix < ref.length && prefix < hyp.length && ref[prefix] === hyp[prefix]) prefix++;
  let suffix = 0;
  while (
    suffix < ref.length - prefix &&
    suffix < hyp.length - prefix &&
    ref[ref.length - 1 - suffix] === hyp[hyp.length - 1 - suffix]
  )
    suffix++;
  const a = ref.slice(prefix, ref.length - suffix);
  const b = hyp.slice(prefix, hyp.length - suffix);
  const counts = { substitutions: 0, deletions: 0, insertions: 0 };
  if (!a.length) {
    counts.insertions = b.length;
    return counts;
  }
  let vp = (1n << BigInt(a.length)) - 1n;
  let vn = 0n;
  let dist = a.length;
  const mask = 1n << BigInt(a.length - 1);
  const block = new Map<string, bigint>();
  let bit = 1n;
  for (const word of a) {
    block.set(word, (block.get(word) ?? 0n) | bit);
    bit <<= 1n;
  }
  const columnsVp: bigint[] = [];
  const columnsVn: bigint[] = [];
  for (const word of b) {
    const x = block.get(word) ?? 0n;
    const d0 = (((x & vp) + vp) ^ vp) | x | vn;
    let hp = vn | ~(d0 | vp);
    let hn = d0 & vp;
    if (hp & mask) dist++;
    if (hn & mask) dist--;
    hp = (hp << 1n) | 1n;
    hn <<= 1n;
    vp = hn | ~(d0 | hp);
    vn = hp & d0;
    columnsVp.push(vp);
    columnsVn.push(vn);
  }
  if (dist === 0) return counts;
  let col = a.length;
  let row = b.length;
  while (row !== 0 && col !== 0) {
    if (columnsVp[row - 1] & (1n << BigInt(col - 1))) {
      counts.deletions++;
      col--;
    } else {
      row--;
      if (row && columnsVn[row - 1] & (1n << BigInt(col - 1))) counts.insertions++;
      else {
        col--;
        if (a[col] !== b[row]) counts.substitutions++;
      }
    }
  }
  counts.deletions += col;
  counts.insertions += row;
  return counts;
}

interface Score {
  success: boolean;
  error: string | null;
  refNorm: string;
  hypNorm: string;
  wer: number;
  hits: number;
  substitutions: number;
  deletions: number;
  insertions: number;
}
function scoreRow(row: Row, spelling: Map<string, string>): Score {
  const normalize = (s: string) => (row.lang === 'zh' ? normalizeChinese(s) : normalizeEnglish(s, spelling));
  const empty = { refNorm: '', hypNorm: '', wer: 0, hits: 0, substitutions: 0, deletions: 0, insertions: 0 };
  if (row.error) return { ...empty, success: false, error: row.error };
  if (row.asrText === null) return { ...empty, success: false, error: 'No transcription' };
  const refNorm = normalize(row.text);
  const hypNorm = normalize(row.asrText);
  if (!refNorm) return { ...empty, refNorm, hypNorm, success: false, error: 'Empty reference after normalization' };
  const ref = jiwerWords(refNorm);
  const counts = editCounts(ref, jiwerWords(hypNorm));
  const hits = ref.length - counts.substitutions - counts.deletions;
  const errors = counts.substitutions + counts.deletions + counts.insertions;
  const wer = ref.length ? errors / ref.length : counts.insertions;
  return { success: true, error: null, refNorm, hypNorm, wer, hits, ...counts };
}

// benchmarks/metrics/wer.py:40-115 (std and latency fields left out).
function werMetrics(scores: Score[]) {
  const ok = scores.filter((score) => score.success);
  const errorsOf = (list: Score[]) => list.reduce((n, s) => n + s.substitutions + s.deletions + s.insertions, 0);
  const refWordsOf = (list: Score[]) => list.reduce((n, s) => n + s.substitutions + s.deletions + s.hits, 0);
  const sorted = ok.map((score) => score.wer).sort((x, y) => x - y);
  const mean = (list: number[]) => (list.length ? list.reduce((x, y) => x + y, 0) / list.length : 0);
  const n = sorted.length;
  const percentile = (q: number) => {
    const index = (n - 1) * q;
    const lo = Math.floor(index);
    const hi = Math.min(n - 1, lo + 1);
    const t = index - lo;
    const diff = sorted[hi] - sorted[lo];
    return t >= 0.5 ? sorted[hi] - diff * (1 - t) : sorted[lo] + diff * t;
  };
  const below = ok.filter((score) => score.wer <= 0.5);
  return {
    total: scores.length,
    evaluated: n,
    skipped: scores.length - n,
    errors: errorsOf(ok),
    refWords: refWordsOf(ok),
    werCorpus: refWordsOf(ok) ? errorsOf(ok) / refWordsOf(ok) : 0,
    werPerSampleMean: mean(sorted),
    werPerSampleMedian: n ? (n % 2 ? sorted[(n - 1) / 2] : (sorted[n / 2 - 1] + sorted[n / 2]) / 2) : 0,
    werPerSampleP95: n ? percentile(0.95) : 0,
    werPerSampleMax: n ? sorted[n - 1] : 0,
    werBelow50Corpus: refWordsOf(below) ? errorsOf(below) / refWordsOf(below) : 0,
    nAbove50: n - below.length,
  };
}

interface Loaded {
  dir: string;
  set: TextSet;
  seeds: number[];
  rows: Map<number, Map<string, Row[]>>;
}
function load(dir: string): Loaded {
  const set = JSON.parse(readFileSync(join(dir, 'texts.json'), 'utf8')) as TextSet;
  const seeds = readdirSync(dir)
    .map((name) => /^seed-(\d+)$/.exec(name)?.[1])
    .filter((seed): seed is string => seed !== undefined)
    .map(Number)
    .sort((x, y) => x - y);
  if (!seeds.length) throw new Error(`${dir} has no seed-<n> directories`);
  const rows = new Map<number, Map<string, Row[]>>();
  for (const seed of seeds) {
    const cells = new Map<string, Row[]>();
    for (const voice of set.voices) {
      const path = join(dir, `seed-${seed}`, `${voice}.jsonl`);
      if (!existsSync(path)) throw new Error(`Missing ${path}`);
      const cell = readJsonl<Row>(path);
      const expected = set.texts.map((item) => itemId(voice, item));
      if (cell.length !== expected.length || cell.some((row, i) => row.id !== expected[i] || row.seed !== seed))
        throw new Error(
          `${path}: ${cell.length} rows, expected ${expected.length} in texts.json order for seed ${seed}`,
        );
      cells.set(voice, cell);
    }
    rows.set(seed, cells);
  }
  return { dir, set, seeds, rows };
}
const GROUPS = ['cv-en', 'cv-zh', 'base-en', 'base-zh'] as const;
const METRICS = [
  ['werCorpus', 'wer corpus'],
  ['werBelow50Corpus', 'wer excl >50%'],
  ['nAbove50', '>50% rows'],
] as const;
const groupOf = (row: Row) => `${row.voice === 'cv' ? 'cv' : 'base'}-${row.lang}`;
function score(run: Loaded, spelling: Map<string, string>) {
  const scores: (Score & { seed: number; id: string; group: string })[] = [];
  const groups: Record<string, Record<number, ReturnType<typeof werMetrics>>> = {};
  const cells: Record<number, Record<string, { rows: number; failed: number; length: number; rtf: number }>> = {};
  for (const seed of run.seeds) {
    const bySeed = new Map<string, Score[]>();
    cells[seed] = {};
    for (const [voice, rows] of run.rows.get(seed)!) {
      for (const row of rows) {
        const result = scoreRow(row, spelling);
        scores.push({ seed, id: row.id, group: groupOf(row), ...result });
        bySeed.set(groupOf(row), [...(bySeed.get(groupOf(row)) ?? []), result]);
      }
      const audio = rows.reduce((sum, row) => sum + row.audioSeconds, 0);
      cells[seed][voice] = {
        rows: rows.length,
        failed: rows.filter((row) => row.error).length,
        length: rows.filter((row) => row.finishReason === 'length').length,
        rtf: audio ? rows.reduce((sum, row) => sum + row.synthesisMs, 0) / 1000 / audio : 0,
      };
    }
    for (const group of GROUPS) {
      const list = bySeed.get(group);
      if (!list?.length) continue;
      (groups[group] ??= {})[seed] = werMetrics(list);
    }
  }
  return { scores, groups, cells };
}

function report() {
  if (!values.out || !values['whisper-json']) throw new Error(usage);
  const raw = readFileSync(resolve(values['whisper-json']));
  if (sha256(raw) !== WHISPER_JSON_SHA256)
    throw new Error(`${values['whisper-json']} sha256 ${sha256(raw)} != openai-whisper 20250625 english.json`);
  const spelling = new Map(Object.entries(JSON.parse(raw.toString('utf8')) as Record<string, string>));
  const candidate = load(resolve(values.out));
  const baseline = values.baseline ? load(resolve(values.baseline)) : null;
  if (baseline) {
    if (baseline.dir === candidate.dir) throw new Error(`Refusing to compare ${candidate.dir} with itself`);
    if (baseline.set.digest !== candidate.set.digest)
      throw new Error(`Text sets differ: baseline ${baseline.set.digest}, candidate ${candidate.set.digest}`);
    if (baseline.seeds.join(',') !== candidate.seeds.join(','))
      throw new Error(`Seeds differ: baseline ${baseline.seeds.join(',')}, candidate ${candidate.seeds.join(',')}`);
    for (const seed of candidate.seeds)
      for (const voice of candidate.set.voices) {
        const [left, right] = [baseline, candidate].map((run) => {
          const path = join(run.dir, `seed-${seed}`, `${voice}.meta.json`);
          const meta = JSON.parse(readFileSync(path, 'utf8')) as { model: string; asrModel: string };
          return `tts ${meta.model}, asr ${meta.asrModel}`;
        });
        if (left !== right) throw new Error(`seed ${seed} ${voice}: baseline ${left}, candidate ${right}`);
      }
  }
  const scored = score(candidate, spelling);
  writeFileSync(join(candidate.dir, 'scores.jsonl'), scored.scores.map((row) => JSON.stringify(row)).join('\n') + '\n');
  const fmt = (x: number) => (Number.isInteger(x) ? String(x) : x.toFixed(6));
  const lines: string[] = [];
  const range = (list: number[]) => ({
    min: Math.min(...list),
    max: Math.max(...list),
    mean: list.reduce((x, y) => x + y, 0) / list.length,
  });
  const summary: Record<string, Record<string, { perSeed: number[]; min: number; max: number; mean: number }>> = {};
  for (const [metric, label] of METRICS) {
    lines.push(
      [label.padEnd(14), 'items', ...candidate.seeds.map((seed) => `seed ${seed}`), 'min', 'max', 'mean'].join('\t'),
    );
    for (const group of GROUPS) {
      const perSeed = scored.groups[group];
      if (!perSeed) continue;
      const list = candidate.seeds.map((seed) => perSeed[seed][metric]);
      const stats = range(list);
      (summary[group] ??= {})[metric] = { perSeed: list, ...stats };
      const items = perSeed[candidate.seeds[0]];
      lines.push(
        [
          group.padEnd(14),
          `${items.evaluated}/${items.total}`,
          ...list.map(fmt),
          fmt(stats.min),
          fmt(stats.max),
          fmt(stats.mean),
        ].join('\t'),
      );
    }
  }
  for (const seed of candidate.seeds)
    for (const [voice, cell] of Object.entries(scored.cells[seed]))
      lines.push(
        `seed ${seed} ${voice}: rows ${cell.rows} failed ${cell.failed} length ${cell.length} rtf ${cell.rtf.toFixed(4)}`,
      );
  lines.push('similarity: not reported (the native TTS API exposes no speaker embedding)');

  let failed = false;
  let comparison: unknown = null;
  if (baseline) {
    const base = score(baseline, spelling);
    const verdicts: Record<string, Record<string, unknown>> = {};
    lines.push(`baseline ${baseline.dir} seeds ${baseline.seeds.join(',')}`);
    for (const group of GROUPS) {
      if (!summary[group] || !base.groups[group]) continue;
      const failedRows = candidate.seeds.reduce((n, seed) => n + scored.groups[group][seed].skipped, 0);
      for (const [metric, label] of METRICS) {
        const { min, max } = range(baseline.seeds.map((seed) => base.groups[group][seed][metric]));
        const value = summary[group][metric].mean;
        const status = value > max || failedRows ? 'FAIL' : value < min ? 'below' : 'in';
        if (status === 'FAIL') failed = true;
        (verdicts[group] ??= {})[metric] = { min, max, candidateMean: value, failedRows, status };
        lines.push(
          `${group} ${label}: candidate mean ${fmt(value)} vs baseline ${fmt(min)}..${fmt(max)}, failed rows ${failedRows} -> ${status}`,
        );
      }
    }
    const identity: Record<string, unknown> = {};
    for (const seed of candidate.seeds)
      for (const voice of candidate.set.voices) {
        const [left, right] = [baseline.rows.get(seed)!.get(voice)!, candidate.rows.get(seed)!.get(voice)!];
        const pcm = left.filter((row, i) => row.pcmSha256 === right[i].pcmSha256).length;
        const text = left.filter((row, i) => row.asrText === right[i].asrText).length;
        const groupsEqual = GROUPS.every(
          (group) =>
            !base.groups[group]?.[seed] ||
            JSON.stringify(base.groups[group][seed]) === JSON.stringify(scored.groups[group][seed]),
        );
        identity[`${seed}/${voice}`] = {
          rows: left.length,
          pcmEqual: pcm,
          asrTextEqual: text,
          groupMetricsEqual: groupsEqual,
        };
        lines.push(
          `identity seed ${seed} ${voice}: pcm ${pcm}/${left.length} asr text ${text}/${left.length} ` +
            `group metrics ${groupsEqual ? 'equal' : 'differ'}; rtf ${base.cells[seed][voice].rtf.toFixed(4)} -> ` +
            `${scored.cells[seed][voice].rtf.toFixed(4)}`,
        );
      }
    comparison = { baseline: baseline.dir, seeds: baseline.seeds, verdicts, identity, pass: !failed };
  }
  writeFileSync(
    join(candidate.dir, 'report.json'),
    JSON.stringify(
      {
        textsDigest: candidate.set.digest,
        seeds: candidate.seeds,
        metrics: 'per seed: wer_corpus, wer_below_50_corpus, n_above_50_pct_wer (sglang-omni); zh is character level',
        summary,
        groups: scored.groups,
        cells: scored.cells,
        similarity: { reported: false, reason: 'the native TTS API exposes no speaker embedding' },
        comparison,
      },
      null,
      1,
    ) + '\n',
  );
  console.log(lines.join('\n'));
  if (failed) process.exitCode = 1;
}

if (positionals[0] === 'run' && positionals.length === 1) await run();
else if (positionals[0] === 'report' && positionals.length === 1) report();
else throw new Error(usage);
