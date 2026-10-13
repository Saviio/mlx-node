//! Container/sample primitives shared by serving and TTS. File/socket ownership
//! stays with the caller; each decode call contains complete sample frames.
use napi::bindgen_prelude::*;
use napi_derive::napi;

#[napi(object)]
#[derive(Clone, Debug)]
pub struct AudioWavFormat {
    pub encoding: u32,
    pub sample_rate: u32,
    pub channels: u32,
    pub bits_per_sample: u32,
    pub block_align: u32,
}

fn invalid() -> Error {
    Error::from_reason("Unsupported or invalid WAV sample format")
}

fn validate(format: &AudioWavFormat) -> Result<()> {
    if format.sample_rate == 0
        || format.channels == 0
        || format.channels > 32
        || !matches!(
            (format.encoding, format.bits_per_sample),
            (1, 8 | 16 | 24 | 32) | (3, 32 | 64)
        )
        || format.block_align != format.channels * (format.bits_per_sample / 8)
    {
        return Err(invalid());
    }
    Ok(())
}

#[napi]
pub fn audio_wav_format(bytes: Buffer) -> Result<AudioWavFormat> {
    if bytes.len() < 16 {
        return Err(invalid());
    }
    let u16_at = |i| u16::from_le_bytes([bytes[i], bytes[i + 1]]) as u32;
    let u32_at = |i| u32::from_le_bytes([bytes[i], bytes[i + 1], bytes[i + 2], bytes[i + 3]]);
    let mut encoding = u16_at(0);
    if encoding == 0xfffe {
        if bytes.len() < 40
            || u16_at(16) < 22
            || bytes[28..40] != [0, 0, 16, 0, 128, 0, 0, 170, 0, 56, 155, 113]
        {
            return Err(invalid());
        }
        encoding = u32_at(24);
        let valid_bits = u16_at(18);
        let container_bits = u16_at(14);
        if valid_bits == 0
            || valid_bits > container_bits
            || (encoding != 1 && valid_bits != container_bits)
        {
            return Err(invalid());
        }
        // Extensible PCM may have lower precision than its container. Its
        // valid bits are left-aligned, so container-width decoding preserves
        // the amplitude without any additional shift or rescaling.
    }
    let format = AudioWavFormat {
        encoding,
        sample_rate: u32_at(4),
        channels: u16_at(2),
        bits_per_sample: u16_at(14),
        block_align: u16_at(12),
    };
    validate(&format)?;
    if u32_at(8) as u64 != format.sample_rate as u64 * format.block_align as u64 {
        return Err(invalid());
    }
    Ok(format)
}

fn decode(bytes: &[u8], format: &AudioWavFormat, mono: bool) -> Result<Vec<f32>> {
    validate(format)?;
    if !bytes.len().is_multiple_of(format.block_align as usize) {
        return Err(Error::from_reason("Incomplete WAV sample frame"));
    }
    let width = (format.bits_per_sample / 8) as usize;
    let channels = format.channels as usize;
    let mut output = Vec::with_capacity(
        bytes.len()
            / if mono {
                format.block_align as usize
            } else {
                width
            },
    );
    for frame in bytes.chunks_exact(format.block_align as usize) {
        let mut sum = 0.0f64;
        for sample in frame.chunks_exact(width) {
            let value = match (format.encoding, width) {
                (3, 4) => f32::from_le_bytes(sample.try_into().map_err(|_| invalid())?),
                (3, 8) => f64::from_le_bytes(sample.try_into().map_err(|_| invalid())?) as f32,
                (1, 1) => (sample[0] as f32 - 128.0) / 128.0,
                (1, 2) => {
                    i16::from_le_bytes(sample.try_into().map_err(|_| invalid())?) as f32 / 32768.0
                }
                (1, 3) => {
                    ((sample[0] as i32 | (sample[1] as i32) << 8 | (sample[2] as i32) << 16) << 8
                        >> 8) as f32
                        / 8388608.0
                }
                (1, 4) => {
                    i32::from_le_bytes(sample.try_into().map_err(|_| invalid())?) as f32
                        / 2147483648.0
                }
                _ => return Err(invalid()),
            };
            if !value.is_finite() {
                return Err(Error::from_reason("WAV contains non-finite samples"));
            }
            if mono {
                sum += value as f64;
            } else {
                output.push(value);
            }
        }
        if mono {
            output.push((sum / channels as f64) as f32);
        }
    }
    Ok(output)
}

#[napi]
pub fn audio_decode_wav_frames(
    bytes: Buffer,
    format: AudioWavFormat,
    mono: bool,
) -> Result<Float32Array> {
    Ok(decode(&bytes, &format, mono)?.into())
}

#[napi]
pub fn audio_encode_pcm16(samples: Float32Array) -> Result<Buffer> {
    let mut output = Vec::with_capacity(samples.len() * 2);
    for value in samples.iter() {
        if !value.is_finite() {
            return Err(Error::from_reason("Non-finite PCM sample"));
        }
        // Match JavaScript Math.round, including negative half-way values.
        let value = ((*value as f64 * 32768.0 + 0.5).floor()).clamp(-32768.0, 32767.0) as i16;
        output.extend_from_slice(&value.to_le_bytes());
    }
    Ok(output.into())
}

#[napi]
pub fn audio_wav_header(sample_rate: u32, channels: u32, data_bytes: u32) -> Result<Buffer> {
    if sample_rate == 0
        || channels == 0
        || channels > 32
        || data_bytes > u32::MAX - 36
        || !(data_bytes as u64).is_multiple_of(channels as u64 * 2)
        || sample_rate as u64 * channels as u64 * 2 > u32::MAX as u64
    {
        return Err(invalid());
    }
    let mut out = Vec::with_capacity(44);
    out.extend_from_slice(b"RIFF");
    out.extend_from_slice(&(data_bytes + 36).to_le_bytes());
    out.extend_from_slice(b"WAVEfmt ");
    out.extend_from_slice(&16u32.to_le_bytes());
    out.extend_from_slice(&1u16.to_le_bytes());
    out.extend_from_slice(&(channels as u16).to_le_bytes());
    out.extend_from_slice(&sample_rate.to_le_bytes());
    out.extend_from_slice(&(sample_rate * channels * 2).to_le_bytes());
    out.extend_from_slice(&(channels as u16 * 2).to_le_bytes());
    out.extend_from_slice(&16u16.to_le_bytes());
    out.extend_from_slice(b"data");
    out.extend_from_slice(&data_bytes.to_le_bytes());
    Ok(out.into())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn extensible_format(encoding: u32, bits: u16, valid_bits: u16) -> Buffer {
        let mut bytes = vec![0; 40];
        bytes[0..2].copy_from_slice(&0xfffeu16.to_le_bytes());
        bytes[2..4].copy_from_slice(&2u16.to_le_bytes());
        bytes[4..8].copy_from_slice(&48000u32.to_le_bytes());
        bytes[8..12].copy_from_slice(&(48000 * bits as u32 / 8 * 2).to_le_bytes());
        bytes[12..14].copy_from_slice(&(bits / 8 * 2).to_le_bytes());
        bytes[14..16].copy_from_slice(&bits.to_le_bytes());
        bytes[16..18].copy_from_slice(&22u16.to_le_bytes());
        bytes[18..20].copy_from_slice(&valid_bits.to_le_bytes());
        bytes[24..28].copy_from_slice(&encoding.to_le_bytes());
        bytes[28..40].copy_from_slice(&[0, 0, 16, 0, 128, 0, 0, 170, 0, 56, 155, 113]);
        bytes.into()
    }

    #[test]
    fn extensible_pcm_precision_preserves_amplitude() {
        for (bits, valid_bits) in [(24, 20), (32, 24)] {
            let format = audio_wav_format(extensible_format(1, bits, valid_bits)).unwrap();
            let samples: Vec<u8> = [0.5, -0.25, -1.0, 0.25]
                .into_iter()
                .flat_map(|v| {
                    let sample = (v * (1u64 << (bits - 1)) as f64) as i32;
                    sample.to_le_bytes()[..bits as usize / 8].to_vec()
                })
                .collect();
            assert_eq!(
                decode(&samples, &format, false).unwrap(),
                vec![0.5, -0.25, -1.0, 0.25]
            );
            assert_eq!(
                decode(&samples, &format, true).unwrap(),
                vec![0.125, -0.375]
            );
        }
    }

    #[test]
    fn extensible_precision_rejects_invalid_formats() {
        for (encoding, bits, valid_bits) in [(1, 32, 0), (1, 24, 25), (3, 32, 24), (3, 64, 32)] {
            assert!(audio_wav_format(extensible_format(encoding, bits, valid_bits)).is_err());
        }
        for bits in [32, 64] {
            assert!(audio_wav_format(extensible_format(3, bits, bits)).is_ok());
        }
    }

    #[test]
    fn pcm_formats_and_downmix() {
        for rate in [16000, 22050, 24000, 44100, 48000, 96000] {
            let f = AudioWavFormat {
                encoding: 1,
                sample_rate: rate,
                channels: 2,
                bits_per_sample: 16,
                block_align: 4,
            };
            assert_eq!(decode(&[0, 128, 0, 64], &f, true).unwrap(), vec![-0.25]);
            assert!(decode(&[0], &f, true).is_err());
        }
        let f = AudioWavFormat {
            encoding: 3,
            sample_rate: 48000,
            channels: 1,
            bits_per_sample: 32,
            block_align: 4,
        };
        assert!(decode(&f32::NAN.to_le_bytes(), &f, false).is_err());
    }

    #[test]
    fn pcm16_wav_roundtrip_preserves_clipping_and_rounding() {
        let samples = vec![-2.0, -1.0, -0.5, -1.0 / 65536.0, 0.0, 0.5, 1.0, 2.0];
        let pcm = audio_encode_pcm16(samples.into()).unwrap();
        let header = audio_wav_header(44100, 2, pcm.len() as u32).unwrap();
        assert_eq!(header.len(), 44);
        assert_eq!(&header[..4], b"RIFF");
        assert_eq!(&header[8..16], b"WAVEfmt ");
        assert_eq!(&header[36..40], b"data");
        assert_eq!(&header[4..8], &(36 + pcm.len() as u32).to_le_bytes());
        assert_eq!(&header[40..44], &(pcm.len() as u32).to_le_bytes());
        let format = audio_wav_format(header[20..36].to_vec().into()).unwrap();
        assert_eq!(format.sample_rate, 44100);
        assert_eq!(format.channels, 2);
        let decoded = audio_decode_wav_frames(pcm, format, false).unwrap();
        assert_eq!(
            decoded.as_ref(),
            &[
                -1.0,
                -1.0,
                -0.5,
                0.0,
                0.0,
                0.5,
                32767.0 / 32768.0,
                32767.0 / 32768.0
            ]
        );
    }

    #[test]
    fn public_codecs_reject_invalid_samples_and_dimensions() {
        for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            assert!(audio_encode_pcm16(vec![value].into()).is_err());
        }
        for (rate, channels, data_bytes) in [
            (0, 1, 0),
            (24000, 0, 0),
            (24000, 33, 0),
            (24000, 2, 2),
            (24000, 1, u32::MAX),
            (u32::MAX, 1, 0),
        ] {
            assert!(audio_wav_header(rate, channels, data_bytes).is_err());
        }
        let format = AudioWavFormat {
            encoding: 1,
            sample_rate: 24000,
            channels: 2,
            bits_per_sample: 16,
            block_align: 4,
        };
        assert!(audio_decode_wav_frames(vec![0; 3].into(), format, true).is_err());
    }

    #[test]
    fn public_decoder_supports_float_widths_and_downmix() {
        for bits in [32, 64] {
            let format = AudioWavFormat {
                encoding: 3,
                sample_rate: 48000,
                channels: 2,
                bits_per_sample: bits,
                block_align: bits / 8 * 2,
            };
            let bytes: Vec<u8> = [-0.5, 0.25]
                .into_iter()
                .flat_map(|value: f64| {
                    if bits == 32 {
                        (value as f32).to_le_bytes().to_vec()
                    } else {
                        value.to_le_bytes().to_vec()
                    }
                })
                .collect();
            let decoded =
                audio_decode_wav_frames(bytes.clone().into(), format.clone(), false).unwrap();
            assert_eq!(decoded.as_ref(), &[-0.5, 0.25]);
            let mono = audio_decode_wav_frames(bytes.into(), format, true).unwrap();
            assert_eq!(mono.as_ref(), &[-0.125]);
        }
    }
}
