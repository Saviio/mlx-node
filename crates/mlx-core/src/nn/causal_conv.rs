//! Explicit state for causal NLC convolutions. State belongs to a stream, not weights.
use crate::array::MxArray;
use crate::nn::{Conv1d, conv_transpose1d::ConvTranspose1d};
use napi::{Error, Result};

// History arrays are immutable: each convolution step replaces the handle.
#[derive(Default, Clone)]
pub struct ConvState {
    pub history: Option<MxArray>,
}

/// Dense (`groups == 1`) causal convolutions run as one GEMM over the K
/// shifted input taps instead of MLX's convolution kernels, which are
/// several times slower on the decoder's short, wide activations (T of 8 to
/// a few thousand, hundreds of channels). The GEMM reduces in a different
/// order, so results agree with the convolution to float32 GEMM tolerance,
/// not bit for bit; depthwise convolutions keep the convolution kernel.
pub struct CausalConv1d {
    conv: Conv1d,
    /// `[K * I, O]` weight for the tap GEMM (dense convolutions only).
    gemm_weight: Option<MxArray>,
    bias: Option<MxArray>,
    kernel: i64,
    dilation: i64,
    history_len: i64,
}

impl CausalConv1d {
    pub fn new(
        weight: &MxArray,
        bias: Option<&MxArray>,
        dilation: u32,
        groups: u32,
    ) -> Result<Self> {
        let shape = weight.shape()?;
        if shape.len() != 3 || dilation == 0 || groups == 0 {
            return Err(Error::from_reason("Invalid causal convolution geometry"));
        }
        let (out_channels, kernel, in_channels) = (shape[0], shape[1], shape[2]);
        let gemm_weight = (groups == 1)
            .then(|| {
                weight
                    .transpose(Some(&[1, 2, 0]))?
                    .reshape(&[kernel * in_channels, out_channels])
            })
            .transpose()?;
        if let Some(gemm_weight) = &gemm_weight {
            gemm_weight.eval();
        }
        Ok(Self {
            conv: Conv1d::from_weights(
                weight,
                bias,
                Some(1),
                Some(0),
                Some(dilation),
                Some(groups),
            )?,
            gemm_weight,
            bias: bias.cloned(),
            kernel,
            dilation: i64::from(dilation),
            history_len: (kernel - 1) * i64::from(dilation),
        })
    }
    /// Trailing inputs that fix the last `outputs` outputs and the history
    /// bit for bit, whatever came before them (an empty state included).
    pub fn inputs_for(&self, outputs: i64) -> i64 {
        outputs + self.history_len
    }
    pub fn step(&self, input: &MxArray, state: &mut ConvState) -> Result<MxArray> {
        let shape = input.shape()?;
        let (batch, n, channels) = (shape[0], shape[1], shape[2]);
        if n == 0 {
            return Err(Error::from_reason(
                "Causal convolution requires nonempty input",
            ));
        }
        let x = match &state.history {
            Some(history) => MxArray::concatenate_many(vec![history, input], Some(1))?,
            None => input.pad(&[0, 0, self.history_len as i32, 0, 0, 0], 0.)?,
        };
        if self.history_len > 0 {
            state.history = Some(x.slice_axis(1, n, n + self.history_len)?.deep_copy()?);
        }
        let Some(gemm_weight) = &self.gemm_weight else {
            return self.conv.forward(&x);
        };
        // Tap j of output t reads x[t + j * dilation]; stacking the K shifted
        // views along the channel axis gives the `[n, K * I]` GEMM operand.
        let columns = if self.kernel == 1 {
            x.reshape(&[batch * n, channels])?
        } else {
            let taps = (0..self.kernel)
                .map(|tap| x.slice_axis(1, tap * self.dilation, tap * self.dilation + n))
                .collect::<Result<Vec<_>>>()?;
            MxArray::concatenate_many(taps.iter().collect(), Some(2))?
                .reshape(&[batch * n, self.kernel * channels])?
        };
        let out_channels = gemm_weight.shape()?[1];
        let output = columns
            .matmul(gemm_weight)?
            .reshape(&[batch, n, out_channels])?;
        match &self.bias {
            Some(bias) => output.add(bias),
            None => Ok(output),
        }
    }
}

/// Transposed convolutions with `kernel == 2 * stride` (every codec
/// upsampler) run as one GEMM plus an overlap-add of the two half-kernel
/// windows. MLX lowers transposed convolutions to its input-dilated fallback
/// kernel, which is the slowest path for these shapes; the GEMM agrees with
/// it to float32 rounding. Other geometries keep the convolution kernel.
pub struct CausalTransposeConv1d {
    conv: ConvTranspose1d,
    /// `[I, K * O]` weight for the GEMM formulation.
    gemm_weight: Option<MxArray>,
}

impl CausalTransposeConv1d {
    pub fn new(weight: MxArray, bias: Option<MxArray>, stride: i32) -> Result<Self> {
        let shape = weight.shape()?;
        let gemm_weight = (shape.len() == 3 && shape[1] == 2 * i64::from(stride))
            .then(|| {
                weight
                    .transpose(Some(&[2, 1, 0]))?
                    .reshape(&[shape[2], shape[1] * shape[0]])
            })
            .transpose()?;
        if let Some(gemm_weight) = &gemm_weight {
            gemm_weight.eval();
        }
        Ok(Self {
            conv: ConvTranspose1d::from_weights(weight, bias, stride)?,
            gemm_weight,
        })
    }
    /// Trailing inputs that fix the last `outputs` emitted samples and the
    /// overlap tail bit for bit, whatever came before them.
    pub fn inputs_for(&self, outputs: i64) -> i64 {
        (outputs + i64::from(self.conv.kernel) - 1) / i64::from(self.conv.stride)
    }
    /// Full-length (`(T - 1) * stride + kernel`) output without bias.
    fn full_output(&self, input: &MxArray) -> Result<MxArray> {
        let Some(gemm_weight) = &self.gemm_weight else {
            return self.conv.forward_without_bias(input);
        };
        let shape = input.shape()?;
        let (batch, frames, channels) = (shape[0], shape[1], shape[2]);
        let stride = i64::from(self.conv.stride);
        let out_channels = gemm_weight.shape()?[1] / (2 * stride);
        // y[t, h, j, o] is tap h * stride + j of frame t; half 0 lands on
        // output rows [t * stride, (t + 1) * stride) and half 1 one stride later.
        let y = input
            .reshape(&[batch * frames, channels])?
            .matmul(gemm_weight)?
            .reshape(&[batch, frames, 2, stride, out_channels])?;
        let emit = frames * stride;
        let first = y
            .slice_axis(2, 0, 1)?
            .reshape(&[batch, emit, out_channels])?
            .pad(&[0, 0, 0, stride as i32, 0, 0], 0.)?;
        let second = y
            .slice_axis(2, 1, 2)?
            .reshape(&[batch, emit, out_channels])?
            .pad(&[0, 0, stride as i32, 0, 0, 0], 0.)?;
        first.add(&second)
    }
    pub fn step(&self, input: &MxArray, state: &mut ConvState) -> Result<MxArray> {
        let emit = input.shape()?[1] * i64::from(self.conv.stride);
        let overlap = i64::from(self.conv.kernel - self.conv.stride);
        let mut output = self.full_output(input)?;
        if let Some(tail) = &state.history {
            let length = output.shape()?[1];
            output = output.add(&tail.pad(&[0, 0, 0, (length - overlap) as i32, 0, 0], 0.)?)?;
        }
        if overlap > 0 {
            state.history = Some(output.slice_axis(1, emit, emit + overlap)?.deep_copy()?);
        }
        // Bias is added only once, after overlap accumulation. Final overlap is the
        // causal right trim; it is discarded on end, never emitted as extra audio.
        self.conv.add_bias(output.slice_axis(1, 0, emit)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    type Step<'a> = &'a dyn Fn(&MxArray, &mut ConvState) -> Result<MxArray>;

    /// Output and state bits of one-input steps over `input[from..]`, from an empty state.
    fn run_from(step: Step, input: &MxArray, from: i64) -> (Vec<u32>, Option<Vec<u32>>) {
        let bits = |x: &MxArray| {
            x.to_float32()
                .unwrap()
                .iter()
                .map(|v| v.to_bits())
                .collect()
        };
        let mut state = ConvState::default();
        let mut out: Vec<u32> = vec![];
        for t in from..input.shape_at(1).unwrap() {
            out.extend(bits(
                &step(&input.slice_axis(1, t, t + 1).unwrap(), &mut state).unwrap(),
            ));
        }
        (out, state.history.as_ref().map(bits))
    }

    #[test]
    fn inputs_for_is_the_exact_trailing_context() {
        let values = |count: i64, seed: f32| -> Vec<f32> {
            (0..count)
                .map(|i| ((i as f32 + seed) * 0.37).sin() + 1.5)
                .collect()
        };
        let weight = |shape: &[i64], seed: f32| {
            MxArray::from_float32(&values(shape.iter().product(), seed), shape).unwrap()
        };
        let (channels, frames) = (3, 24);
        let input = weight(&[1, frames, channels], 0.5);
        let causal = CausalConv1d::new(&weight(&[2, 5, channels], 1.), None, 3, 1).unwrap();
        let depthwise =
            CausalConv1d::new(&weight(&[channels, 7, 1], 2.), None, 1, channels as u32).unwrap();
        // kernel == 2 * stride takes the GEMM path; kernel == stride the conv kernel.
        let overlapped =
            CausalTransposeConv1d::new(weight(&[2, 6, channels], 3.), None, 3).unwrap();
        let plain = CausalTransposeConv1d::new(weight(&[2, 2, channels], 4.), None, 2).unwrap();
        let layers: [(Step, &dyn Fn(i64) -> i64, i64, i64); 4] = [
            (&|x, s| causal.step(x, s), &|n| causal.inputs_for(n), 1, 2),
            (
                &|x, s| depthwise.step(x, s),
                &|n| depthwise.inputs_for(n),
                1,
                channels,
            ),
            (
                &|x, s| overlapped.step(x, s),
                &|n| overlapped.inputs_for(n),
                3,
                2,
            ),
            (&|x, s| plain.step(x, s), &|n| plain.inputs_for(n), 2, 2),
        ];
        for (layer, (step, inputs_for, per_input, width)) in layers.iter().enumerate() {
            let (full, full_state) = run_from(*step, &input, 0);
            for samples in [0, 1, 4, 7] {
                let needed = inputs_for(samples);
                let tail = |start: i64| {
                    let (out, state) = run_from(*step, &input, start);
                    let keep = (samples * width) as usize;
                    (out[out.len().saturating_sub(keep)..].to_vec(), state)
                };
                let expected = (
                    full[full.len() - (samples * width) as usize..].to_vec(),
                    full_state.clone(),
                );
                assert!(needed * per_input >= samples);
                assert_eq!(
                    tail(frames - needed),
                    expected,
                    "layer {layer}, {samples} samples"
                );
                if needed > 0 {
                    assert_ne!(
                        tail(frames - needed + 1),
                        expected,
                        "layer {layer}, {samples} samples: one input fewer must change bits"
                    );
                }
            }
        }
    }

    #[test]
    fn chunking_preserves_dilation_overlap_and_bias() {
        let input = MxArray::from_float32(&[1., -2., 3., 4., -5., 6., 7.], &[1, 7, 1]).unwrap();
        let weight = MxArray::from_float32(&[0.1, 0.2, 0.3, 0.4, 0.5], &[1, 5, 1]).unwrap();
        let bias = MxArray::from_float32(&[1.25], &[1]).unwrap();
        let causal = CausalConv1d::new(&weight, Some(&bias), 3, 1).unwrap();
        // Kernel larger than twice the stride exercises a tail that spans
        // several short input chunks, with a nonzero bias added exactly once.
        let transpose = CausalTransposeConv1d::new(weight, Some(bias), 2).unwrap();
        for transposed in [false, true] {
            let apply = |x: &MxArray, state: &mut ConvState| {
                if transposed {
                    transpose.step(x, state)
                } else {
                    causal.step(x, state)
                }
                .unwrap()
            };
            let expected = apply(&input, &mut ConvState::default())
                .to_float32()
                .unwrap();
            for chunks in [vec![1; 7], vec![2, 1, 4], vec![7]] {
                let mut state = ConvState::default();
                let mut at = 0;
                let mut actual = vec![];
                for n in chunks {
                    let out = apply(&input.slice_axis(1, at, at + n).unwrap(), &mut state);
                    actual.extend_from_slice(&out.to_float32().unwrap());
                    at += n;
                }
                assert_eq!(actual.len(), expected.len());
                for (a, b) in actual.iter().zip(expected.iter()) {
                    assert!((a - b).abs() < 1e-5);
                }
            }
        }
    }
}
