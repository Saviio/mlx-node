use super::{config::TransformerConfig, weights::Weights};
use crate::{
    array::{DType, MxArray, scaled_dot_product_attention},
    nn::{Activations, LayerNorm, Linear, RMSNorm, RoPE},
    transformer::KVCache,
};
use napi::{Error, Result};

#[derive(Default)]
pub struct AttentionState {
    pub kv: KVCache,
    pub position: usize,
}

enum Norm {
    Rms(RMSNorm),
    Layer(LayerNorm),
}
impl Norm {
    fn forward(&self, x: &MxArray) -> Result<MxArray> {
        match self {
            Self::Rms(n) => n.forward(x),
            Self::Layer(n) => n.forward(x),
        }
    }
}

pub struct DecoderLayer {
    q: Linear,
    k: Linear,
    v: Linear,
    o: Linear,
    q_norm: Option<RMSNorm>,
    k_norm: Option<RMSNorm>,
    norm1: Norm,
    norm2: Norm,
    gate: Option<Linear>,
    up: Linear,
    down: Linear,
    attn_scale: Option<MxArray>,
    mlp_scale: Option<MxArray>,
    config: TransformerConfig,
    /// Effective attention policy can differ from checkpoint metadata: the
    /// offline Mimi encoder supplies a full causal mask explicitly.
    attention_window: Option<usize>,
    rope: RoPE,
}
impl DecoderLayer {
    pub fn load(
        w: &Weights,
        prefix: &str,
        c: &TransformerConfig,
        qk_norm: bool,
        layer_scale: bool,
    ) -> Result<Self> {
        c.validate()?;
        if c.hidden_act != "silu" {
            return Err(Error::from_reason("Unsupported TTS decoder activation"));
        }
        let attn = format!("{prefix}.self_attn");
        for (name, out, input) in [
            ("q_proj", c.num_attention_heads * c.head_dim, c.hidden_size),
            ("k_proj", c.num_key_value_heads * c.head_dim, c.hidden_size),
            ("v_proj", c.num_key_value_heads * c.head_dim, c.hidden_size),
            ("o_proj", c.hidden_size, c.num_attention_heads * c.head_dim),
        ] {
            w.expect_linear(&format!("{attn}.{name}"), out, input)?;
            if c.attention_bias {
                w.expect_shape(&format!("{attn}.{name}.bias"), &[out])?;
            }
        }
        for name in ["input_layernorm", "post_attention_layernorm"] {
            w.expect_shape(&format!("{prefix}.{name}.weight"), &[c.hidden_size])?;
        }
        for name in ["gate_proj", "up_proj"] {
            w.expect_linear(
                &format!("{prefix}.mlp.{name}"),
                c.intermediate_size,
                c.hidden_size,
            )?;
        }
        w.expect_linear(
            &format!("{prefix}.mlp.down_proj"),
            c.hidden_size,
            c.intermediate_size,
        )?;
        Ok(Self {
            q: w.linear(&format!("{attn}.q_proj"))?,
            k: w.linear(&format!("{attn}.k_proj"))?,
            v: w.linear(&format!("{attn}.v_proj"))?,
            o: w.linear(&format!("{attn}.o_proj"))?,
            q_norm: if qk_norm {
                Some(w.rms(&format!("{attn}.q_norm"), c.rms_norm_eps)?)
            } else {
                None
            },
            k_norm: if qk_norm {
                Some(w.rms(&format!("{attn}.k_norm"), c.rms_norm_eps)?)
            } else {
                None
            },
            norm1: Norm::Rms(w.rms(&format!("{prefix}.input_layernorm"), c.rms_norm_eps)?),
            norm2: Norm::Rms(w.rms(
                &format!("{prefix}.post_attention_layernorm"),
                c.rms_norm_eps,
            )?),
            gate: Some(w.linear(&format!("{prefix}.mlp.gate_proj"))?),
            up: w.linear(&format!("{prefix}.mlp.up_proj"))?,
            down: w.linear(&format!("{prefix}.mlp.down_proj"))?,
            attn_scale: if layer_scale {
                Some(w.get(&format!("{prefix}.self_attn_layer_scale.scale"))?)
            } else {
                None
            },
            mlp_scale: if layer_scale {
                Some(w.get(&format!("{prefix}.mlp_layer_scale.scale"))?)
            } else {
                None
            },
            config: c.clone(),
            attention_window: c.sliding_window,
            rope: RoPE::new(c.head_dim as i32, Some(false), Some(c.rope_theta), None),
        })
    }
    pub fn forward(&self, x: &MxArray, state: &mut AttentionState) -> Result<MxArray> {
        let c = &self.config;
        let shape = x.shape()?;
        let (b, n) = (shape[0], shape[1]);
        let h = self.norm1.forward(x)?;
        let mut q = self
            .q
            .forward(&h)?
            .reshape(&[b, n, c.num_attention_heads, c.head_dim])?;
        let mut k = self
            .k
            .forward(&h)?
            .reshape(&[b, n, c.num_key_value_heads, c.head_dim])?;
        let v = self
            .v
            .forward(&h)?
            .reshape(&[b, n, c.num_key_value_heads, c.head_dim])?
            .transpose(Some(&[0, 2, 1, 3]))?;
        if let Some(norm) = &self.q_norm {
            q = norm.forward(&q)?;
        }
        if let Some(norm) = &self.k_norm {
            k = norm.forward(&k)?;
        }
        // TTS uses identical temporal/height/width positions; interleaved MRoPE
        // therefore equals ordinary nontraditional RoPE for the three identical position axes.
        q = self.rope.forward(
            &q.transpose(Some(&[0, 2, 1, 3]))?,
            Some(state.position as i32),
        )?;
        k = self.rope.forward(
            &k.transpose(Some(&[0, 2, 1, 3]))?,
            Some(state.position as i32),
        )?;
        let previous = state.kv.get_offset() as i64;
        let (k, v) = state.kv.update_and_fetch(&k, &v)?;
        let total = previous + n;
        let mask = if n > 1 || self.attention_window.is_some() {
            let mut data = vec![0f32; (n * total) as usize];
            for row in 0..n {
                for col in 0..total {
                    if col > previous + row
                        || self
                            .attention_window
                            .is_some_and(|window| col + window as i64 <= previous + row)
                    {
                        data[(row * total + col) as usize] = f32::NEG_INFINITY;
                    }
                }
            }
            Some(MxArray::from_float32(&data, &[n, total])?.astype(x.dtype()?)?)
        } else {
            None
        };
        let h = scaled_dot_product_attention(
            &q,
            &k,
            &v,
            (c.head_dim as f64).powf(-0.5),
            mask.as_ref(),
        )?
        .transpose(Some(&[0, 2, 1, 3]))?
        .reshape(&[b, n, c.num_attention_heads * c.head_dim])?;
        let mut h = self.o.forward(&h)?;
        if let Some(scale) = &self.attn_scale {
            h = h.mul(scale)?;
        }
        let x = x.add(&h)?;
        let h = self.norm2.forward(&x)?;
        let h = match &self.gate {
            Some(gate) => Activations::swiglu_compiled(&gate.forward(&h)?, &self.up.forward(&h)?)?,
            // Mimi's hidden_act="gelu" uses erf GELU (Transformers 4.57.3),
            // unlike the tanh approximation used by some other model families.
            None => Activations::gelu_exact(&self.up.forward(&h)?)?,
        };
        let mut h = self.down.forward(&h)?;
        if let Some(scale) = &self.mlp_scale {
            h = h.mul(scale)?;
        }
        state.position += n as usize;
        if let Some(window) = self.attention_window {
            state.kv.retain_recent(window.saturating_sub(1))?;
        }
        x.add(&h)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn qwen3_tts_encoder_mlp_matches_exact_gelu() {
        let mut config: TransformerConfig = serde_json::from_value(
            serde_json::from_str::<serde_json::Value>(include_str!("fixtures/codec.json")).unwrap()
                ["encoder_config"]
                .clone(),
        )
        .unwrap();
        config.hidden_size = 2;
        config.intermediate_size = 2;
        config.num_attention_heads = 1;
        config.num_key_value_heads = 1;
        config.head_dim = 2;
        let zeros = MxArray::from_float32(&[0.; 4], &[2, 2]).unwrap();
        let identity = MxArray::from_float32(&[1., 0., 0., 1.], &[2, 2]).unwrap();
        let linear = |weight: &MxArray| Linear::from_weights(weight, None).unwrap();
        let layer = DecoderLayer {
            q: linear(&zeros),
            k: linear(&zeros),
            v: linear(&zeros),
            o: linear(&zeros),
            q_norm: None,
            k_norm: None,
            norm1: Norm::Layer(LayerNorm::new(2, Some(1e-6)).unwrap()),
            norm2: Norm::Layer(LayerNorm::new(2, Some(1e-6)).unwrap()),
            gate: None,
            up: linear(&identity),
            down: linear(&identity),
            attn_scale: None,
            mlp_scale: None,
            config,
            attention_window: None,
            rope: RoPE::new(2, Some(false), Some(10000.), None),
        };
        let input = MxArray::from_float32(&[-1., 1.], &[1, 1, 2]).unwrap();
        let output = layer
            .forward(&input, &mut AttentionState::default())
            .unwrap()
            .to_float32()
            .unwrap();
        // x + GELU(LayerNorm(x)); scalar erf reference with eps=1e-6.
        // The tanh approximation differs by ~1.53e-4 at both positions.
        for (actual, expected) in output.iter().zip([-1.158655295589131, 1.8413442044112442]) {
            assert!((f64::from(*actual) - expected).abs() < 2e-6);
        }
    }
}

pub struct Decoder {
    layers: Vec<DecoderLayer>,
    norm: Option<RMSNorm>,
    pub dtype: DType,
}
impl Decoder {
    pub fn load(
        w: &Weights,
        prefix: &str,
        c: &TransformerConfig,
        qk: bool,
        scales: bool,
    ) -> Result<Self> {
        let layers = (0..c.num_hidden_layers)
            .map(|i| DecoderLayer::load(w, &format!("{prefix}.layers.{i}"), c, qk, scales))
            .collect::<Result<_>>()?;
        let norm = w.rms(&format!("{prefix}.norm"), c.rms_norm_eps)?;
        let dtype = norm.get_weight().dtype()?;
        Ok(Self {
            layers,
            norm: Some(norm),
            dtype,
        })
    }
    pub fn load_encoder(w: &Weights, prefix: &str, c: &TransformerConfig) -> Result<Self> {
        c.validate()?;
        if c.hidden_act != "gelu" {
            return Err(Error::from_reason("Unsupported TTS encoder activation"));
        }
        let layers = (0..c.num_hidden_layers)
            .map(|i| {
                let p = format!("{prefix}.layers.{i}");
                let a = format!("{p}.self_attn");
                Ok(DecoderLayer {
                    q: w.linear(&format!("{a}.q_proj"))?,
                    k: w.linear(&format!("{a}.k_proj"))?,
                    v: w.linear(&format!("{a}.v_proj"))?,
                    o: w.linear(&format!("{a}.o_proj"))?,
                    q_norm: None,
                    k_norm: None,
                    norm1: Norm::Layer(w.norm(&format!("{p}.input_layernorm"), c.rms_norm_eps)?),
                    norm2: Norm::Layer(
                        w.norm(&format!("{p}.post_attention_layernorm"), c.rms_norm_eps)?,
                    ),
                    gate: None,
                    up: w.linear(&format!("{p}.mlp.fc1"))?,
                    down: w.linear(&format!("{p}.mlp.fc2"))?,
                    attn_scale: Some(w.get(&format!("{p}.self_attn_layer_scale.scale"))?),
                    mlp_scale: Some(w.get(&format!("{p}.mlp_layer_scale.scale"))?),
                    config: c.clone(),
                    // Match Qwen's offline encoder and transformers 4.57.3
                    // Mimi SDPA/eager: its explicit causal mask does not apply
                    // config.sliding_window (unlike its FlashAttention path).
                    attention_window: None,
                    rope: RoPE::new(c.head_dim as i32, Some(false), Some(c.rope_theta), None),
                })
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            layers,
            norm: None,
            dtype: w
                .get(&format!("{prefix}.layers.0.self_attn.q_proj.weight"))?
                .dtype()?,
        })
    }
    pub fn state(&self) -> Vec<AttentionState> {
        (0..self.layers.len())
            .map(|_| AttentionState::default())
            .collect()
    }
    pub fn forward(&self, input: &MxArray, states: &mut [AttentionState]) -> Result<MxArray> {
        let mut x = input.astype(self.dtype)?;
        for (layer, state) in self.layers.iter().zip(states) {
            x = layer.forward(&x, state)?;
        }
        match &self.norm {
            Some(norm) => norm.forward(&x),
            None => Ok(x),
        }
    }
}
