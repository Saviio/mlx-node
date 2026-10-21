"""Pinned, independent reference. Never imports the addon or executes repository code."""

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import math
import numpy as np
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[3]
MANIFEST = json.loads((ROOT / "fixtures/rerank/bge/manifest.json").read_text())


def sigmoid(x):
    if not math.isfinite(x):
        raise ValueError("nonfinite logit")
    return 1 / (1 + math.exp(-x)) if x >= 0 else math.exp(x) / (1 + math.exp(x))


def prepare(tokenizer, case, config):
    policy = {
        "maxPairTokens": min(
            8192,
            config["max_position_embeddings"] - config["pad_token_id"] - 1,
            tokenizer.model_max_length,
        ),
        "maxQueryTokens": 512,
        "maxDocumentTokens": 8192,
        "maxBatchTokens": 8192,
        "maxBatchSize": 8,
        **case.get("policy", {}),
    }
    query = tokenizer.encode(case["query"], add_special_tokens=False)
    docs = [
        tokenizer.encode(text, add_special_tokens=False) for text in case["documents"]
    ]
    input_tokens = len(query) + sum(map(len, docs))
    truncated_query = len(query) > policy["maxQueryTokens"]
    query = query[: policy["maxQueryTokens"]]
    remaining = min(
        policy["maxDocumentTokens"], policy["maxPairTokens"] - len(query) - 4
    )
    truncated_docs = [i for i, d in enumerate(docs) if len(d) > remaining]
    if (truncated_query or truncated_docs) and not case.get("truncate"):
        raise ValueError("fixture requires truncation")
    pairs = []
    for i, d in enumerate(docs):
        # Pinned Transformers ID-level pair construction preserves empty pairs.
        encoded = tokenizer.prepare_for_model(
            query,
            d[:remaining],
            add_special_tokens=True,
            padding=False,
            truncation=False,
            return_attention_mask=True,
            return_token_type_ids=True,
        )
        ids = encoded["input_ids"]
        positions = (
            np.cumsum(np.array(ids) != config["pad_token_id"])
            * (np.array(ids) != config["pad_token_id"])
            + config["pad_token_id"]
        ).tolist()
        pairs.append(
            {
                "index": i,
                "tokens": ids,
                "positions": positions,
                "mask": encoded["attention_mask"],
                "types": encoded["token_type_ids"],
            }
        )
    pairs.sort(key=lambda p: (-len(p["tokens"]), p["index"]))
    return (
        policy,
        pairs,
        {
            "inputTokens": input_tokens,
            "promptTokens": sum(len(p["tokens"]) for p in pairs),
        },
        truncated_query,
        truncated_docs,
    )


class MLXReference:
    """Functional XLM-R encoder; independent of native model and quantization loaders."""

    def __init__(self, path, config):
        import mlx.core as mx

        self.mx, self.c = mx, config
        index = path / "model.safetensors.index.json"
        files = (
            sorted(set(json.loads(index.read_text())["weight_map"].values()))
            if index.exists()
            else ["model.safetensors"]
        )
        self.w = {}
        for name in files:
            for key, value in mx.load(str(path / name)).items():
                if key in self.w:
                    raise ValueError(f"duplicate {key}")
                self.w[key] = value
        self.quant = config.get("quantization", config.get("quantization_config", {}))
        mx.eval(self.w)

    def quant_params(self, name):
        q = self.quant.get(name, self.quant)
        return {
            "group_size": q.get("group_size", self.quant.get("group_size", 64)),
            "bits": q.get("bits", self.quant.get("bits", 4)),
            "mode": q.get("mode", self.quant.get("mode", "affine")),
        }

    def embedding(self, name, ids):
        mx, w = self.mx, self.w
        if name + ".scales" in w:
            weight, scales = w[name + ".weight"][ids], w[name + ".scales"][ids]
            biases = w.get(name + ".biases")
            return mx.dequantize(
                weight,
                scales,
                None if biases is None else biases[ids],
                **self.quant_params(name),
            ).astype(self.w["roberta.embeddings.position_embeddings.weight"].dtype)
        return w[name + ".weight"][ids]

    def linear(self, name, x):
        mx, w = self.mx, self.w
        if name + ".scales" in w:
            y = mx.quantized_matmul(
                x,
                w[name + ".weight"],
                w[name + ".scales"],
                w.get(name + ".biases"),
                transpose=True,
                **self.quant_params(name),
            )
        else:
            # Fix the projection rounding boundary explicitly. The shared native
            # Linear also evaluates matmul then bias, rather than fused addmm.
            return (x @ w[name + ".weight"].T) + w[name + ".bias"]
        result = y + w[name + ".bias"]
        # The profile fixes activation dtype at projection boundaries, including
        # affine checkpoints whose sidecars retain a higher precision.
        return result.astype(x.dtype) if name + ".scales" in w else result

    def norm(self, name, x):
        return self.mx.fast.layer_norm(
            x,
            self.w[name + ".weight"],
            self.w[name + ".bias"],
            self.c["layer_norm_eps"],
        )

    def __call__(self, batch):
        import mlx.nn as nn

        mx, c = self.mx, self.c
        ids, positions, types, mask = (
            mx.array(batch[k]) for k in ["tokens", "positions", "types", "mask"]
        )
        x = self.norm(
            "roberta.embeddings.LayerNorm",
            self.embedding("roberta.embeddings.word_embeddings", ids)
            + self.embedding("roberta.embeddings.token_type_embeddings", types)
            + self.embedding("roberta.embeddings.position_embeddings", positions),
        )
        b, n, h = x.shape
        heads, d = c["num_attention_heads"], h // c["num_attention_heads"]
        mask = mx.where(
            mask[:, None, None, :],
            mx.array(0, x.dtype),
            mx.array(-float("inf"), x.dtype),
        )
        stages = [np.array(x[:, 0].astype(mx.float32)).tolist()]
        for i in range(c["num_hidden_layers"]):
            base = f"roberta.encoder.layer.{i}"
            q, k, v = (
                self.linear(base + ".attention.self." + key, x)
                .reshape(b, n, heads, d)
                .transpose(0, 2, 1, 3)
                for key in ["query", "key", "value"]
            )
            attn = (
                mx.fast.scaled_dot_product_attention(q, k, v, scale=d**-0.5, mask=mask)
                .transpose(0, 2, 1, 3)
                .reshape(b, n, h)
            )
            x = self.norm(
                base + ".attention.output.LayerNorm",
                self.linear(base + ".attention.output.dense", attn) + x,
            )
            x = self.norm(
                base + ".output.LayerNorm",
                self.linear(
                    base + ".output.dense",
                    nn.gelu(self.linear(base + ".intermediate.dense", x)),
                )
                + x,
            )
            mx.eval(x)
            stages.append(np.array(x[:, 0].astype(mx.float32)).tolist())
        logits = self.linear(
            "classifier.out_proj", mx.tanh(self.linear("classifier.dense", x[:, 0]))
        ).astype(mx.float32)
        mx.eval(logits)
        return np.array(logits).reshape(-1).tolist(), stages


class TorchReference:
    def __init__(self, path, config):
        import torch
        from transformers import AutoModelForSequenceClassification

        torch.set_num_threads(4)
        self.torch = torch
        self.model = AutoModelForSequenceClassification.from_pretrained(
            path,
            local_files_only=True,
            trust_remote_code=False,
            torch_dtype=torch.float32,
        ).eval()

    def __call__(self, batch):
        torch = self.torch
        inputs = {
            name: torch.tensor(batch[key])
            for name, key in [
                ("input_ids", "tokens"),
                ("position_ids", "positions"),
                ("attention_mask", "mask"),
                ("token_type_ids", "types"),
            ]
        }
        with torch.inference_mode():
            output = self.model(**inputs, output_hidden_states=True)
        return output.logits[:, 0].tolist(), [
            x[:, 0].tolist() for x in output.hidden_states
        ]


def padded(pairs, pad):
    n = max(len(p["tokens"]) for p in pairs)
    return {
        key: [
            p[key] + [pad if key in ["tokens", "positions"] else 0] * (n - len(p[key]))
            for p in pairs
        ]
        for key in ["tokens", "positions", "types", "mask"]
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("backend", choices=["torch", "mlx"])
    ap.add_argument("model", type=Path)
    ap.add_argument("output", type=Path)
    ap.add_argument(
        "--cases", type=Path, default=ROOT / "fixtures/rerank/bge/cases.json"
    )
    args = ap.parse_args()
    for package in [
        "transformers",
        "tokenizers",
        "torch" if args.backend == "torch" else "mlx",
    ]:
        assert (
            importlib.metadata.version(package) == MANIFEST["reference"][package]
        ), package
    config = json.loads((args.model / "config.json").read_text())
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, local_files_only=True, trust_remote_code=False
    )
    model = (TorchReference if args.backend == "torch" else MLXReference)(
        args.model, config
    )
    report = {"backend": args.backend, "checksums": {}, "cases": []}
    for file in sorted(args.model.glob("*")):
        if file.suffix in [".json", ".safetensors"]:
            with file.open("rb") as stream:
                report["checksums"][file.name] = hashlib.file_digest(
                    stream, "sha256"
                ).hexdigest()
    for case in json.loads(args.cases.read_text()):
        policy, pairs, usage, tq, td = prepare(tokenizer, case, config)
        rows, start, padded_tokens = [], 0, 0
        while start < len(pairs):
            size = min(
                policy["maxBatchSize"],
                policy["maxBatchTokens"] // len(pairs[start]["tokens"]),
                len(pairs) - start,
            )
            group = pairs[start : start + size]
            batch = padded(group, config["pad_token_id"])
            logits, stages = model(batch)
            padded_tokens += size * len(batch["tokens"][0])
            for i, pair in enumerate(group):
                rows.append(
                    {
                        **pair,
                        "logit": logits[i],
                        "score": sigmoid(logits[i]),
                        "stages": [stage[i] for stage in stages],
                    }
                )
            start += size
        report["cases"].append(
            {
                "name": case["name"],
                "pairs": rows,
                "usage": usage,
                "truncatedQuery": tq,
                "truncatedDocumentIndices": td,
                "paddedTokens": padded_tokens,
            }
        )
        print(case["name"], [(p["index"], p["score"]) for p in rows], flush=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
