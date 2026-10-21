"""Compact evidence and artifact hashes; missing gates remain visibly pending."""

import argparse
import hashlib
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("root", type=Path)
parser.add_argument("output", type=Path)
args = parser.parse_args()
repo = Path(__file__).resolve().parents[3]
manifest = json.loads((repo / "fixtures/rerank/bge/manifest.json").read_text())


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def optional(name):
    path = args.root / name
    return json.loads(path.read_text()) if path.exists() else None


numerical = {}
for path in sorted((args.root / "logs").glob("*-golden.log")):
    text = path.read_text()
    if path.name.startswith("jina-"):
        numerical[path.stem] = {"passed": "test result: ok. 1 passed" in text}
        continue
    rows = []
    for line in text.splitlines():
        if '{"case"' in line:
            rows.append(json.loads(line[line.index('{"case"') :]))
    numerical[path.stem] = {
        "passed": bool(rows) and "test result: ok. 1 passed" in text,
        "cases": rows,
    }

quality = {}
for dtype in ["fp32", "affine", "mxfp4"]:
    raw = optional(f"quality-{dtype}.json")
    if raw is None:
        quality[dtype] = {"complete": False}
        continue
    quality[dtype] = {
        "complete": len(raw["datasets"]) == len(manifest["datasets"])
        and all(
            len(d["queries"]) == manifest["qualitySelection"]["queriesPerDataset"]
            for d in raw["datasets"]
        ),
        "inputSha256": raw["inputSha256"],
        "addonSha256": raw["addonSha256"],
        "datasets": [
            {
                **{k: v for k, v in d.items() if k != "queries"},
                "queryCount": len(d["queries"]),
            }
            for d in raw["datasets"]
        ],
    }

checkpoints = {}
for dtype in ["fp32", "fp16", "bf16", "affine", "mxfp4", "mixed"]:
    folder = args.root / dtype
    checkpoints[dtype] = {
        path.name: sha256(path)
        for path in sorted(folder.iterdir() if folder.exists() else [])
        if path.is_file()
        and (
            path.suffix in [".json", ".safetensors", ".model"]
            or path.name.startswith("LICENSE")
        )
    }

report = {
    "baselineCommit": "1fc3baee",
    "environment": optional("environment.json"),
    "manifest": manifest,
    "sourceChecksums": {
        str(p.relative_to(repo)): sha256(p)
        for p in sorted(
            [
                *(repo / "crates/mlx-core/src/models/xlm_roberta_reranker").glob(
                    "*.rs"
                ),
                repo / "crates/mlx-core/src/models/rerank_runtime.rs",
                repo / "crates/mlx-core/src/convert.rs",
                repo / "scripts/rerank/bge/reference.py",
            ]
        )
    },
    "checkpoints": checkpoints,
    "referenceChecksums": {
        p.name: sha256(p) for p in sorted(args.root.glob("*-reference.json"))
    },
    "addonChecksums": {
        str(p.relative_to(args.root)): sha256(p)
        for p in sorted(args.root.glob("*/mlx-core.darwin-arm64.node"))
    },
    "numerical": numerical,
    "quality": quality,
    "batchConsistency": optional("batch-consistency.json"),
    "surfaces": optional("surfaces.json"),
    "performance": {
        dtype: optional(f"performance-{dtype}.json")
        for dtype in ["fp16", "bf16", "affine", "mxfp4"]
    },
    "regression": optional("regression-comparison.json"),
    "coexistence": optional("regression-current.json"),
    "logChecksums": {
        p.name: sha256(p) for p in sorted((args.root / "logs").glob("*.log"))
    },
}
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
