"""Compact evidence and artifact hashes; missing gates remain visibly pending."""

import argparse
import hashlib
import json
import subprocess
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

quality = json.loads(
    subprocess.check_output(
        ["node", str(repo / "scripts/rerank/bge/quality-summary.mjs"), str(args.root)],
        text=True,
    )
)

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
                *(repo / "scripts/rerank/bge").glob("*.py"),
                *(repo / "scripts/rerank/bge").glob("*.mjs"),
                *(repo / "scripts/rerank/bge").glob("*.ts"),
            ]
        )
    },
    "fixtureChecksums": {
        str(p.relative_to(repo)): sha256(p)
        for p in sorted((repo / "fixtures/rerank/bge").iterdir())
        if p.is_file()
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
    "forwardCancellation": {
        dtype: optional(f"cancellation-{dtype}.json")
        for dtype in ["fp16", "bf16", "affine", "mxfp4"]
    },
    "performancePrecheck": optional("performance-fp16-precheck.json"),
    "verificationRuns": [
        json.loads(line)
        for line in (args.root / "final-checks.jsonl").read_text().splitlines()
        if line.strip()
    ] if (args.root / "final-checks.jsonl").exists() else [],
    "performance": {
        dtype: optional(f"performance-{dtype}.json")
        for dtype in ["fp16", "bf16", "affine", "mxfp4"]
    },
    "regression": {
        attempt: optional(f"regression-comparison-{attempt}.json")
        for attempt in ["initial", "repeat"]
    },
    "coexistence": {
        "initial": optional("regression-current.json"),
        "repeat": optional("regression-current-repeat.json"),
    },
    "logChecksums": {
        p.name: sha256(p) for p in sorted((args.root / "logs").glob("*.log"))
    },
}
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
