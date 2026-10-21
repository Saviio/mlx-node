"""Download pinned data/model artifacts, never remote Python code."""

import argparse
import json
from pathlib import Path
from huggingface_hub import snapshot_download, hf_hub_download

parser = argparse.ArgumentParser()
parser.add_argument("root", type=Path)
args = parser.parse_args()
repo = Path(__file__).resolve().parents[3]
manifest = json.loads((repo / "fixtures/rerank/bge/manifest.json").read_text())
snapshot_download(
    repo_id=manifest["model"]["repository"],
    revision=manifest["model"]["revision"],
    local_dir=args.root / "fp32",
    allow_patterns=[
        "*.json",
        "*.safetensors",
        "sentencepiece.bpe.model",
        "README.md",
        "LICENSE*",
        "NOTICE*",
    ],
    max_workers=2,
)
for dataset in manifest["datasets"]:
    hf_hub_download(
        repo_id=dataset["repository"],
        repo_type="dataset",
        revision=dataset["revision"],
        filename=dataset["file"],
        local_dir=args.root / "datasets" / dataset["repository"].split("/")[-1],
    )
