"""Deterministically select the frozen quality subset; retain all candidates."""

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import pyarrow.parquet as pq
from transformers import AutoTokenizer

ap = argparse.ArgumentParser()
ap.add_argument("model", type=Path)
ap.add_argument("datasets", type=Path)
ap.add_argument("output", type=Path)
args = ap.parse_args()
root = Path(__file__).resolve().parents[3]
manifest = json.loads((root / "fixtures/rerank/bge/manifest.json").read_text())
tokenizer = AutoTokenizer.from_pretrained(
    args.model, local_files_only=True, trust_remote_code=False
)
output = {"datasets": []}
for dataset in manifest["datasets"]:
    file = args.datasets / dataset["repository"].split("/")[-1] / dataset["file"]
    if file.suffix == ".parquet":
        rows = pq.read_table(file).to_pylist()
    else:
        with gzip.open(file, "rt") as stream:
            rows = [json.loads(line) for line in stream]
    rows.sort(key=lambda r: hashlib.sha256(r["query"].encode()).hexdigest())
    selected = []
    skipped = []
    for row in rows:
        digest = hashlib.sha256(row["query"].encode()).hexdigest()
        docs = row["positive"] + row["negative"]
        if not row["query"].strip() or not row["positive"] or not docs:
            skipped.append(
                {"sha256": digest, "reason": "empty query, positives or candidates"}
            )
            continue
        raw_tokens = sum(
            len(ids)
            for ids in tokenizer([row["query"]] + docs, add_special_tokens=False)[
                "input_ids"
            ]
        )
        if len(docs) > 1024 or raw_tokens > 1000000:
            skipped.append(
                {
                    "sha256": digest,
                    "reason": "input protection",
                    "candidates": len(docs),
                    "inputTokens": raw_tokens,
                }
            )
            continue
        selected.append(
            {
                "sha256": digest,
                "query": row["query"],
                "documents": docs,
                "labels": [1] * len(row["positive"]) + [0] * len(row["negative"]),
                "inputTokens": raw_tokens,
            }
        )
        if len(selected) == manifest["qualitySelection"]["queriesPerDataset"]:
            break
    assert len(selected) == manifest["qualitySelection"]["queriesPerDataset"]
    output["datasets"].append(
        {
            **dataset,
            "fileSha256": hashlib.sha256(file.read_bytes()).hexdigest(),
            "queries": selected,
            "skipped": skipped,
        }
    )
    print(
        dataset["repository"],
        len(selected),
        "queries",
        sum(len(r["documents"]) for r in selected),
        "candidates",
        len(skipped),
        "skipped",
        flush=True,
    )
args.output.write_text(json.dumps(output, ensure_ascii=False) + "\n")
