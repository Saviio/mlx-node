"""Build a deliberate same-checkpoint test of F32 sidecars and a packed classifier."""

import argparse, json, shutil
from pathlib import Path
import mlx.core as mx

ap = argparse.ArgumentParser()
ap.add_argument("source", type=Path)
ap.add_argument("output", type=Path)
a = ap.parse_args()
a.output.mkdir(parents=True, exist_ok=False)
w = mx.load(str(a.source / "model.safetensors"))
c = json.loads((a.source / "config.json").read_text())
assert c["quantization"]["mode"] == "affine"
for key in list(w):
    if key.endswith((".scales", ".biases")):
        w[key] = w[key].astype(mx.float32)
for base in ["classifier.dense", "classifier.out_proj"]:
    q, s, b = mx.quantize(w[base + ".weight"], group_size=64, bits=4, mode="affine")
    w[base + ".weight"] = q
    w[base + ".scales"] = s.astype(mx.float32)
    w[base + ".biases"] = b.astype(mx.float32)
mx.save_safetensors(str(a.output / "model.safetensors"), w)
for file in a.source.iterdir():
    if (
        file.is_file()
        and file.suffix != ".safetensors"
        and file.name != "model.safetensors.index.json"
    ):
        shutil.copy2(file, a.output / file.name)
(a.output / "model.safetensors.index.json").write_text(
    json.dumps(
        {
            "metadata": {"total_size": sum(t.nbytes for t in w.values())},
            "weight_map": {name: "model.safetensors" for name in w},
        },
        sort_keys=True,
    )
    + "\n"
)
