import sys
from pathlib import Path

import mlx.core as mx
from mlx_lm import load

model_dir, out_dir = sys.argv[1], Path(sys.argv[2])
model, tok = load(model_dir) # model is nn.Module and tok is a tokenizer for text encode/decode

"""
The goal here is to first use the mlx functions as a reference to the ones I build from scratch
"""
ids = mx.array([tok.encode("Pen pineapple apple pen")])

inner = model.model
h = inner.embed_tokens(ids)
out = {
    "input_ids": ids,
    "embed_out": h,
    "layer0_attn_norm": inner.layers[0].input_layernorm(h)
}

backend = "gpu" if mx.default_device() == mx.gpu else "cpu"
out_dir.mkdir(parents = True, exist_ok = True)
# check is stored in safetensors
mx.save_safetensors(str(out_dir / f"ref_{backend}.safetensors"), out)