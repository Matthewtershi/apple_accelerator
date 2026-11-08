import sys
import mlx.core as mx

def main():
    if len(sys.argv) != 2:
        sys.exit("usage: inspect_reference.py <model.safetensors>")
    path = sys.argv[1] # argv[0] is the script name, argv[1] is the first user provided argument 

    weights = mx.load(path)

    for name, arr in sorted(weights.items()):
        dtype = str(arr.dtype).removeprefix("mlx.core.")
        print(f"{name} {list(arr.shape)} {dtype}")

    def note(*args):
        print(*args, file = sys.stderr)
    
    for key in ["model.embed_tokens.weight", "lm_head.weight"]:
        note(f"{key}: {list(weights[key].shape)}")

    for key in ["model.layers.0.self_attn.q_norm.weight", "model.layers.0.self_attn.k_norm.weight"]:
        note(f"{key} present: {key in weights}")

    # keep track of the worst error seen
    worst = 0.0
    for name in sorted(n for n in weights if n.endswith(".scales")):
        prefix = name.removesuffix(".scales")
        # upcast fp16 to fp32 so the comparison itself doesn't add rounding error
        s = weights[prefix + ".scales"].astype(mx.float32)
        b = weights[prefix + ".biases"].astype(mx.float32)
        
        err = mx.max(mx.abs(b + s)).item()
        worst = max(worst, err)

    note(f"max |bias + scale| over all quantized tensors: {worst}")

    w = weights["model.layers.0.self_attn.q_proj.weight"]
    counts = [0, 0, 0, 0] # count of each code value 0..3 (over all 16 positions in each word)
    for shift in range(0, 32 ,2):
        codes = (w >> shift) & 3
        for v in range(4):
            counts[v] += (codes == v).sum().item()
    
    note(f"q_proj layer 0 code counts [0, 1, 2, 3]: {counts}")

if __name__ == "__main__":
    main()