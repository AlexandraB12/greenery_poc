import os
import torch

MODEL_PATH = "models/streetview_model.pth"

print("Exists:", os.path.exists(MODEL_PATH))
print("Size (MB):", os.path.getsize(MODEL_PATH) / (1024 * 1024))

# 1) Try TorchScript
try:
    m = torch.jit.load(MODEL_PATH, map_location="cpu")
    print("\n✅ Looks like TorchScript!")
    print(m)
    raise SystemExit
except Exception as e:
    print("\nNot TorchScript:", type(e).__name__, str(e)[:200])

# 2) Try torch.load (state_dict or checkpoint)
obj = torch.load(MODEL_PATH, map_location="cpu")

print("\n✅ torch.load succeeded. Type:", type(obj))

if isinstance(obj, dict):
    print("Dict keys:", list(obj.keys())[:30])
    # Common patterns:
    for k in ["state_dict", "model_state_dict", "model", "net", "weights"]:
        if k in obj and isinstance(obj[k], dict):
            print(f"Found nested weights under key: '{k}' (len={len(obj[k])})")
            sample = list(obj[k].keys())[:10]
            print("Sample weight keys:", sample)
            break
    else:
        # Maybe it's directly a state_dict
        if all(isinstance(k, str) for k in obj.keys()):
            sample = list(obj.keys())[:10]
            print("Sample keys:", sample)

elif hasattr(obj, "keys"):
    try:
        print("Keys sample:", list(obj.keys())[:10])
    except Exception:
        pass

print("\n✅ Done.")
