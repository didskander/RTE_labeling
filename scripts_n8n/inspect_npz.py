import numpy as np
import json

npz_path = r"C:\Users\didsk\Desktop\Relay-protection\src\data\rte_events\DATA_S.npz"
data = np.load(npz_path, allow_pickle=True)

result = {}
for k in data.files:
    arr = data[k]
    result[k] = {
        "shape": list(arr.shape) if hasattr(arr, "shape") else None,
        "dtype": str(arr.dtype) if hasattr(arr, "dtype") else str(type(arr)),
    }

print(json.dumps(result, indent=2))