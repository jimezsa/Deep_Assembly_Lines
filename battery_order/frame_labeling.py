import pickle
import os

ERROR = 1
CORRECT = 0

# manual annotations (post-stride frame_idx)
ERROR_RANGES = {          # start/end indexes included
    "wrong1": (345, 450), 
    "wrong2": (315, 345), 
    "wrong3": (150, 210), 
    "wrong4": (360, 450), 
    "wrong5": (270, 315), 
    # "correct1": (, ),
    # "correct2": (, ),
    # "correct3": (, ),
    # "correct4": (, ),
    # "correct5": (, )
}

IN_DIR = "yolo_cache"
OUT_DIR = "labeled_cache"
os.makedirs(OUT_DIR, exist_ok=True)

for fname in os.listdir(IN_DIR):
    if not fname.endswith(".pkl"):
        continue

    sequence = fname.replace(".pkl", "")
    in_path = os.path.join(IN_DIR, fname)

    with open(in_path, "rb") as f:
        data = pickle.load(f)

    if sequence not in ERROR_RANGES:
        print(f"⚠ No annotation for {sequence}, skipping")
        continue

    err_start, err_end = ERROR_RANGES[sequence]

    for frame in data:
        idx = frame["frame_idx"]
        if err_start <= idx <= err_end:
            frame["label"] = ERROR
        else:
            frame["label"] = CORRECT

    out_path = os.path.join(OUT_DIR, fname)
    with open(out_path, "wb") as f:
        pickle.dump(data, f)

    print(f"✔ Labeled {sequence}")
