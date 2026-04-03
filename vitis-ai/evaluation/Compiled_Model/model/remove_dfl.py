#!/usr/bin/env python3
"""Strip ``DFL`` layer objects from a pickled quantization config on disk.

Some Vitis AI / PyTorch checkpoints store nested ``DFL`` modules inside pickles.
This utility recursively removes nodes whose class name is ``DFL`` so the file can
be consumed by tooling that does not expect those objects.

**Defaults:** reads ``config_yolov26.pkl`` and writes the cleaned result back to the
same path (in-place). Edit ``INPUT_PATH`` / ``OUTPUT_PATH`` below if needed.

Run from this directory::

    python remove_dfl.py

**Warning:** Back up the pickle before overwriting.
"""

import pickle
from pathlib import Path

INPUT_PATH = "config_yolov26.pkl"
OUTPUT_PATH = "config_yolov26.pkl"


def remove_dfl(obj):
    """Recursively remove ``DFL`` instances from tuples, lists, and dicts."""
    if obj.__class__.__name__ == "DFL":
        return None

    if isinstance(obj, tuple):
        return tuple(
            remove_dfl(x)
            for x in obj
            if x.__class__.__name__ != "DFL"
        )

    if isinstance(obj, list):
        return [
            remove_dfl(x)
            for x in obj
            if x.__class__.__name__ != "DFL"
        ]

    if isinstance(obj, dict):
        cleaned = {}
        for k, v in obj.items():
            if v.__class__.__name__ != "DFL":
                cleaned[k] = remove_dfl(v)
        return cleaned

    return obj


def main() -> None:
    in_path = Path(INPUT_PATH)
    if not in_path.is_file():
        raise FileNotFoundError(f"Input pickle not found: {in_path.resolve()}")

    with open(in_path, "rb") as f:
        data = pickle.load(f)
    print(data)
    print("Loaded successfully")

    cleaned = remove_dfl(data)
    print("DFL removed")

    out_path = Path(OUTPUT_PATH)
    with open(out_path, "wb") as f:
        pickle.dump(cleaned, f)

    print("Saved cleaned file:", out_path.resolve())


if __name__ == "__main__":
    main()
