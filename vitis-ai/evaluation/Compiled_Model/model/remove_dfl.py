#!/usr/bin/env python3
import pickle

INPUT_PATH = "config_yolov26.pkl"
OUTPUT_PATH = "config_yolov26.pkl"


# -------------------------------
# Recursively remove DFL objects
# -------------------------------
def remove_dfl(obj):
    # Remove object if it is DFL
    if obj.__class__.__name__ == "DFL":
        return None

    # Tuple
    if isinstance(obj, tuple):
        return tuple(
            remove_dfl(x)
            for x in obj
            if x.__class__.__name__ != "DFL"
        )

    # List
    if isinstance(obj, list):
        return [
            remove_dfl(x)
            for x in obj
            if x.__class__.__name__ != "DFL"
        ]

    # Dict
    if isinstance(obj, dict):
        cleaned = {}
        for k, v in obj.items():
            if v.__class__.__name__ != "DFL":
                cleaned[k] = remove_dfl(v)
        return cleaned

    # Other objects remain unchanged
    return obj


# -------------------------------
# Load pickle
# -------------------------------
with open(INPUT_PATH, "rb") as f:
    data = pickle.load(f)
    print(data)
print("Loaded successfully")

# -------------------------------
# Remove DFL
# -------------------------------
cleaned = remove_dfl(data)

print("DFL removed")

# -------------------------------
# Save new pickle
# -------------------------------
with open(OUTPUT_PATH, "wb") as f:
    pickle.dump(cleaned, f)

print("Saved cleaned file:", OUTPUT_PATH)
