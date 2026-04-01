# OBB test data

Place the **test set** for OBB (oriented bounding box) evaluation here. Expected layout (COCO-style):

```
test_data/
├── images/
├── labels/
├── test2017.txt
├── train2017.txt
└── val2017.txt
```

Adjust filenames and paths as required by `OBB/evaluate.py` and the OBB post-processing pipeline.
