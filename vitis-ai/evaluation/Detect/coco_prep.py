import sys
import json

model_name = sys.argv[1]

predictions = []

listed_lines = []
with open(f'output_boxes_{model_name}.txt', 'r') as f:
    for line in f:
        items = line.strip().split(" ")
        listed_lines.append(items)

for i in listed_lines:
    if i[0] == "Image":
        j = 4
        image_id = i[2]  # keep filename as string
        while (listed_lines[listed_lines.index(i)+j][0] != ''):
            bbox_line = listed_lines[listed_lines.index(i)+j]
            bbox = [round(float(bbox_line[k]),5) for k in range(5)]
            predictions.append({"image_id": image_id, "category_id": 1, "bbox": bbox})
            j += 1

merged_dict = {}
for item in predictions:
    img_id = item['image_id']
    val = item['bbox']
    if img_id not in merged_dict:
        merged_dict[img_id] = []
    merged_dict[img_id].append(val)

with open(f"coco-preped-{model_name}.json", 'w') as f:
    json.dump(merged_dict, f)
