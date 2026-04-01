import os
import glob
import shutil
import yaml
import pandas as pd

from pathlib import Path
from collections import Counter
from ultralytics import YOLO
from sklearn.model_selection import KFold

from .utils import load_config

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

def run_cross_validation():
    """Run the k-fold cross-validation process and return the summary DataFrame."""
    # ===================== Load Config ======================
    cv_cfg = load_config('cross_validation')

    # Paths and parameters from config
    SOURCE_PATH = Path(cv_cfg.get('dataset_path', './dataset/hpd3_320/yolov8'))
    TARGET_IMAGES_PATH = SOURCE_PATH / 'images'
    TARGET_LABELS_PATH = SOURCE_PATH / 'labels'
    YAML_FILE = Path(cv_cfg.get('data_yaml_path', SOURCE_PATH / 'data.yaml'))
    ksplit = cv_cfg.get('k_splits', 10)
    random_state = cv_cfg.get('random_state', 42)
    model_path = cv_cfg.get('model', 'yolov8n.pt')
    epochs = cv_cfg.get('epochs', 300)

    os.makedirs(TARGET_IMAGES_PATH, exist_ok=True)
    os.makedirs(TARGET_LABELS_PATH, exist_ok=True)

    # Copy files from train and valid into one combined folder
    for split in ['train', 'valid']:
        for dtype in ['images', 'labels']:
            files = glob.glob(str(SOURCE_PATH / split / f"{dtype}/*"))
            for file in files:
                shutil.copy(file, TARGET_IMAGES_PATH if dtype == 'images' else TARGET_LABELS_PATH)

    # ===================== Read Labels & Classes ======================
    with open(YAML_FILE, 'r', encoding="utf-8") as y:
        data_yaml = yaml.safe_load(y)
        class_names = data_yaml['names']
    class_ids = list(range(len(class_names)))

    # Build DataFrame of label occurrences per image
    label_files = sorted(TARGET_LABELS_PATH.glob("*.txt"))
    index = [label.stem for label in label_files]
    labels_df = pd.DataFrame(columns=class_ids, index=index)

    for label_file in label_files:
        counter = Counter()
        with open(label_file, 'r') as f:
            lines = f.readlines()
        for line in lines:
            cls_id = int(line.strip().split()[0])
            counter[cls_id] += 1
        labels_df.loc[label_file.stem] = counter

    labels_df = labels_df.fillna(0.0)

    # ===================== KFold ======================
    kf = KFold(n_splits=ksplit, shuffle=True, random_state=random_state)
    kfolds = list(kf.split(labels_df))

    # Analyze label distribution
    fold_names = [f"split_{i+1}" for i in range(ksplit)]
    label_distribution = pd.DataFrame(index=fold_names, columns=class_ids)

    for i, (train_idx, val_idx) in enumerate(kfolds):
        train_sum = labels_df.iloc[train_idx].sum()
        val_sum = labels_df.iloc[val_idx].sum()
        ratio = val_sum / (train_sum + 1e-7)
        label_distribution.loc[f"split_{i+1}"] = ratio

    label_distribution.to_csv(SOURCE_PATH / 'kfold_label_distribution.csv')

    # ===================== Create KFold Sets ======================
    KSET_PATH = SOURCE_PATH / 'kfold'
    shutil.rmtree(KSET_PATH, ignore_errors=True)
    KSET_PATH.mkdir(parents=True, exist_ok=True)

    all_images = {
        img.stem: img
        for img in TARGET_IMAGES_PATH.iterdir()
        if img.is_file() and img.suffix.lower() in IMAGE_EXTENSIONS
    }

    yaml_paths = []

    for i, (train_idx, val_idx) in enumerate(kfolds):
        train_files = labels_df.iloc[train_idx].index
        val_files = labels_df.iloc[val_idx].index

        # Generate train and val .txt files (full absolute paths)
        train_txt = KSET_PATH / f'train_{i}.txt'
        val_txt = KSET_PATH / f'val_{i}.txt'

        with open(train_txt, 'w') as f:
            for name in train_files:
                if name in all_images:
                    f.write(str(all_images[name]) + '\n')

        with open(val_txt, 'w') as f:
            for name in val_files:
                if name in all_images:
                    f.write(str(all_images[name]) + '\n')

        # Write YAML file
        yaml_data = {
            'train': str(train_txt.resolve()),
            'val': str(val_txt.resolve()),
            'names': class_names
        }

        yaml_path = KSET_PATH / f'data_fold_{i}.yaml'
        with open(yaml_path, 'w') as yf:
            yaml.safe_dump(yaml_data, yf)

        yaml_paths.append(str(yaml_path))

    # ===================== YOLO Params from Config ======================
    # Remove keys that are not YOLO params
    yolo_param_exclude = {'dataset_path', 'data_yaml_path', 'k_splits', 'random_state', 'task', 'project'}
    yolo_params = {k: v for k, v in cv_cfg.items() if k not in yolo_param_exclude}
    yolo_params['epochs'] = epochs
    yolo_params['project'] = cv_cfg.get('project', 'kfold_result')

    # ===================== Train Each Fold ======================
    results = []
    for i, yaml_path in enumerate(yaml_paths):
        print(f"\n🚀 Training Fold {i+1} with {yaml_path}")
        model = YOLO(model_path)
        model.train(data=yaml_path, name=f'fold_{i}', **yolo_params)
        results.append(model.metrics)

    # ===================== Collect Metrics ======================
    metric_values = dict()
    for res in results:
        for metric, value in res.results_dict.items():
            metric_values.setdefault(metric, []).append(value)

    metric_df = pd.DataFrame(metric_values)
    summary = metric_df.describe().loc[['mean', 'std', 'min', 'max']]
    summary.to_csv(SOURCE_PATH / 'kfold_metrics_summary.csv')

    print("\n✅ KFold Training Done!")
    print(summary)
    return summary

# If run as a script, execute the function
if __name__ == "__main__":
    run_cross_validation()

