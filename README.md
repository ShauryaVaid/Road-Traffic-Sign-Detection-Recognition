# Traffic Sign Recognition: Data & Preprocessing (Member A)

Branch: `feature/data-preprocessing`
Dataset: Kaggle "GTSRB - German Traffic Sign Recognition Benchmark" (43 classes, ~39k train / ~12.6k test images)

## Run everything
```bash
pip install -r requirements.txt
python run_pipeline.py                  # download -> clean -> split -> augment -> EDA
# already downloaded the dataset manually?
python -m src.download_data --path "C:/path/to/unzipped/folder"
python run_pipeline.py --skip-download
```
Kaggle download needs a free Kaggle account + API token (`~/.kaggle/kaggle.json`), or download the zip by hand and use `--path`.

## What each step does
| Step | Script | Output |
|---|---|---|
| 0 | `src/download_data.py` | `data/raw/` |
| 1 | `src/preprocess.py` | drops corrupt images, crops sign using ROI, resizes to 48x48 RGB, `outputs/eda/cleaning_report.txt` |
| 2 | `src/split_data.py` | `train.npz`, `val.npz`, `test.npz` (80/20 split **by sign track**, no leakage) |
| 3 | `src/augment.py` | `train_balanced.npz`, `outputs/eda/augmentation_preview.png` |
| 4 | `src/eda.py` | charts in `outputs/eda/` |

## Hand-off contract (for B, C, D)
Every `.npz` has `X` (uint8, N x 48 x 48 x 3, RGB) and `y` (int64, class id 0-42). Data files are NOT in git; run the pipeline locally (about 5 min).

```python
from src.data_loader import load_split, class_weights, keras_dataset, get_torch_dataset
X_train, y_train = load_split("train_balanced")   # or "train", "val", "test"
X_train = X_train.astype("float32") / 255.0
```
- **C (classification):** train on `train_balanced` (or `train` + `class_weights(y)`), tune on `val`, report final numbers on `test` only once.
- **B (detection):** `src/preprocess.py` shows the ROI crop logic; for the final pipeline, crop detected boxes and resize to `IMG_SIZE` (48), RGB, divide by 255 before classifying.
- **D (integration):** input to the classifier is always 48x48 RGB float32 in [0,1]. Class names are in `src/config.py` (`CLASS_NAMES`).

## Design decisions
- **No flip augmentation:** left/right signs would swap meaning.
- **Group split:** each physical sign has ~30 frames; random splitting would leak them into validation.
- **Val/test never augmented**; only the train split is.
- Change `IMG_SIZE` etc. in `src/config.py`, then rerun the pipeline.

## Smoke test without the real dataset
```bash
GTSRB_DATA_DIR=/tmp/fake python tests/make_fake_dataset.py
GTSRB_DATA_DIR=/tmp/fake python run_pipeline.py --skip-download --target 40
```
