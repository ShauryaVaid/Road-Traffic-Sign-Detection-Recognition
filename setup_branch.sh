#!/usr/bin/env bash
# Run from the ROOT of your team's git repo AFTER copying these files into it.
#   bash setup_branch.sh
set -e
BRANCH="feature/data-preprocessing"

BASE=$(git symbolic-ref --short refs/remotes/origin/HEAD 2>/dev/null | sed 's#origin/##' || echo main)
git checkout "$BASE" && git pull
git checkout -b "$BRANCH"

git add .gitignore requirements.txt src/__init__.py src/config.py
git commit -m "data: project config, requirements, gitignore"

git add src/download_data.py src/preprocess.py
git commit -m "data: Kaggle GTSRB download and cleaning/preprocessing (ROI crop, 48x48 RGB)"

git add src/split_data.py
git commit -m "data: group-aware stratified train/val split (no track leakage)"

git add src/augment.py
git commit -m "data: augmentation pipeline (no flips) and class balancing"

git add src/eda.py src/data_loader.py run_pipeline.py tests/ outputs/
git commit -m "data: EDA charts, Keras/PyTorch loaders, pipeline runner, smoke test"

git add README.md setup_branch.sh
git commit -m "docs: Member A README and hand-off contract"

git push -u origin "$BRANCH"
echo "Pushed $BRANCH. Now open a Pull Request into $BASE on GitHub."
