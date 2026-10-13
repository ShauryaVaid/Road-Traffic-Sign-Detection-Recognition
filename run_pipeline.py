"""Run Member A's whole pipeline in order.   python run_pipeline.py [--skip-download] [--target 1000]"""
import argparse
import subprocess
import sys

ap = argparse.ArgumentParser()
ap.add_argument("--skip-download", action="store_true")
ap.add_argument("--target", default="1000")
a = ap.parse_args()

steps = [] if a.skip_download else [["-m", "src.download_data"]]
steps += [["-m", "src.preprocess"], ["-m", "src.split_data"],
          ["-m", "src.augment", "--target", a.target], ["-m", "src.eda"]]
for s in steps:
    print("\n=== python", " ".join(s))
    if subprocess.call([sys.executable, *s]) != 0:
        sys.exit("Step failed: " + " ".join(s))
print("\nDone. Processed data in data/processed/, charts in outputs/eda/")
