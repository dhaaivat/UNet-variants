#!/bin/bash
# Zip the code (no data, no runs, no venv) for upload as a Kaggle dataset. Output: dist/unet-code.zip
set -e
cd "$(dirname "$0")/.."
mkdir -p dist
rm -f dist/unet-code.zip
zip -qr dist/unet-code.zip train.py requirements.txt pytest.ini models utils scripts splits tests kaggle \
  -x '*/__pycache__/*' '*.pyc' '*/.DS_Store'
echo "wrote dist/unet-code.zip ($(du -h dist/unet-code.zip | cut -f1))"
unzip -l dist/unet-code.zip | tail -n +4 | awk '{print $4}' | grep -v '^$' | head -40
