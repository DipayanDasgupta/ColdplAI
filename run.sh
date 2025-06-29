#!/bin/bash
set -e

echo "========================================="
echo "STEP 1: PREPROCESSING MIDI FILES"
echo "========================================="
python src/preprocess.py

echo "========================================="
echo "STEP 2: TRAINING THE MODEL"
echo "========================================="
python src/train.py

echo "========================================="
echo "STEP 3: GENERATING NEW MUSIC"
echo "========================================="
python src/generate.py

echo "========================================="
echo "PIPELINE COMPLETE!"
echo "Your generated song is in the 'outputs' directory."
echo "========================================="