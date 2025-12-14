#!/bin/bash

set -e

echo "--- 1. STEP: Data Processing ---"
python src/01_data_processing.py

echo "--- 2. STEP: Model Training ---"
python src/02_train.py

echo "--- 3. STEP: Evaluation ---"
python src/03_evaluation.py

echo "--- 4. STEP: Inference (Prediction) ---"
python src/04_inference.py

echo "SUCCESS: Pipeline finished successfully."