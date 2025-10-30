#!/usr/bin/env python3
"""
run.py — Main training and submission script for the CVD Prediction project.

This script:
  1. Loads and preprocesses the BRFSS dataset.
  2. Builds polynomial features.
  3. Trains a logistic regression model.
  4. Generates predictions and saves the submission file.
"""

import numpy as np
from helpers import *
from implementations import *
from preprocessing import *
from model_evaluation import *

# -------------------------------------------------------------
# Configuration
# -------------------------------------------------------------
DATA_FOLDER = "./data/"
OUTPUT_FILE = "logistic_regression_submission.csv"

# Hyperparameters previously determined via cross-validation
BEST_GAMMA = 0.5
BEST_DEGREE = 2
MAX_ITERS = 1000
BEST_THRESHOLD = 0.17

# -------------------------------------------------------------
# Main pipeline
# -------------------------------------------------------------
def main():
    """Main pipeline: load data, preprocess, train model, predict, save."""
    print("🔹 Loading data...")
    data = load_csv_data(DATA_FOLDER, max_rows=None, dictionnary=True)

    print("🔹 Preprocessing data...")
    preprocess_data(
        data,
        nan_drop_threshold=0.9,
        correlation_threshold=0.01,
        n_std=3,
        only_health_related=False,
        split_val=True,
        val_size=0.1,
    )

    print("🔹 Building polynomial features...")
    x_train_poly = build_poly(data["x_train"], BEST_DEGREE, to_expand=data["continuos"])
    x_val_poly = build_poly(data["x_val"], BEST_DEGREE, to_expand=data["continuos"])
    x_test_poly = build_poly(data["x_test"], BEST_DEGREE, to_expand=data["continuos"])

    print("🔹 Training logistic regression...")
    w_initial = np.zeros(x_train_poly.shape[1])
    w, loss = logistic_regression(
        data["y_train"], x_train_poly, w_initial, MAX_ITERS, BEST_GAMMA, return_history=False
    )
    print(f"✅ Training complete. Final loss: {loss:.4f}")

    print("🔹 Generating predictions...")
    test_pred = predict_labels_logistic(x_test_poly, w, threshold=BEST_THRESHOLD)
    test_pred = np.where(test_pred == 1, 1, -1)

    print(f"🔹 Saving predictions to '{OUTPUT_FILE}'...")
    create_csv_submission(data["test_ids"], test_pred, OUTPUT_FILE)
    print("🎉 Submission file created successfully.")


# -------------------------------------------------------------
# Entry point
# -------------------------------------------------------------
if __name__ == "__main__":
    main()
