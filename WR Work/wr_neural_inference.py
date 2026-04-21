"""
Inference script for scoring a new cohort of WR prospects using the
trained GKF model checkpoint.

1. Load the pickled checkpoint from Model_Output/model_checkpoint.pkl.
2. Prepare WR_test.csv using the same preprocessing the training run
   applied (imputation statistics from WR_train.csv, label encoders
   from the checkpoint).
3. Run the test data through all five GKF fold models and average
   their probability outputs (ensemble mean).
4. Apply the threshold from fold 1 to produce binary predictions.
5. Print the top 15 candidates and save full results to Predictions/.
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
import torch

from wr_neural_model import EXCLUDED_COLS

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_checkpoint(model_path: str) -> dict:
    """
    Load the pickled model checkpoint from disk.

    The checkpoint is a plain Python dict containing:
        cv_strategy: str -- 'GKF' (always, for inference)
        cv_results: list[dict] -- one dict per fold, each with
                         'model', 'scaler', 'best_threshold', etc.
        feature_names : list[str] -- ordered feature column names
        label_encoders: dict -- {col: fitted LabelEncoder}
    """
    if not os.path.exists(model_path):
        print(f"ERROR: {model_path} not found")
        print("Run wr_neural_pipeline.py first to train the model")
        sys.exit(1)

    with open(model_path, 'rb') as f:
        return pickle.load(f)


def prepare_test_data(test_csv: str, train_csv: str, checkpoint: dict) -> tuple:
    """
    Preprocess the test CSV to match the training feature schema exactly.
    """
    test_df  = pd.read_csv(test_csv)
    train_df = pd.read_csv(train_csv)

    print(f"Loaded test data: {len(test_df)} WRs")

    feature_names = checkpoint['feature_names']
    encoders      = checkpoint['label_encoders']

    # Mirror the exact drop list from WRDraftPredictor.prepare_data.
    drop_cols = [
        'overall', 'round', 'pick', 'nfl_team', 'name', 'position', 'college_team',
        "regular_season_wins", "regular_season_losses",
        "postseason_games", "postseason_wins", "postseason_losses",
    ] + [c for c in EXCLUDED_COLS if c in test_df.columns]

    numeric_cols     = [c for c in test_df.select_dtypes(include=[np.number]).columns
                        if c not in drop_cols]
    categorical_cols = [c for c in test_df.select_dtypes(include=['object']).columns
                        if c not in drop_cols + ['name']]

    # Impute using training set statistics to avoid leakage.
    for col in numeric_cols:
        if col in train_df.columns and test_df[col].isnull().sum() > 0:
            test_df[col].fillna(train_df[col].median(), inplace=True)

    for col in categorical_cols:
        if col in train_df.columns and test_df[col].isnull().sum() > 0:
            fill_val = train_df[col].mode()[0] if len(train_df[col].mode()) > 0 else 'Unknown'
            test_df[col].fillna(fill_val, inplace=True)

    # Encode using the checkpoint's fitted encoders.
    # Unseen categories (values the encoder never saw in training) are
    # mapped to 0 rather than raising a ValueError.
    for col in categorical_cols:
        if col in encoders:
            le = encoders[col]
            test_df[col] = test_df[col].astype(str).apply(
                lambda x: le.transform([x])[0] if x in le.classes_ else 0
            )

    # Align to the exact feature set and order from training.
    # fill_value=0 handles any feature that was present in training
    # but is entirely absent from the test file.
    missing = [c for c in feature_names if c not in test_df.columns]
    if missing:
        print(f"WARNING: features in checkpoint not found in test data: {missing}")
    X_test = test_df.reindex(columns=feature_names, fill_value=0).copy()
    print(f"Feature shape: {X_test.shape}")

    return X_test, test_df


def predict(X_test: pd.DataFrame, checkpoint: dict) -> tuple:
    """
    Score the test data using all five GKF fold models.

    Each fold model applies its own fitted scaler before running the
    forward pass. Predictions are collected as probabilities (sigmoid
    of logits) and then averaged across folds to produce the ensemble
    mean probability.
    """
    print("\nGenerating predictions from 5 fold models...")

    cv_results = checkpoint['cv_results']
    all_preds  = []

    for fold_idx, result in enumerate(cv_results):
        model  = result['model'].to(DEVICE)
        scaler = result['scaler']

        # Apply this fold's scaler. Each fold's scaler was fit on that
        # fold's training data, so we must use it (not a global scaler).
        X_scaled = scaler.transform(X_test)
        X_scaled = np.nan_to_num(X_scaled, nan=0.0)
        X_tensor = torch.FloatTensor(X_scaled).to(DEVICE)

        model.eval()
        with torch.no_grad():
            logits = model(X_tensor).cpu().numpy().flatten()
            # Clip logits before sigmoid to prevent numerical overflow.
            proba  = 1.0 / (1.0 + np.exp(-np.clip(logits, -100, 100)))

        all_preds.append(proba)
        print(f"  Fold {fold_idx + 1}: mean prob = {proba.mean():.4f}")

    mean_proba = np.mean(all_preds, axis=0)
    std_proba  = np.std(all_preds, axis=0)

    print(f"\nEnsemble mean probability: {mean_proba.mean():.4f}")

    return mean_proba, std_proba, np.array(all_preds)


def main():
    """
    Run the full WR inference pipeline.

    1. Load the checkpoint.
    2. Preprocess WR_test.csv.
    3. Generate ensemble predictions.
    4. Build the results DataFrame with probability, confidence bounds,
       and binary first-round prediction.
    5. Print the top 15 candidates.
    6. Save predictions_full.csv to Predictions/.
    """
    print("\n" + "=" * 90)
    print("WR DRAFT PREDICTION INFERENCE")
    print("=" * 90)

    cwd        = os.getcwd()
    output_dir = os.path.join(cwd, 'Predictions')
    os.makedirs(output_dir, exist_ok=True)

    #  Step 1: Load checkpoint 
    print("\n[1] Loading checkpoint...")
    checkpoint = load_checkpoint(os.path.join(cwd, 'Model_Output', 'model_checkpoint.pkl'))
    print(f"    Strategy:    {checkpoint['cv_strategy']}")
    print(f"    Fold models: {len(checkpoint['cv_results'])}")
    print(f"    Features:    {len(checkpoint['feature_names'])}")

    #  Step 2: Preprocess test data 
    print("\n[2] Preparing test data...")
    X_test, test_df = prepare_test_data('WR_test.csv', 'WR_train.csv', checkpoint)

    #  Step 3: Generate predictions 
    print("\n[3] Making predictions...")
    mean_proba, std_proba, all_preds = predict(X_test, checkpoint)

    #  Step 4: Build results DataFrame 
    print("\n[4] Results")
    print("=" * 90)

    results_df = test_df[['name', 'college_team', 'year']].copy()
    results_df['pred_r1_probability'] = mean_proba
    results_df['confidence_std']      = std_proba
    # Min and max across folds give a range on how much the models agree.
    results_df['confidence_min']      = all_preds.min(axis=0)
    results_df['confidence_max']      = all_preds.max(axis=0)

    # Use fold 1's threshold. See module docstring for why.
    threshold = checkpoint['cv_results'][0]['best_threshold']
    results_df['pred_first_round'] = (mean_proba > threshold).astype(int)

    n_r1 = results_df['pred_first_round'].sum()
    print(f"\nTotal WRs:        {len(results_df)}")
    print(f"Predicted R1:     {n_r1} ({100 * n_r1 / len(results_df):.1f}%)")
    print(f"Mean confidence:  {mean_proba.mean():.4f}")
    print(f"Threshold used:   {threshold:.3f}")

    #  Step 5: Print top 15 
    print(f"\nTop 15 by R1 probability:")
    top = results_df.nlargest(15, 'pred_r1_probability')
    for idx, (_, row) in enumerate(top.iterrows(), 1):
        symbol = "Y" if row['pred_first_round'] == 1 else " "
        print(f"  {idx:2d}. [{symbol}] {row['name']:<25} {row['college_team']:<20} "
              f"{row['pred_r1_probability']:.4f}")

    #  Step 6: Save 
    results_df.to_csv(os.path.join(output_dir, 'predictions_full.csv'), index=False)
    print(f"\nResults saved to {output_dir}/")
    print("=" * 90)


if __name__ == '__main__':
    main()
