"""
Inference script for scoring a new cohort of QB prospects.

Workflow
1. Load the training data and fit the label encoders via prepare_data.
2. Re-train the GKF models (same hyperparameter tuning + cross-validation as the main pipeline). This recovers the final fold model, which has 
   seen the most training data.
3. Score the QBs in QB_Test.csv using the last GKF fold model.
4. Save predictions to QB_Test_Predictions.csv.

Why re-train instead of loading a saved model?
XGBoost models are serialised separately from the predictor's Python
state (label encoders, feature names, thresholds). Re-training via GKF
is the simplest way to recover everything in one shot without a separate
checkpoint-loading step. Training is fast enough that this is not a
bottleneck in practice.
"""

import pandas as pd
import os
from Model_Code.qb_draft_model import QBDraftPredictor


def run_inference(train_file='Model_Code/QB_train.csv', test_file='QB_Test.csv'):
    """
    Re-train on historical data, then score the test cohort.
    """

    # Step 1: Fit preprocessing on training data 
    predictor = QBDraftPredictor(train_file)
    X_train, y_train, df_train = predictor.prepare_data()

    # Step 2: Train via GKF to recover the fold models 
    print("\nRestoring trained models via GKF...")
    predictor.train_and_evaluate_grouped(X_train, y_train, df_train)

    # Step 3: Score the new QB class 
    if not os.path.exists(test_file):
        print(f"Error: {test_file} not found.")
        return

    print(f"\nProcessing predictions for: {test_file}")

    test_names = pd.read_csv(test_file)['name']

    predictions_df = predictor.generate_predictions_for_new_qbs(test_file)

    if predictions_df is not None:
        # Re-attach names. generate_predictions_for_new_qbs returns a
        # DataFrame sorted by probability, so we use .values to align
        # with the sorted order after the fact.
        predictions_df.insert(0, 'name', test_names.values)

        predictions_df = predictions_df.sort_values('probability_first_round', ascending=False)

        # Step 4: Save and display 
        output_path = 'QB_Test_Predictions.csv'
        predictions_df.to_csv(output_path, index=False)

        print("\n" + "=" * 30)
        print("FINAL PREDICTIONS")
        print("=" * 30)
        print(predictions_df.to_string(index=False))
        print(f"\nFull results saved to: {output_path}")


if __name__ == "__main__":
    run_inference()
