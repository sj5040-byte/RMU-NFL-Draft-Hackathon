import pandas as pd
import os
from Model_Code.qb_draft_model import QBDraftPredictor


def run_inference(train_file='Model_Code/QB_train.csv', test_file='QB_Test.csv'):
    # 1. Initialize Predictor and Prepare training context
    # We must call prepare_data to fit the LabelEncoders used during training
    predictor = QBDraftPredictor(train_file)
    X_train, y_train, df_train = predictor.prepare_data()

    # 2. Setup the models via GKF (Primary Benchmark)
    # This populates predictor.cv_results with the models and thresholds
    print("\nRestoring trained models via GKF...")
    predictor.train_and_evaluate_grouped(X_train, y_train, df_train)

    # 3. Generate predictions for the new QBs
    if not os.path.exists(test_file):
        print(f"Error: {test_file} not found.")
        return

    print(f"\nProcessing predictions for: {test_file}")

    # Load raw names to join with results later
    test_names = pd.read_csv(test_file)['name']

    # Use your built-in prediction method
    predictions_df = predictor.generate_predictions_for_new_qbs(test_file)

    if predictions_df is not None:
        # Add the names back for readability
        predictions_df.insert(0, 'name', test_names.values)

        # Sort by highest probability
        predictions_df = predictions_df.sort_values('probability_first_round', ascending=False)

        # 4. Save and Display
        output_path = 'QB_Test_Predictions.csv'
        predictions_df.to_csv(output_path, index=False)

        print("\n" + "=" * 30)
        print("FINAL PREDICTIONS")
        print("=" * 30)
        print(predictions_df.to_string(index=False))
        print(f"\nFull results saved to: {output_path}")


if __name__ == "__main__":
    run_inference()