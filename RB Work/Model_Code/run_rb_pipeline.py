import os
import sys
import numpy as np
import pandas as pd

from rb_draft_model import RBDraftPredictor


def main():

    predictor = RBDraftPredictor('TrainingData/RB_train.csv')
    X, y, df = predictor.prepare_data()

    # Output directory for results
    output_dir = os.path.join(os.path.dirname(__file__), '../Model_Output')
    os.makedirs(output_dir, exist_ok=True)


    # Grouped K-Fold (Primary Benchmark)
    gkf_results = predictor.train_and_evaluate_grouped(X, y, df)
    predictor.print_cross_validation_summary()

    # Save GKF results
    gkf_rows = []
    for r in gkf_results:
        gkf_rows.append({
            'fold': r.get('fold'),
            'test_years': r.get('test_years'),
            'train_size': r['train_size'],
            'test_size': r['test_size'],
            'accuracy': r['accuracy'],
            'f1_score': r['f1_score'],
            'roc_auc': r['roc_auc'],
            'best_threshold': r['best_threshold'],
            'true_negatives': int(r['confusion_matrix'][0, 0]),
            'false_positives': int(r['confusion_matrix'][0, 1]),
            'false_negatives': int(r['confusion_matrix'][1, 0]),
            'true_positives': int(r['confusion_matrix'][1, 1]),
        })
    gkf_csv_path = os.path.join(output_dir, 'cv_results_gkf.csv')
    pd.DataFrame(gkf_rows).to_csv(gkf_csv_path, index=False)
    print(f"Saved: {gkf_csv_path}")

    # Feature importance
    importance = predictor.feature_importance_analysis()
    if importance is not None:
        imp_df = pd.DataFrame(list(importance.items()), columns=["feature", "importance"])
        imp_csv_path = os.path.join(output_dir, 'feature_importance.csv')
        imp_df.to_csv(imp_csv_path, index=False)
        print(f"Saved: {imp_csv_path}")

    # Stratified K-Fold (Stability Check)
    skf_results = predictor.train_and_evaluate_skf(X, y)
    predictor.print_cross_validation_summary()

    # Save SKF results
    skf_rows = []
    for r in skf_results:
        skf_rows.append({
            'fold': r.get('fold'),
            'train_size': r['train_size'],
            'test_size': r['test_size'],
            'accuracy': r['accuracy'],
            'f1_score': r['f1_score'],
            'roc_auc': r['roc_auc'],
            'best_threshold': r['best_threshold'],
            'true_negatives': int(r['confusion_matrix'][0, 0]),
            'false_positives': int(r['confusion_matrix'][0, 1]),
            'false_negatives': int(r['confusion_matrix'][1, 0]),
            'true_positives': int(r['confusion_matrix'][1, 1]),
        })
    skf_csv_path = os.path.join(output_dir, 'cv_results_skf.csv')
    pd.DataFrame(skf_rows).to_csv(skf_csv_path, index=False)
    print(f"Saved: {skf_csv_path}")

    # Generate predictions for new RBs (if test data exists) AFTER model training
    test_data_path = os.path.join(os.path.dirname(__file__), '../../TestingData/RB_Test.csv')
    if os.path.exists(test_data_path):
        predictions_df = predictor.generate_predictions_for_new_rbs(test_data_path)
        if predictions_df is not None:
            pred_csv_path = os.path.join(output_dir, 'RB_Test_Predictions.csv')
            predictions_df.to_csv(pred_csv_path, index=False)
            print(f"Saved: {pred_csv_path}")

    # Grouped K-Fold (Primary Benchmark)
    gkf_results = predictor.train_and_evaluate_grouped(X, y, df)
    predictor.print_cross_validation_summary()

    # Save GKF results
    gkf_rows = []
    for r in gkf_results:
        gkf_rows.append({
            'fold': r.get('fold'),
            'test_years': r.get('test_years'),
            'train_size': r['train_size'],
            'test_size': r['test_size'],
            'accuracy': r['accuracy'],
            'f1_score': r['f1_score'],
            'roc_auc': r['roc_auc'],
            'best_threshold': r['best_threshold'],
            'true_negatives': int(r['confusion_matrix'][0, 0]),
            'false_positives': int(r['confusion_matrix'][0, 1]),
            'false_negatives': int(r['confusion_matrix'][1, 0]),
            'true_positives': int(r['confusion_matrix'][1, 1]),
        })
    gkf_csv_path = os.path.join(output_dir, 'cv_results_gkf.csv')
    pd.DataFrame(gkf_rows).to_csv(gkf_csv_path, index=False)
    print(f"Saved: {gkf_csv_path}")

    # Feature importance
    importance = predictor.feature_importance_analysis()
    if importance is not None:
        imp_df = pd.DataFrame(list(importance.items()), columns=["feature", "importance"])
        imp_csv_path = os.path.join(output_dir, 'feature_importance.csv')
        imp_df.to_csv(imp_csv_path, index=False)
        print(f"Saved: {imp_csv_path}")

    # Stratified K-Fold (Stability Check)
    skf_results = predictor.train_and_evaluate_skf(X, y)
    predictor.print_cross_validation_summary()

    # Save SKF results
    skf_rows = []
    for r in skf_results:
        skf_rows.append({
            'fold': r.get('fold'),
            'train_size': r['train_size'],
            'test_size': r['test_size'],
            'accuracy': r['accuracy'],
            'f1_score': r['f1_score'],
            'roc_auc': r['roc_auc'],
            'best_threshold': r['best_threshold'],
            'true_negatives': int(r['confusion_matrix'][0, 0]),
            'false_positives': int(r['confusion_matrix'][0, 1]),
            'false_negatives': int(r['confusion_matrix'][1, 0]),
            'true_positives': int(r['confusion_matrix'][1, 1]),
        })
    skf_csv_path = os.path.join(output_dir, 'cv_results_skf.csv')
    pd.DataFrame(skf_rows).to_csv(skf_csv_path, index=False)
    print(f"Saved: {skf_csv_path}")

    print("\n" + "=" * 60)
    print("MODEL TRAINING COMPLETE")
    print("=" * 60)

if __name__ == '__main__':
    main()
