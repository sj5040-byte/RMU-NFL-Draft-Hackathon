"""
Orchestrates the full WR draft prediction training workflow.

Pipeline steps
1. Data preparation (printed summary only -- no output saved)
2. GKF (Grouped K-Fold) -- primary benchmark
       Trains five independent neural networks on chronological year
       blocks. Saves cv_results_gkf.csv.
3. SKF (Stratified K-Fold) -- stability check
       Trains five independent neural networks on random stratified
       splits. Saves cv_results_skf.csv.
4. GKF vs SKF side-by-side comparison. Saves strategy_comparison.csv.
5. Gradient-based feature importance from GKF models.
       Saves feature_importance.csv.
6. Model checkpoint from GKF models.
       Saves model_checkpoint.pkl for use by wr_neural_inference.py.

"""

import os
import sys
import time
import numpy as np
import pandas as pd
import pickle

from wr_neural_model import WRDraftPredictor


def setup_output_directory() -> str:
    """
    Create (if needed) and return the output directory.
    """
    output_dir = os.path.join(os.path.dirname(__file__), 'Model_Output')
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def run_cv_strategy(predictor: WRDraftPredictor,
                    strategy_name: str,
                    X, y, df,
                    output_dir: str) -> dict:
    """
    Run one CV strategy end-to-end: train, evaluate, print summary,
    save results CSV, and return aggregate metrics.

    The elapsed wall-clock time is measured and included in the summary
    dict. Neural network training is significantly slower than XGBoost,
    so timing helps gauge the cost of each strategy at a glance.
    """
    start_time = time.time()
    divider    = '=' * 70

    print(f"\n{divider}")
    print(f"  STRATEGY: {strategy_name.upper()}")
    print(divider)

    if strategy_name.lower() == 'gkf':
        predictor.evaluate_gkf(X, y, df, n_splits=5)
    elif strategy_name.lower() == 'skf':
        predictor.evaluate_skf(X, y, n_splits=5)
    else:
        raise ValueError(f"Unknown strategy: {strategy_name}")

    predictor.print_cross_validation_summary()

    # --- Save per-fold results CSV ---
    rows = []
    for r in predictor.cv_results:
        rows.append({
            'strategy':        strategy_name,
            'fold_label':      r.get('fold_label', '?'),
            'train_size':      r['train_size'],
            'test_size':       r['test_size'],
            'accuracy':        r['accuracy'],
            'f1_score':        r['f1_score'],
            'roc_auc':         r['roc_auc'],
            'best_threshold':  r['best_threshold'],
            'true_negatives':  int(r['confusion_matrix'][0, 0]),
            'false_positives': int(r['confusion_matrix'][0, 1]),
            'false_negatives': int(r['confusion_matrix'][1, 0]),
            'true_positives':  int(r['confusion_matrix'][1, 1]),
        })
    csv_path = os.path.join(output_dir, f"cv_results_{strategy_name}.csv")
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"Saved: cv_results_{strategy_name}.csv")

    # --- Build aggregate summary ---
    f1s  = [r['f1_score'] for r in predictor.cv_results]
    accs = [r['accuracy'] for r in predictor.cv_results]
    aucs = [r['roc_auc']  for r in predictor.cv_results if not np.isnan(r['roc_auc'])]

    elapsed = time.time() - start_time

    return {
        'strategy':        strategy_name,
        'mean_f1':         np.mean(f1s),
        'std_f1':          np.std(f1s),
        'mean_acc':        np.mean(accs),
        'std_acc':         np.std(accs),
        'mean_auc':        np.mean(aucs) if aucs else np.nan,
        'std_auc':         np.std(aucs)  if aucs else np.nan,
        'n_folds':         len(predictor.cv_results),
        'elapsed_seconds': elapsed,
    }


def print_strategy_comparison(summaries: list):
    """
    Print a side-by-side table comparing GKF and SKF metrics.
    """
    print(f"\n{'=' * 90}")
    print("CROSS-VALIDATION STRATEGY COMPARISON")
    print(f"{'=' * 90}")
    print(f"\n{'Strategy':<20} {'F1 Score':>18} {'Accuracy':>18} "
          f"{'AUC':>15} {'Time (sec)':>12}")
    print(f"{'-' * 90}")

    for s in summaries:
        f1_str   = f"{s['mean_f1']:.4f}  (+/-{s['std_f1']:.4f})"
        acc_str  = f"{s['mean_acc']:.4f}  (+/-{s['std_acc']:.4f})"
        auc_str  = f"{s['mean_auc']:.4f}" if not np.isnan(s['mean_auc']) else "    N/A"
        time_str = f"{s['elapsed_seconds']:.1f}s"
        print(f"{s['strategy'].upper():<20} {f1_str:>18} {acc_str:>18} "
              f"{auc_str:>15} {time_str:>12}")


def main():
    """
    1.  Set up the output directory.
    2.  Load data via a base predictor for statistics printing.
    3.  GKF: instantiate a fresh predictor, train, save CSV.
    4.  SKF: instantiate a fresh predictor, train, save CSV.
    5.  Print side-by-side comparison. Save strategy_comparison.csv.
    6.  Gradient-based feature importance from GKF models.
        Save feature_importance.csv.
    7.  Pickle the GKF checkpoint (models + scalers + encoders).
        Save model_checkpoint.pkl.
    8.  Print dataset statistics.
    """
    print("\n" + "=" * 90)
    print(" " * 20 + "WR DRAFT PREDICTION MODEL")
    print(" " * 15 + "PyTorch Deep Learning | Two CV Strategies Compared")
    print("=" * 90)

    output_dir = setup_output_directory()
    print(f"\nOutput directory: {output_dir}")

    data_file = 'WR_train.csv'
    if not os.path.exists(data_file):
        print(f"\nERROR: {data_file} not found in current directory")
        sys.exit(1)

    # --- Step 1: Data preparation (summary only) ---
    print("\n" + "-" * 90)
    print("STEP 1: DATA PREPARATION")
    print("-" * 90)

    # predictor_base is used only for the data summary print and
    # dataset statistics at the end. It is not trained.
    predictor_base = WRDraftPredictor(data_file)
    X, y, df       = predictor_base.prepare_data()

    # --- Step 2: Run both CV strategies ---
    print("\n" + "-" * 90)
    print("STEP 2: EVALUATION -- TWO CROSS-VALIDATION STRATEGIES")
    print("-" * 90)
    print("\nEach strategy trains independent neural networks.\n")

    summaries = []

    # Strategy A: GKF -- primary benchmark.
    # Fresh predictor so weights and cv_results are independent.
    print("\n" + "=" * 90)
    print("RUNNING STRATEGY 1 / 2: GKF (Grouped K-Fold)")
    print("=" * 90)
    predictor_gkf = WRDraftPredictor(data_file)
    X_gkf, y_gkf, df_gkf = predictor_gkf.prepare_data()
    summary_gkf = run_cv_strategy(predictor_gkf, 'gkf', X_gkf, y_gkf, df_gkf, output_dir)
    summaries.append(summary_gkf)

    # Strategy B: SKF -- stability check.
    # Another fresh predictor, completely independent of predictor_gkf.
    print("\n" + "=" * 90)
    print("RUNNING STRATEGY 2 / 2: SKF (Stratified K-Fold)")
    print("=" * 90)
    predictor_skf = WRDraftPredictor(data_file)
    X_skf, y_skf, df_skf = predictor_skf.prepare_data()
    summary_skf = run_cv_strategy(predictor_skf, 'skf', X_skf, y_skf, df_skf, output_dir)
    summaries.append(summary_skf)

    # --- Step 3: Compare strategies ---
    print_strategy_comparison(summaries)

    comparison_df = pd.DataFrame(summaries)
    comparison_df.to_csv(os.path.join(output_dir, 'strategy_comparison.csv'), index=False)
    print(f"\nSaved: strategy_comparison.csv")

    # --- Step 4: Feature importance from GKF models ---
    print("\n" + "-" * 90)
    print("STEP 3: FEATURE IMPORTANCE (from GKF primary benchmark)")
    print("-" * 90)

    importance = predictor_gkf.get_feature_importance_from_gradients()
    if importance:
        imp_df = pd.DataFrame([
            {'feature': f, 'mean_importance': s}
            for f, s in sorted(importance.items(), key=lambda x: x[1], reverse=True)
        ])
        imp_df.to_csv(os.path.join(output_dir, 'feature_importance.csv'), index=False)
        print("Saved: feature_importance.csv")

    # --- Step 5: Save GKF checkpoint for inference ---
    # The checkpoint bundles everything wr_neural_inference.py needs:
    # the trained models (one per fold), their fitted scalers, the
    # feature name list, and the label encoders.
    print("\n" + "-" * 90)
    print("STEP 4: SAVE MODEL CHECKPOINT")
    print("-" * 90)

    checkpoint = {
        'cv_strategy':    predictor_gkf.cv_strategy,
        'cv_results':     predictor_gkf.cv_results,
        'feature_names':  predictor_gkf.feature_names,
        'label_encoders': predictor_gkf.label_encoders,
    }
    checkpoint_path = os.path.join(output_dir, 'model_checkpoint.pkl')
    with open(checkpoint_path, 'wb') as f:
        pickle.dump(checkpoint, f)
    print(f"Saved model checkpoint to: model_checkpoint.pkl")

    # --- Dataset statistics ---
    print("\n" + "=" * 90)
    print("DATASET STATISTICS")
    print("=" * 90)
    print(f"  Total WRs:       {len(df)}")
    print(f"  First-round WRs: {(df['round'] == 1).sum()} ({(df['round'] == 1).mean() * 100:.1f}%)")
    print(f"  Years covered:   {df['year'].min()}-{df['year'].max()}")
    print(f"  Total features:  {len(X.columns)}")

    print(f"\n{'=' * 90}")
    print("TRAINING PIPELINE COMPLETE")
    print(f"{'=' * 90}\n")


if __name__ == '__main__':
    main()