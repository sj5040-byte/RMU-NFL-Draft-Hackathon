"""
Orchestrates the full QB draft prediction workflow from raw data to
saved outputs. Run this file directly to reproduce all results.

Pipeline steps
1. Data preparation
2. GKF (Grouped K-Fold) -- primary benchmark
       Chronological year groups. No temporal leakage. Honest F1.
3. SKF (Stratified K-Fold) -- stability check
       Random stratified splits. Used to detect leakage and confirm
       that GKF variance is not just fold-size noise.
4. GKF vs SKF comparison table with leakage verdict
5. Feature importance (from GKF models)
6. Draft trend analysis
"""

import os
import sys
import numpy as np
import pandas as pd

from qb_draft_model import QBDraftPredictor
from qb_visualizer import QBDraftVisualizer, QBDraftAnalyzer


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def setup_output_directory() -> str:
    """
    Create (if needed) and return the top-level output directory.
    """
    output_dir = os.path.join(os.path.dirname(__file__), 'Model_Output')
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def run_cv_strategy(predictor: QBDraftPredictor,
                    strategy: str,
                    X, y, df,
                    output_dir: str) -> dict:
    """
    Run one CV strategy end-to-end: train, evaluate, visualise, save CSV.

    Reuses the same predictor instance across GKF and SKF so both
    strategies share identical hyperparameters and feature preprocessing.

    Parameters
    ----------
    predictor : QBDraftPredictor
        Instantiated and data-prepared predictor.
    strategy : str
        'GKF' or 'SKF'.
    X : pd.DataFrame
        Feature matrix.
    y : pd.Series
        Binary target.
    df : pd.DataFrame
        Full DataFrame (used by GKF for year-based grouping).
    output_dir : str
        Directory where CSVs and plots are saved.

    Returns
    -------
    dict
        Aggregate metrics: strategy, mean_f1, std_f1, mean_acc, std_acc,
        mean_auc, n_folds.
    """
    divider = '=' * 70
    print(f"\n{divider}")
    if strategy == 'GKF':
        print("  PRIMARY BENCHMARK: Grouped K-Fold (chronological year groups)")
    else:
        print("  STABILITY CHECK: Stratified K-Fold (random stratified splits)")
    print(divider)

    if strategy == 'GKF':
        predictor.train_and_evaluate_grouped(X, y, df)
    elif strategy == 'SKF':
        predictor.train_and_evaluate_skf(X, y, n_splits=5)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    predictor.print_cross_validation_summary()

    # --- Save per-fold results to CSV ---
    rows = []
    for r in predictor.cv_results:
        rows.append({
            'strategy':        strategy,
            'fold_label':      r.get('fold_label', r.get('fold', '?')),
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
    csv_path = os.path.join(output_dir, f"cv_results_{strategy.lower()}.csv")
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"Saved: cv_results_{strategy.lower()}.csv")

    # --- Generate visualizations ---
    # Each strategy writes into its own subdirectory so GKF and SKF
    # plots never overwrite each other.
    vis = QBDraftVisualizer(
        cv_results=predictor.cv_results,
        feature_names=predictor.feature_names,
        output_dir=output_dir,
        strategy=strategy.lower()
    )
    vis.plot_fold_performance()
    vis.plot_roc_curves()
    vis.plot_aggregated_confusion_matrix()
    vis.plot_threshold_distribution()

    # --- Build summary dict ---
    f1s  = [r['f1_score'] for r in predictor.cv_results]
    accs = [r['accuracy'] for r in predictor.cv_results]
    aucs = [r['roc_auc']  for r in predictor.cv_results if not np.isnan(r['roc_auc'])]

    return {
        'strategy':  strategy,
        'mean_f1':   np.mean(f1s),
        'std_f1':    np.std(f1s),
        'mean_acc':  np.mean(accs),
        'std_acc':   np.std(accs),
        'mean_auc':  np.mean(aucs) if aucs else np.nan,
        'n_folds':   len(predictor.cv_results),
    }


def print_comparison_table(gkf_summary: dict, skf_summary: dict):
    """
    Print a side-by-side GKF vs SKF metric table and render a leakage
    interpretation verdict.
    
    Parameters
    ----------
    gkf_summary : dict
        Output of run_cv_strategy for 'GKF'.
    skf_summary : dict
        Output of run_cv_strategy for 'SKF'.
    """
    print("\n" + "=" * 70)
    print("GKF vs SKF COMPARISON")
    print("=" * 70)
    print(f"\n  {'Strategy':<30} {'Folds':>5} {'Mean F1':>9} "
          f"{'Std F1':>8} {'Mean Acc':>9} {'Mean AUC':>9}")
    print(f"  {'--' * 32}")

    for s in [gkf_summary, skf_summary]:
        tag     = 'Primary Benchmark' if s['strategy'] == 'GKF' else 'Stability Check'
        name    = f"{s['strategy']} ({tag})"
        auc_str = f"{s['mean_auc']:.4f}" if not np.isnan(s['mean_auc']) else '   N/A'
        print(f"  {name:<30} {s['n_folds']:>5} {s['mean_f1']:>9.4f} "
              f"{s['std_f1']:>8.4f} {s['mean_acc']:>9.4f} {auc_str:>9}")

# ----------------------------------------------------------------------
# Main pipeline
# ----------------------------------------------------------------------

def main():
    """
    Run the full QB draft prediction pipeline.
    """
    print("\n" + "=" * 70)
    print(" " * 15 + "QB DRAFT PREDICTION MODEL")
    print(" " * 8 + "XGBoost + Optuna | GKF Primary | SKF Stability Check")
    print("=" * 70)

    output_dir = setup_output_directory()
    print(f"\nOutput directory: {output_dir}")

    data_file = 'QB_train.csv'
    if not os.path.exists(data_file):
        print(f"\nERROR: {data_file} not found in current directory")
        sys.exit(1)

    # --- Step 1: Data preparation ---
    print("\n" + "--" * 35)
    print("STEP 1: DATA PREPARATION")
    print("--" * 35)

    predictor = QBDraftPredictor(data_file)
    X, y, df  = predictor.prepare_data()

    # --- Step 2: GKF primary benchmark ---
    print("\n" + "--" * 35)
    print("STEP 2: PRIMARY BENCHMARK -- Grouped K-Fold")
    print("--" * 35)

    gkf_summary = run_cv_strategy(predictor, 'GKF', X, y, df, output_dir)

    # --- Step 3: SKF stability check ---
    print("\n" + "--" * 35)
    print("STEP 3: STABILITY CHECK -- Stratified K-Fold")
    print("--" * 35)

    skf_summary = run_cv_strategy(predictor, 'SKF', X, y, df, output_dir)

    # --- Step 4: Compare and interpret ---
    print_comparison_table(gkf_summary, skf_summary)

    comparison_df = pd.DataFrame([gkf_summary, skf_summary])
    comparison_df.to_csv(os.path.join(output_dir, 'cv_strategy_comparison.csv'), index=False)
    print(f"\nSaved: cv_strategy_comparison.csv")

    # --- Step 5: Feature importance ---
    # SKF overwrote cv_results in memory. Re-run GKF (params already
    # tuned) to restore the GKF models before computing importance.
    print("\n" + "--" * 35)
    print("STEP 4: FEATURE IMPORTANCE (from GKF models)")
    print("--" * 35)

    predictor.train_and_evaluate_grouped(X, y, df)
    importance = predictor.feature_importance_analysis()

    if importance:
        imp_df = pd.DataFrame([
            {'feature': f, 'mean_importance': s}
            for f, s in sorted(importance.items(), key=lambda x: x[1], reverse=True)
        ])
        imp_df.to_csv(os.path.join(output_dir, 'feature_importance.csv'), index=False)
        print("Saved: feature_importance.csv")

    # --- Step 6: Draft trend analysis ---
    print("\n" + "--" * 35)
    print("STEP 5: DRAFT TREND ANALYSIS")
    print("--" * 35)
    QBDraftAnalyzer.analyze_draft_trends(df, output_dir)

    # --- Dataset statistics ---
    print("\n" + "=" * 70)
    print("DATASET STATISTICS")
    print("=" * 70)
    print(f"  Total QBs:       {len(df)}")
    print(f"  First-round QBs: {(df['round'] == 1).sum()} ({(df['round'] == 1).mean() * 100:.1f}%)")
    print(f"  Years covered:   {df['year'].min()}-{df['year'].max()}")

    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE")
    print("=" * 70)


if __name__ == '__main__':
    main()
