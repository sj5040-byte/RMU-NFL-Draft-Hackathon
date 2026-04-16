"""
QB Draft Prediction Pipeline
==============================

Orchestrates the full ML workflow:

  Step 1 — Data preparation
  Step 2 — PRIMARY: Grouped K-Fold (GKF)
               Train, evaluate, visualise, save CSVs.
               GKF is the honest benchmark: chronological year groups,
               no temporal leakage, stable F1 per fold.
  Step 3 — STABILITY CHECK: Stratified K-Fold (SKF)
               Same hyperparameters, random stratified splits.
               Used to detect leakage (SKF >> GKF = leakage present)
               and to confirm that GKF variance is not just fold-size noise.
  Step 4 — GKF vs SKF comparison table with leakage interpretation.
  Step 5 — Feature importance (from GKF models).
  Step 6 — Draft trend analysis.

Requirements:
    pandas, numpy, xgboost, optuna, scikit-learn, matplotlib, seaborn
"""

import os
import sys
import numpy as np
import pandas as pd

from qb_draft_model import QBDraftPredictor
from qb_visualizer import QBDraftVisualizer, QBDraftAnalyzer


# ── Helpers ───────────────────────────────────────────────────────────────────

def setup_output_directory() -> str:
    """Create and return the top-level output directory."""
    output_dir = os.path.join(os.path.dirname(__file__), 'Model_Output')
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def run_cv_strategy(predictor: QBDraftPredictor,
                    strategy: str,
                    X, y, df,
                    output_dir: str) -> dict:
    """
    Run a single CV strategy, print its summary, save a results CSV,
    generate visualizations, and return a summary metrics dict.

    Args:
        predictor:  QBDraftPredictor instance (reused across strategies).
        strategy:   'GKF' or 'SKF'.
        X, y, df:   Prepared feature matrix, target, and full DataFrame.
        output_dir: Top-level output directory.

    Returns:
        dict with strategy name and aggregate metrics.
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

    # Save per-fold results CSV
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

    # Visualizations
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
    Print a side-by-side comparison of GKF and SKF metrics with a
    leakage interpretation verdict.
    """
    print("\n" + "=" * 70)
    print("GKF vs SKF COMPARISON")
    print("=" * 70)
    print(f"\n  {'Strategy':<30} {'Folds':>5} {'Mean F1':>9} "
          f"{'Std F1':>8} {'Mean Acc':>9} {'Mean AUC':>9}")
    print(f"  {'─' * 65}")

    for s in [gkf_summary, skf_summary]:
        tag     = 'Primary Benchmark' if s['strategy'] == 'GKF' else 'Stability Check'
        name    = f"{s['strategy']} ({tag})"
        auc_str = f"{s['mean_auc']:.4f}" if not np.isnan(s['mean_auc']) else '   N/A'
        print(f"  {name:<30} {s['n_folds']:>5} {s['mean_f1']:>9.4f} "
              f"{s['std_f1']:>8.4f} {s['mean_acc']:>9.4f} {auc_str:>9}")

    # Leakage verdict
    f1_gap = skf_summary['mean_f1'] - gkf_summary['mean_f1']
    print(f"\n  SKF - GKF mean F1 gap: {f1_gap:+.4f}")
    print(f"\n  Interpretation:")

    if f1_gap > 0.08:
        print(f"    SKF F1 is substantially higher than GKF ({f1_gap:.4f} gap).")
        print(f"    This indicates temporal leakage is inflating SKF scores.")
        print(f"    GKF ({gkf_summary['mean_f1']:.4f}) is the honest estimate of real-world performance.")
    elif f1_gap > 0.03:
        print(f"    Moderate gap ({f1_gap:.4f}). Some mild leakage may be present.")
        print(f"    Prefer GKF as the reported benchmark; SKF serves as an upper bound.")
    else:
        print(f"    SKF ≈ GKF (gap = {f1_gap:.4f}). Temporal leakage is minimal.")
        print(f"    GKF remains the primary benchmark.")
        print(f"    SKF std ({skf_summary['std_f1']:.4f}) vs GKF std ({gkf_summary['std_f1']:.4f})")
        print(f"    confirms that GKF variance reflects era-level difficulty, not noise.")

    print(f"\n  Reported model performance (GKF): "
          f"F1 = {gkf_summary['mean_f1']:.4f} ± {gkf_summary['std_f1']:.4f}, "
          f"AUC = {gkf_summary['mean_auc']:.4f}")


# ── Main pipeline ─────────────────────────────────────────────────────────────

def main():
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

    # ── Step 1: Data preparation ──────────────────────────────────────
    print("\n" + "−" * 70)
    print("STEP 1: DATA PREPARATION")
    print("−" * 70)

    predictor = QBDraftPredictor(data_file)
    X, y, df  = predictor.prepare_data()

    # ── Step 2: GKF — Primary benchmark ──────────────────────────────
    print("\n" + "−" * 70)
    print("STEP 2: PRIMARY BENCHMARK — Grouped K-Fold")
    print("−" * 70)

    gkf_summary = run_cv_strategy(predictor, 'GKF', X, y, df, output_dir)

    # ── Step 3: SKF — Stability check ────────────────────────────────
    print("\n" + "−" * 70)
    print("STEP 3: STABILITY CHECK — Stratified K-Fold")
    print("−" * 70)

    skf_summary = run_cv_strategy(predictor, 'SKF', X, y, df, output_dir)

    # ── Step 4: Comparison and leakage verdict ────────────────────────
    print_comparison_table(gkf_summary, skf_summary)

    # Save comparison CSV
    comparison_df = pd.DataFrame([gkf_summary, skf_summary])
    comparison_df.to_csv(os.path.join(output_dir, 'cv_strategy_comparison.csv'), index=False)
    print(f"\nSaved: cv_strategy_comparison.csv")

    # ── Step 5: Feature importance (re-run GKF to restore its models) ─
    # GKF models are still in memory from Step 2 when predictor.cv_results
    # was populated by train_and_evaluate_grouped. Step 3 overwrote those
    # with SKF results, so we run a brief GKF pass to recover them.
    print("\n" + "−" * 70)
    print("STEP 4: FEATURE IMPORTANCE (from GKF models)")
    print("−" * 70)

    # Re-run GKF silently to recover models (params already tuned, no retuning)
    predictor.train_and_evaluate_grouped(X, y, df)
    importance = predictor.feature_importance_analysis()

    if importance:
        imp_df = pd.DataFrame([
            {'feature': f, 'mean_importance': s}
            for f, s in sorted(importance.items(), key=lambda x: x[1], reverse=True)
        ])
        imp_df.to_csv(os.path.join(output_dir, 'feature_importance.csv'), index=False)
        print("Saved: feature_importance.csv")

    # ── Step 6: Draft trend analysis ─────────────────────────────────
    print("\n" + "−" * 70)
    print("STEP 5: DRAFT TREND ANALYSIS")
    print("−" * 70)
    QBDraftAnalyzer.analyze_draft_trends(df, output_dir)

    # ── Dataset statistics ─────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("DATASET STATISTICS")
    print("=" * 70)
    print(f"  Total QBs:       {len(df)}")
    print(f"  First-round QBs: {(df['round'] == 1).sum()} ({(df['round'] == 1).mean() * 100:.1f}%)")
    print(f"  Years covered:   {df['year'].min()}–{df['year'].max()}")

    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE")
    print("=" * 70)


if __name__ == '__main__':
    main()