"""
run_rb_pipeline.py

1. Data preparation
2. GKF (Grouped K-Fold) -- primary benchmark
       Chronological year groups. No temporal leakage. Honest F1.
       Saves cv_results_gkf.csv.
3. Feature importance from GKF models.
       Saves feature_importance.csv.
4. SKF (Stratified K-Fold) -- stability check
       Random stratified splits. Saves cv_results_skf.csv.
5. GKF vs SKF strategy comparison with leakage interpretation.
       Saves cv_strategy_comparison.csv.
6. RB Test predictions (if test data file exists).
       Saves RB_Test_Predictions.csv.
"""

import os
import sys
import numpy as np
import pandas as pd

from rb_draft_model import RBDraftPredictor
from rb_visualizer import RBDraftAnalyzer, RBDraftVisualizer
from rb_visualizer import RBDraftAnalyzer, RBDraftVisualizer


def _build_summary(predictor: RBDraftPredictor, strategy: str) -> dict:
    """
    Build an aggregate summary dict from the predictor's current
    cv_results.
    Called immediately after each CV strategy completes, before
    cv_results gets overwritten by the next strategy.
    """
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
    Print a side-by-side GKF vs SKF metric table and a leakage verdict.
    Verdict rules
    -------------
    gap > 0.08  ->  SKF substantially inflated. Trust GKF only.
    gap > 0.03  ->  Moderate inflation. GKF is preferred; SKF is an
                    upper bound.
    gap <= 0.03 ->  Leakage minimal. GKF is the primary benchmark and
                    SKF tightens the confidence interval.
    """
    print("\n" + "=" * 70)
    print("GKF vs SKF COMPARISON")
    print("=" * 70)
    print(f"\n  {'Strategy':<30} {'Folds':>5} {'Mean F1':>9} "
          f"{'Std F1':>8} {'Mean Acc':>9} {'Mean AUC':>9}")
    print(f"  {'-' * 65}")

    for s in [gkf_summary, skf_summary]:
        tag     = 'Primary Benchmark' if s['strategy'] == 'GKF' else 'Stability Check'
        name    = f"{s['strategy']} ({tag})"
        auc_str = f"{s['mean_auc']:.4f}" if not np.isnan(s['mean_auc']) else '   N/A'
        print(f"  {name:<30} {s['n_folds']:>5} {s['mean_f1']:>9.4f} "
              f"{s['std_f1']:>8.4f} {s['mean_acc']:>9.4f} {auc_str:>9}")

    f1_gap = skf_summary['mean_f1'] - gkf_summary['mean_f1']
    print(f"\n  SKF - GKF mean F1 gap: {f1_gap:+.4f}")
    print(f"\n  Interpretation:")

    if f1_gap > 0.08:
        print(f"    SKF F1 is substantially higher than GKF ({f1_gap:.4f} gap).")
        print(f"    Temporal leakage is inflating SKF scores.")
        print(f"    GKF ({gkf_summary['mean_f1']:.4f}) is the honest real-world estimate.")
    elif f1_gap > 0.03:
        print(f"    Moderate gap ({f1_gap:.4f}). Mild leakage may be present.")
        print(f"    Prefer GKF as the reported benchmark; SKF is an upper bound.")
    else:
        print(f"    SKF ~= GKF (gap = {f1_gap:.4f}). Temporal leakage is minimal.")
        print(f"    GKF remains the primary benchmark.")
        print(f"    SKF std ({skf_summary['std_f1']:.4f}) vs GKF std ({gkf_summary['std_f1']:.4f})")
        print(f"    confirms GKF variance reflects era-level difficulty, not noise.")

    print(f"\n  Reported model performance (GKF): "
          f"F1 = {gkf_summary['mean_f1']:.4f} +/- {gkf_summary['std_f1']:.4f}, "
          f"AUC = {gkf_summary['mean_auc']:.4f}")


def main():
    """
    Run the full RB draft prediction pipeline.
    1.  Initialise predictor and prepare data.
    2.  GKF: train, evaluate, print summary, save cv_results_gkf.csv.
    3.  Feature importance: compute from GKF models, save
        feature_importance.csv. Must run before SKF because SKF
        overwrites cv_results in memory.
    4.  SKF: train, evaluate, print summary, save cv_results_skf.csv.
    5.  Strategy comparison: print table, save
        cv_strategy_comparison.csv.
    6.  Predictions: score RB_Test.csv if it exists, save
        RB_Test_Predictions.csv.TrainingData
    """
    print("\n" + "=" * 70)
    print(" " * 12 + "RB DRAFT PREDICTION MODEL")
    print(" " * 5 + "XGBoost + Optuna | GKF Primary | SKF Stability Check")
    print("=" * 70)

    script_dir = os.path.dirname(__file__)
    project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))

    # Output directory sits one level up from the script location.
    output_dir = os.path.join(script_dir, '../Model_Output')
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nOutput directory: {output_dir}")

    # --- Step 1: Data preparation ---
    print("\n" + "--" * 35)
    print("STEP 1: DATA PREPARATION")
    print("--" * 35)

    train_data_path = os.path.join(project_root, 'TrainingData', 'RB_train.csv')
    predictor = RBDraftPredictor(train_data_path)
    X, y, df  = predictor.prepare_data()

    # --- Step 2: GKF primary benchmark ---
    print("\n" + "--" * 35)
    print("STEP 2: PRIMARY BENCHMARK -- Grouped K-Fold")
    print("--" * 35)

    gkf_results = predictor.train_and_evaluate_grouped(X, y, df)
    predictor.print_cross_validation_summary()

    # Capture GKF summary before SKF overwrites cv_results.
    gkf_summary = _build_summary(predictor, 'GKF')

    # Save GKF per-fold results.
    gkf_rows = []
    for r in gkf_results:
        gkf_rows.append({
            'fold':            r.get('fold'),
            'test_years':      r.get('test_years'),
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
    gkf_csv_path = os.path.join(output_dir, 'cv_results_gkf.csv')
    pd.DataFrame(gkf_rows).to_csv(gkf_csv_path, index=False)
    print(f"Saved: {gkf_csv_path}")

    gkf_viz = RBDraftVisualizer(
        cv_results=gkf_results,
        feature_names=predictor.feature_names,
        output_dir=output_dir,
        strategy='gkf'
    )
    gkf_viz.plot_fold_performance()
    gkf_viz.plot_roc_curves()
    gkf_viz.plot_aggregated_confusion_matrix()
    gkf_viz.plot_threshold_distribution()

    gkf_viz = RBDraftVisualizer(
        cv_results=gkf_results,
        feature_names=predictor.feature_names,
        output_dir=output_dir,
        strategy='gkf'
    )
    gkf_viz.plot_fold_performance()
    gkf_viz.plot_roc_curves()
    gkf_viz.plot_aggregated_confusion_matrix()
    gkf_viz.plot_threshold_distribution()

    # --- Step 3: Feature importance (must run before SKF) ---
    # GKF models are still in predictor.cv_results at this point.
    # SKF will overwrite them in the next step.
    print("\n" + "--" * 35)
    print("STEP 3: FEATURE IMPORTANCE (from GKF models)")
    print("--" * 35)

    importance = predictor.feature_importance_analysis()
    if importance is not None:
        imp_df = pd.DataFrame(
            list(importance.items()),
            columns=["feature", "importance"]
        )
        imp_csv_path = os.path.join(output_dir, 'feature_importance.csv')
        imp_df.to_csv(imp_csv_path, index=False)
        print(f"Saved: {imp_csv_path}")

    # --- Step 4: SKF stability check ---
    print("\n" + "--" * 35)
    print("STEP 4: STABILITY CHECK -- Stratified K-Fold")
    print("--" * 35)

    skf_results = predictor.train_and_evaluate_skf(X, y)
    predictor.print_cross_validation_summary()

    # Capture SKF summary immediately after it runs.
    skf_summary = _build_summary(predictor, 'SKF')

    # Save SKF per-fold results.
    skf_rows = []
    for r in skf_results:
        skf_rows.append({
            'fold':            r.get('fold'),
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
    skf_csv_path = os.path.join(output_dir, 'cv_results_skf.csv')
    pd.DataFrame(skf_rows).to_csv(skf_csv_path, index=False)
    print(f"Saved: {skf_csv_path}")

    skf_viz = RBDraftVisualizer(
        cv_results=skf_results,
        feature_names=predictor.feature_names,
        output_dir=output_dir,
        strategy='skf'
    )
    skf_viz.plot_fold_performance()
    skf_viz.plot_roc_curves()
    skf_viz.plot_aggregated_confusion_matrix()
    skf_viz.plot_threshold_distribution()

    RBDraftAnalyzer.analyze_draft_trends(df, output_dir=output_dir)

    skf_viz = RBDraftVisualizer(
        cv_results=skf_results,
        feature_names=predictor.feature_names,
        output_dir=output_dir,
        strategy='skf'
    )
    skf_viz.plot_fold_performance()
    skf_viz.plot_roc_curves()
    skf_viz.plot_aggregated_confusion_matrix()
    skf_viz.plot_threshold_distribution()

    RBDraftAnalyzer.analyze_draft_trends(df, output_dir=output_dir)

    # --- Step 5: Strategy comparison ---
    print("\n" + "--" * 35)
    print("STEP 5: STRATEGY COMPARISON")
    print("--" * 35)

    print_comparison_table(gkf_summary, skf_summary)

    comparison_df = pd.DataFrame([gkf_summary, skf_summary])
    comparison_csv_path = os.path.join(output_dir, 'cv_strategy_comparison.csv')
    comparison_df.to_csv(comparison_csv_path, index=False)
    print(f"\nSaved: {comparison_csv_path}")

    # --- Step 6: Predictions on new RBs ---
    # Predictions use the SKF-trained models because SKF is the last
    # strategy that ran and its models are now in cv_results. If you
    # want GKF-based predictions, re-run GKF before calling this.
    print("\n" + "--" * 35)
    print("STEP 6: PREDICTIONS FOR NEW RBs")
    print("--" * 35)

    test_data_path = os.path.join(project_root, 'TestingData', 'RB_Test.csv')
    if os.path.exists(test_data_path):
        predictions_df = predictor.generate_predictions_for_new_rbs(test_data_path)
        if predictions_df is not None:
            pred_csv_path = os.path.join(output_dir, 'RB_Test_Predictions.csv')
            predictions_df.to_csv(pred_csv_path, index=False)
            print(f"Saved: {pred_csv_path}")
    else:
        print(f"Test data not found at: {test_data_path}")
        print("Skipping prediction step.")

    print("\n" + "=" * 70)
    print("MODEL TRAINING COMPLETE")
    print("=" * 70)


if __name__ == '__main__':
    main()