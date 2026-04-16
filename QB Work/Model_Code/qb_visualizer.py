import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc
import warnings
import os

warnings.filterwarnings('ignore')


class QBDraftVisualizer:
    """
    Generate visualizations for the QB draft prediction model.

    Supports two evaluation strategies:
      - GKF (Grouped K-Fold): primary benchmark, chronological year groups.
      - SKF (Stratified K-Fold): stability check, random stratified splits.

    Both share identical plotting logic. The strategy label is embedded in
    chart titles and file names so outputs from each run never overwrite
    each other.
    """

    # Human-readable titles for each strategy
    STRATEGY_LABELS = {
        'gkf': 'Grouped K-Fold (Primary Benchmark)',
        'skf': 'Stratified K-Fold (Stability Check)',
    }

    def __init__(self, cv_results: list, feature_names: list,
                 output_dir: str = './output', strategy: str = 'gkf'):
        """
        Args:
            cv_results:    List of fold result dicts from QBDraftPredictor.
            feature_names: List of feature names used in the model.
            output_dir:    Directory to save plots (created if absent).
            strategy:      'gkf' or 'skf' — controls labels and sub-folder.
        """
        self.cv_results    = cv_results
        self.feature_names = feature_names
        self.strategy      = strategy.lower()
        self.output_dir    = os.path.join(output_dir, self.strategy)
        self.strategy_label = self.STRATEGY_LABELS.get(
            self.strategy, strategy.upper()
        )

        os.makedirs(self.output_dir, exist_ok=True)

        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (14, 8)

    # ── Fold-level performance ─────────────────────────────────────────

    def plot_fold_performance(self):
        """
        Line charts of Accuracy, F1-Score, and ROC-AUC across all folds.

        For GKF folds are labelled as "group_1" … "group_5".
        For SKF folds are labelled as "fold_1" … "fold_5".
        The strategy name is embedded in the chart title so the two runs
        are always visually distinguishable.
        """
        if not self.cv_results:
            print("No results to plot")
            return

        labels     = [r.get('fold_label', str(r.get('fold', i)))
                      for i, r in enumerate(self.cv_results)]
        x          = list(range(len(labels)))
        accuracies = [r['accuracy'] for r in self.cv_results]
        f1_scores  = [r['f1_score'] for r in self.cv_results]
        roc_aucs   = [r['roc_auc']  for r in self.cv_results]

        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        fig.suptitle(self.strategy_label, fontsize=14, fontweight='bold', y=1.02)

        # Accuracy
        axes[0].plot(x, accuracies, marker='o', linewidth=2, markersize=8, color='#2E86AB')
        axes[0].fill_between(x, accuracies, alpha=0.3, color='#2E86AB')
        axes[0].axhline(np.mean(accuracies), color='#2E86AB', linestyle='--',
                        linewidth=1.2, alpha=0.7, label=f'Mean {np.mean(accuracies):.3f}')
        axes[0].set_title('Accuracy Across Folds', fontsize=13, fontweight='bold')
        axes[0].set_xlabel('Fold')
        axes[0].set_ylabel('Accuracy')
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
        axes[0].legend(fontsize=8)
        axes[0].grid(True, alpha=0.3)
        axes[0].set_ylim([0, 1])

        # F1-Score
        axes[1].plot(x, f1_scores, marker='s', linewidth=2, markersize=8, color='#A23B72')
        axes[1].fill_between(x, f1_scores, alpha=0.3, color='#A23B72')
        axes[1].axhline(np.mean(f1_scores), color='#A23B72', linestyle='--',
                        linewidth=1.2, alpha=0.7, label=f'Mean {np.mean(f1_scores):.3f}')
        axes[1].set_title('F1-Score Across Folds', fontsize=13, fontweight='bold')
        axes[1].set_xlabel('Fold')
        axes[1].set_ylabel('F1-Score')
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
        axes[1].legend(fontsize=8)
        axes[1].grid(True, alpha=0.3)
        axes[1].set_ylim([0, 1])

        # ROC-AUC
        valid_roc = [(xi, r) for xi, r in zip(x, roc_aucs) if not np.isnan(r)]
        if valid_roc:
            vx, vr = zip(*valid_roc)
            axes[2].plot(vx, vr, marker='^', linewidth=2, markersize=8, color='#F18F01')
            axes[2].fill_between(vx, vr, alpha=0.3, color='#F18F01')
            mean_auc = np.mean(vr)
            axes[2].axhline(mean_auc, color='#F18F01', linestyle='--',
                            linewidth=1.2, alpha=0.7, label=f'Mean {mean_auc:.3f}')
            axes[2].legend(fontsize=8)
        axes[2].set_title('ROC-AUC Across Folds', fontsize=13, fontweight='bold')
        axes[2].set_xlabel('Fold')
        axes[2].set_ylabel('ROC-AUC')
        axes[2].set_xticks(x)
        axes[2].set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
        axes[2].grid(True, alpha=0.3)
        axes[2].set_ylim([0, 1])

        plt.tight_layout()
        out = os.path.join(self.output_dir, 'fold_performance.png')
        plt.savefig(out, dpi=300, bbox_inches='tight')
        print(f"Saved: {out}")
        plt.close()

    # ── ROC curves ────────────────────────────────────────────────────

    def plot_roc_curves(self):
        """
        One ROC curve subplot per fold, arranged in a 2-row grid.
        AUC is annotated on each subplot. Folds with only one class in
        the test set are flagged and skipped gracefully.
        """
        n_folds = len(self.cv_results)
        ncols   = (n_folds + 1) // 2
        fig, axes = plt.subplots(2, ncols, figsize=(14, 10))
        axes = axes.flatten()
        fig.suptitle(f'ROC Curves — {self.strategy_label}',
                     fontsize=13, fontweight='bold', y=1.01)

        for idx, result in enumerate(self.cv_results):
            y_test        = result['y_test']
            y_pred_proba  = result['y_pred_proba']
            fold_label    = result.get('fold_label', str(result.get('fold', idx + 1)))

            if len(np.unique(y_test)) < 2:
                axes[idx].text(0.5, 0.5, 'Only one class in test set',
                               ha='center', va='center', fontsize=9)
                axes[idx].set_title(f"{fold_label} (Skipped)", fontweight='bold')
                continue

            fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
            roc_auc_val = auc(fpr, tpr)

            axes[idx].plot(fpr, tpr, linewidth=2,
                           label=f'AUC = {roc_auc_val:.3f}', color='#2E86AB')
            axes[idx].plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
            axes[idx].set_title(fold_label, fontweight='bold')
            axes[idx].set_xlabel('False Positive Rate')
            axes[idx].set_ylabel('True Positive Rate')
            axes[idx].legend(loc='lower right', fontsize=9)
            axes[idx].grid(True, alpha=0.3)
            axes[idx].set_xlim([0, 1])
            axes[idx].set_ylim([0, 1])

        for idx in range(n_folds, len(axes)):
            axes[idx].set_visible(False)

        plt.tight_layout()
        out = os.path.join(self.output_dir, 'roc_curves.png')
        plt.savefig(out, dpi=300, bbox_inches='tight')
        print(f"Saved: {out}")
        plt.close()

    # ── Confusion matrix heatmap ───────────────────────────────────────

    def plot_aggregated_confusion_matrix(self):
        """
        Heatmap of the confusion matrix aggregated across all folds.
        Shows raw counts and row-normalised percentages side by side.
        """
        if not self.cv_results:
            print("No results to plot")
            return

        total_cm = sum(r['confusion_matrix'] for r in self.cv_results)
        cm_norm  = total_cm.astype(float) / total_cm.sum(axis=1, keepdims=True)

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle(f'Aggregated Confusion Matrix — {self.strategy_label}',
                     fontsize=13, fontweight='bold')

        labels_txt = ['Not First Round', 'First Round']

        sns.heatmap(total_cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=labels_txt, yticklabels=labels_txt,
                    ax=axes[0], cbar=False)
        axes[0].set_title('Raw Counts', fontweight='bold')
        axes[0].set_xlabel('Predicted')
        axes[0].set_ylabel('Actual')

        sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
                    xticklabels=labels_txt, yticklabels=labels_txt,
                    ax=axes[1], cbar=False, vmin=0, vmax=1)
        axes[1].set_title('Row-Normalised (Recall per Class)', fontweight='bold')
        axes[1].set_xlabel('Predicted')
        axes[1].set_ylabel('Actual')

        plt.tight_layout()
        out = os.path.join(self.output_dir, 'confusion_matrix.png')
        plt.savefig(out, dpi=300, bbox_inches='tight')
        print(f"Saved: {out}")
        plt.close()

    # ── Threshold distribution ─────────────────────────────────────────

    def plot_threshold_distribution(self):
        """
        Bar chart of the optimal classification threshold per fold.
        Useful for diagnosing whether the threshold is stable across
        folds or wildly different (which would indicate calibration issues).
        """
        if not self.cv_results:
            return

        labels     = [r.get('fold_label', str(r.get('fold', i)))
                      for i, r in enumerate(self.cv_results)]
        thresholds = [r['best_threshold'] for r in self.cv_results]

        fig, ax = plt.subplots(figsize=(10, 5))
        bars = ax.bar(labels, thresholds, color='#6B4FBB', alpha=0.75, edgecolor='black')
        ax.axhline(np.mean(thresholds), color='red', linestyle='--', linewidth=1.5,
                   label=f'Mean threshold = {np.mean(thresholds):.3f}')
        ax.axhline(0.5, color='grey', linestyle=':', linewidth=1.2,
                   label='Default threshold = 0.50')

        for bar, val in zip(bars, thresholds):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=9)

        ax.set_title(f'Optimal Classification Threshold per Fold\n{self.strategy_label}',
                     fontsize=13, fontweight='bold')
        ax.set_xlabel('Fold')
        ax.set_ylabel('Threshold')
        ax.set_ylim([0, 1.05])
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        out = os.path.join(self.output_dir, 'threshold_distribution.png')
        plt.savefig(out, dpi=300, bbox_inches='tight')
        print(f"Saved: {out}")
        plt.close()


class QBDraftAnalyzer:
    """Descriptive analysis of QB draft trends over time."""

    @staticmethod
    def analyze_draft_trends(df: pd.DataFrame, output_dir: str = './output'):
        """
        Bar charts of total QBs drafted and first-round QBs per year.
        Also computes and prints a year-by-year summary table.
        """
        trends = df.groupby('year').agg(
            total_qbs=('name', 'count'),
            first_round_qbs=('round', lambda x: (x == 1).sum())
        )

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle('QB Draft Trends (2010 - 2024)',
                     fontsize=14, fontweight='bold')

        axes[0].bar(trends.index, trends['total_qbs'],
                    color='#2E86AB', alpha=0.75, edgecolor='black')
        axes[0].set_title('Total QBs Drafted Per Year', fontweight='bold', fontsize=13)
        axes[0].set_xlabel('Year')
        axes[0].set_ylabel('Number of QBs')
        axes[0].grid(True, alpha=0.3, axis='y')

        axes[1].bar(trends.index, trends['first_round_qbs'],
                    color='#A23B72', alpha=0.75, edgecolor='black')
        axes[1].set_title('First-Round QBs Per Year', fontweight='bold', fontsize=13)
        axes[1].set_xlabel('Year')
        axes[1].set_ylabel('Number of First-Round QBs')
        axes[1].grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        out = os.path.join(output_dir, 'draft_trends.png')
        plt.savefig(out, dpi=300, bbox_inches='tight')
        print(f"Saved: {out}")
        plt.close()

        print("\nDraft Trends Summary:")
        print(trends.to_string())


if __name__ == '__main__':
    print("This module is for visualization. Import it in your main pipeline script.")