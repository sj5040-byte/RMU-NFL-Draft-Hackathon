import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import auc, roc_curve

warnings.filterwarnings('ignore')


class RBDraftVisualizer:
	"""Generate visualizations for RB draft model evaluation."""

	STRATEGY_LABELS = {
		'gkf': 'Grouped K-Fold (Primary Benchmark)',
		'skf': 'Stratified K-Fold (Stability Check)',
	}

	def __init__(self, cv_results: list, feature_names: list,
				 output_dir: str = './output', strategy: str = 'gkf'):
		self.cv_results = cv_results
		self.feature_names = feature_names
		self.strategy = strategy.lower()
		self.output_dir = os.path.join(output_dir, self.strategy)
		self.strategy_label = self.STRATEGY_LABELS.get(self.strategy, strategy.upper())

		os.makedirs(self.output_dir, exist_ok=True)

		sns.set_style('whitegrid')
		plt.rcParams['figure.figsize'] = (14, 8)

	def plot_fold_performance(self):
		if not self.cv_results:
			print('No results to plot')
			return

		labels = [r.get('fold_label', str(r.get('fold', i))) for i, r in enumerate(self.cv_results)]
		x = list(range(len(labels)))
		accuracies = [r['accuracy'] for r in self.cv_results]
		f1_scores = [r['f1_score'] for r in self.cv_results]
		roc_aucs = [r['roc_auc'] for r in self.cv_results]

		fig, axes = plt.subplots(1, 3, figsize=(16, 5))
		fig.suptitle(self.strategy_label, fontsize=14, fontweight='bold', y=1.02)

		axes[0].plot(x, accuracies, marker='o', linewidth=2, markersize=8, color='#2E86AB')
		axes[0].fill_between(x, accuracies, alpha=0.3, color='#2E86AB')
		axes[0].axhline(np.mean(accuracies), color='#2E86AB', linestyle='--', linewidth=1.2,
						alpha=0.7, label=f'Mean {np.mean(accuracies):.3f}')
		axes[0].set_title('Accuracy Across Folds', fontsize=13, fontweight='bold')
		axes[0].set_xlabel('Fold')
		axes[0].set_ylabel('Accuracy')
		axes[0].set_xticks(x)
		axes[0].set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
		axes[0].legend(fontsize=8)
		axes[0].grid(True, alpha=0.3)
		axes[0].set_ylim([0, 1])

		axes[1].plot(x, f1_scores, marker='s', linewidth=2, markersize=8, color='#A23B72')
		axes[1].fill_between(x, f1_scores, alpha=0.3, color='#A23B72')
		axes[1].axhline(np.mean(f1_scores), color='#A23B72', linestyle='--', linewidth=1.2,
						alpha=0.7, label=f'Mean {np.mean(f1_scores):.3f}')
		axes[1].set_title('F1-Score Across Folds', fontsize=13, fontweight='bold')
		axes[1].set_xlabel('Fold')
		axes[1].set_ylabel('F1-Score')
		axes[1].set_xticks(x)
		axes[1].set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
		axes[1].legend(fontsize=8)
		axes[1].grid(True, alpha=0.3)
		axes[1].set_ylim([0, 1])

		valid_roc = [(xi, r) for xi, r in zip(x, roc_aucs) if not np.isnan(r)]
		if valid_roc:
			vx, vr = zip(*valid_roc)
			axes[2].plot(vx, vr, marker='^', linewidth=2, markersize=8, color='#F18F01')
			axes[2].fill_between(vx, vr, alpha=0.3, color='#F18F01')
			mean_auc = np.mean(vr)
			axes[2].axhline(mean_auc, color='#F18F01', linestyle='--', linewidth=1.2,
							alpha=0.7, label=f'Mean {mean_auc:.3f}')
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
		print(f'Saved: {out}')
		plt.close()

	def plot_roc_curves(self):
		if not self.cv_results:
			print('No results to plot')
			return

		n_folds = len(self.cv_results)
		ncols = max(1, (n_folds + 1) // 2)
		fig, axes = plt.subplots(2, ncols, figsize=(14, 10))
		axes = np.array(axes).reshape(-1)
		fig.suptitle(f'ROC Curves - {self.strategy_label}', fontsize=13, fontweight='bold', y=1.01)

		for idx, result in enumerate(self.cv_results):
			y_test = result['y_test']
			y_pred_proba = result['y_pred_proba']
			fold_label = result.get('fold_label', str(result.get('fold', idx + 1)))

			if len(np.unique(y_test)) < 2:
				axes[idx].text(0.5, 0.5, 'Only one class in test set', ha='center', va='center', fontsize=9)
				axes[idx].set_title(f'{fold_label} (Skipped)', fontweight='bold')
				continue

			fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
			roc_auc_val = auc(fpr, tpr)

			axes[idx].plot(fpr, tpr, linewidth=2, label=f'AUC = {roc_auc_val:.3f}', color='#2E86AB')
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
		print(f'Saved: {out}')
		plt.close()

	def plot_aggregated_confusion_matrix(self):
		if not self.cv_results:
			print('No results to plot')
			return

		total_cm = sum(r['confusion_matrix'] for r in self.cv_results)
		cm_norm = total_cm.astype(float) / total_cm.sum(axis=1, keepdims=True)

		fig, axes = plt.subplots(1, 2, figsize=(12, 5))
		fig.suptitle(f'Aggregated Confusion Matrix - {self.strategy_label}', fontsize=13, fontweight='bold')
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
		axes[1].set_title('Row-Normalized (Recall per Class)', fontweight='bold')
		axes[1].set_xlabel('Predicted')
		axes[1].set_ylabel('Actual')

		plt.tight_layout()
		out = os.path.join(self.output_dir, 'confusion_matrix.png')
		plt.savefig(out, dpi=300, bbox_inches='tight')
		print(f'Saved: {out}')
		plt.close()

	def plot_threshold_distribution(self):
		if not self.cv_results:
			return

		labels = [r.get('fold_label', str(r.get('fold', i))) for i, r in enumerate(self.cv_results)]
		thresholds = [r['best_threshold'] for r in self.cv_results]

		fig, ax = plt.subplots(figsize=(10, 5))
		bars = ax.bar(labels, thresholds, color='#6B4FBB', alpha=0.75, edgecolor='black')
		ax.axhline(np.mean(thresholds), color='red', linestyle='--', linewidth=1.5,
				   label=f'Mean threshold = {np.mean(thresholds):.3f}')
		ax.axhline(0.5, color='grey', linestyle=':', linewidth=1.2, label='Default threshold = 0.50')

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
		print(f'Saved: {out}')
		plt.close()


class RBDraftAnalyzer:
	"""Descriptive analysis of RB draft trends over time."""

	@staticmethod
	def analyze_draft_trends(df: pd.DataFrame, output_dir: str = './output'):
		trends = df.groupby('year').agg(
			total_rbs=('name', 'count'),
			first_round_rbs=('round', lambda x: (x == 1).sum())
		)

		fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor='black')
		fig.suptitle('RB Draft Trends (2010 - 2024)', fontsize=16, fontweight='bold',
					 color='white', x=0.02, ha='left')

		for ax in axes:
			ax.set_facecolor('black')
			ax.tick_params(colors='#d9d9d9', labelsize=9)
			for spine in ax.spines.values():
				spine.set_color('#444444')
			ax.grid(True, axis='y', linestyle='--', linewidth=0.7, alpha=0.25, color='#aaaaaa')
			ax.set_axisbelow(True)

		bars0 = axes[0].bar(trends.index, trends['total_rbs'], color='#ff6a00', alpha=0.95,
							 edgecolor='#ff8c42', linewidth=0.8, width=0.62)
		axes[0].set_title('Total RBs Drafted Per Year', fontweight='bold', fontsize=13, color='white')
		axes[0].set_xlabel('Year', color='#d9d9d9')
		axes[0].set_ylabel('Number of RBs', color='#d9d9d9')
		axes[0].set_xticks(trends.index)
		axes[0].set_xticklabels(trends.index, rotation=45, ha='right', color='#d9d9d9')
		axes[0].set_ylim(0, max(trends['total_rbs']) * 1.25 if len(trends) else 1)
		for bar, val in zip(bars0, trends['total_rbs']):
			axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() - 0.35,
					 f'{int(val)}', ha='center', va='top', fontsize=9, color='black', fontweight='bold')

		bars1 = axes[1].bar(trends.index, trends['first_round_rbs'], color='#ff6a00', alpha=0.95,
							 edgecolor='#ff8c42', linewidth=0.8, width=0.62)
		axes[1].set_title('First-Round RBs Per Year', fontweight='bold', fontsize=13, color='white')
		axes[1].set_xlabel('Year', color='#d9d9d9')
		axes[1].set_ylabel('Number of First-Round RBs', color='#d9d9d9')
		axes[1].set_xticks(trends.index)
		axes[1].set_xticklabels(trends.index, rotation=45, ha='right', color='#d9d9d9')
		axes[1].set_ylim(0, max(trends['first_round_rbs'].max(), 1) * 1.25)
		for bar, val in zip(bars1, trends['first_round_rbs']):
			axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() - 0.18,
					 f'{int(val)}', ha='center', va='top', fontsize=9, color='black', fontweight='bold')

		plt.tight_layout()
		out = os.path.join(output_dir, 'draft_trends.png')
		plt.savefig(out, dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())
		print(f'Saved: {out}')
		plt.close()

		print('\nDraft Trends Summary:')
		print(trends.to_string())


if __name__ == '__main__':
	print('This module is for visualization. Import it in your pipeline script.')
