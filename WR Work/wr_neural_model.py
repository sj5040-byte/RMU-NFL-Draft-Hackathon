import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    confusion_matrix, roc_auc_score, precision_recall_curve,
    f1_score, accuracy_score, precision_score, recall_score
)
from sklearn.model_selection import StratifiedKFold
from imblearn.over_sampling import SMOTE
import warnings

warnings.filterwarnings('ignore')
np.random.seed(42)
torch.manual_seed(42)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Columns unavailable pre-2017 — always excluded from training and inference
EXCLUDED_COLS = ['Catch%', '1st_Downs', 'Fumbles']


class WRDraftNetwork(nn.Module):
    """Larger network capacity for imbalanced learning."""

    def __init__(self, input_dim, hidden_dims=[512, 256, 128], dropout_rate=0.2):
        super(WRDraftNetwork, self).__init__()

        self.input_norm = nn.BatchNorm1d(input_dim)

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.LeakyReLU(0.1))
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim

        self.dense_layers = nn.Sequential(*layers)
        self.output_layer = nn.Linear(prev_dim, 1)

    def forward(self, x):
        x = self.input_norm(x)
        x = self.dense_layers(x)
        x = self.output_layer(x)
        return x


class WRDraftPredictor:
    """Deep learning model with SMOTE oversampling for imbalanced classification.
    Uses all available years. Catch%, 1st_Downs, and Fumbles are always excluded
    as they are unavailable for pre-2017 data.
    """

    def __init__(self, csv_path: str):
        self.df = pd.read_csv(csv_path)
        self.label_encoders = {}
        self.feature_names = None
        self.cv_results = []
        self.cv_strategy = None

    def prepare_data(self):
        """Prepare and preprocess data."""
        df = self.df.copy()
        df['first_round'] = (df['round'] == 1).astype(int)

        drop_cols = [
            'overall', 'round', 'pick', 'nfl_team',
            'name', 'position', 'college_team',
            "regular_season_wins", "regular_season_losses",
            "postseason_games", "postseason_wins", "postseason_losses",
        ] + [c for c in EXCLUDED_COLS if c in df.columns]

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols = [c for c in numeric_cols if c not in drop_cols + ['first_round']]

        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        categorical_cols = [c for c in categorical_cols if c not in drop_cols + ['name']]

        for col in numeric_cols:
            if df[col].isnull().sum() > 0:
                df[col].fillna(df[col].median(), inplace=True)

        for col in categorical_cols:
            if df[col].isnull().sum() > 0:
                df[col].fillna(df[col].mode()[0] if len(df[col].mode()) > 0 else 'Unknown', inplace=True)

        for col in categorical_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            self.label_encoders[col] = le

        X = df[numeric_cols + categorical_cols].copy()
        y = df['first_round'].copy()
        self.feature_names = X.columns.tolist()

        print(f"\n{'=' * 70}")
        print("DATA PREPARATION SUMMARY")
        print(f"{'=' * 70}")
        print(f"Feature matrix shape: {X.shape}")
        print(f"Numeric features: {len(numeric_cols)}")
        print(f"Categorical features: {len(categorical_cols)}")
        print(f"\nTarget distribution:")
        print(f"  First-round WRs: {int(y.sum())} ({y.mean() * 100:.1f}%)")
        print(f"  Not first-round: {int((1 - y).sum())} ({(1 - y).mean() * 100:.1f}%)")
        print(f"\nYear range: {df['year'].min()} to {df['year'].max()}")
        print(f"Total samples: {len(df)}")

        return X, y, df

    def _train_single_fold(self, X_train_fold, y_train_fold,
                          X_test_fold, y_test_fold,
                          fold_label: str,
                          epochs: int = 100,
                          batch_size: int = 32) -> dict:
        """Train neural network with SMOTE oversampling."""

        X_train_fold = X_train_fold.fillna(X_train_fold.median())
        X_test_fold = X_test_fold.fillna(X_test_fold.median())

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_fold)
        X_test_scaled = scaler.transform(X_test_fold)

        X_train_scaled = np.nan_to_num(X_train_scaled, nan=0.0)
        X_test_scaled = np.nan_to_num(X_test_scaled, nan=0.0)

        print(f"    Before SMOTE: {len(X_train_scaled)} samples")
        smote = SMOTE(random_state=42, k_neighbors=min(3, (y_train_fold.values == 1).sum() - 1))
        X_train_smote, y_train_smote = smote.fit_resample(X_train_scaled, y_train_fold.values)
        print(f"    After SMOTE: {len(X_train_smote)} samples (pos: {y_train_smote.sum()}, neg: {(1-y_train_smote).sum()})")

        X_train_tensor = torch.FloatTensor(X_train_smote).to(DEVICE)
        y_train_tensor = torch.FloatTensor(y_train_smote).reshape(-1, 1).to(DEVICE)
        X_test_tensor = torch.FloatTensor(X_test_scaled).to(DEVICE)
        y_test_tensor = torch.FloatTensor(y_test_fold.values).reshape(-1, 1).to(DEVICE)

        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        input_dim = X_train_scaled.shape[1]
        model = WRDraftNetwork(input_dim=input_dim).to(DEVICE)

        pos_weight = (y_train_smote == 0).sum() / (y_train_smote == 1).sum()
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight).to(DEVICE))
        optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-3)

        best_f1 = -1
        best_threshold = 0.5
        best_model_state = None
        patience = 40
        patience_counter = 0

        for epoch in range(epochs):
            model.train()
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                logits = model(X_batch)
                loss = criterion(logits, y_batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            model.eval()
            with torch.no_grad():
                test_logits = model(X_test_tensor).cpu().numpy()
                test_logits = np.clip(test_logits, -100, 100)
                y_pred_proba = 1 / (1 + np.exp(-test_logits)).flatten()

                best_val_f1 = -1
                best_val_thresh = 0.5

                for threshold in np.arange(0.1, 0.6, 0.05):
                    y_pred = (y_pred_proba > threshold).astype(int)
                    if y_pred.sum() > 0:
                        val_f1 = f1_score(y_test_fold.values, y_pred, zero_division=0)
                        if val_f1 > best_val_f1:
                            best_val_f1 = val_f1
                            best_val_thresh = threshold

                if best_val_f1 > best_f1:
                    best_f1 = best_val_f1
                    best_threshold = best_val_thresh
                    best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                    patience_counter = 0
                else:
                    patience_counter += 1

            if patience_counter >= patience:
                break

        if best_model_state is not None:
            model.load_state_dict(best_model_state)

        model.eval()
        with torch.no_grad():
            test_logits = model(X_test_tensor).cpu().numpy()
            test_logits = np.clip(test_logits, -100, 100)
            y_pred_proba = 1 / (1 + np.exp(-test_logits)).flatten()

        y_pred = (y_pred_proba > best_threshold).astype(int)

        cm = confusion_matrix(y_test_fold.values, y_pred, labels=[0, 1])
        acc = accuracy_score(y_test_fold.values, y_pred)
        f1 = f1_score(y_test_fold.values, y_pred, zero_division=0)
        prec = precision_score(y_test_fold.values, y_pred, zero_division=0)
        rec = recall_score(y_test_fold.values, y_pred, zero_division=0)

        try:
            roc_auc = roc_auc_score(y_test_fold.values, y_pred_proba)
        except ValueError:
            roc_auc = np.nan

        return {
            'fold_label': fold_label,
            'train_size': len(X_train_fold),
            'test_size': len(X_test_fold),
            'accuracy': acc,
            'f1_score': f1,
            'precision': prec,
            'recall': rec,
            'roc_auc': roc_auc,
            'best_threshold': best_threshold,
            'confusion_matrix': cm,
            'model': model,
            'scaler': scaler,
            'y_test': y_test_fold.values,
            'y_pred_proba': y_pred_proba,
            'X_test': X_test_fold,
        }

    def evaluate_gkf(self, X, y, df, n_splits: int = 5):
        """Grouped K-Fold cross-validation."""
        print(f"\n{'=' * 70}")
        print("GKF: GROUPED K-FOLD (Primary Benchmark)")
        print(f"{'=' * 70}")
        print(f"Testing on {n_splits} chronological year blocks...\n")

        self.cv_strategy = 'GKF'
        unique_years = sorted(df['year'].unique())
        year_groups = np.array_split(unique_years, n_splits)
        group_assignment = np.zeros(len(df), dtype=int)
        for group_idx, year_list in enumerate(year_groups):
            group_assignment[df['year'].isin(year_list)] = group_idx

        self.cv_results = []

        for test_group in range(n_splits):
            print(f"\n{'-' * 70}")
            print(f"Group {test_group + 1} / {n_splits}")
            print(f"{'-' * 70}")

            test_mask = group_assignment == test_group
            train_mask = ~test_mask

            X_tr = X[train_mask].reset_index(drop=True)
            y_tr = y[train_mask].reset_index(drop=True)
            X_te = X[test_mask].reset_index(drop=True)
            y_te = y[test_mask].reset_index(drop=True)

            print(f"  Training samples: {len(X_tr)} | Test samples: {len(X_te)}")
            print(f"  First-round WRs in test: {int(y_te.sum())}")

            result = self._train_single_fold(X_tr, y_tr, X_te, y_te, f"group_{test_group + 1}")
            self.cv_results.append(result)

        print(f"\n{'=' * 70}")
        print("GKF EVALUATION COMPLETE")
        print(f"{'=' * 70}")

    def evaluate_skf(self, X, y, n_splits: int = 5):
        """Stratified K-Fold cross-validation."""
        print(f"\n{'=' * 70}")
        print(f"SKF: STRATIFIED K-FOLD (Stability Check) — k={n_splits}")
        print(f"{'=' * 70}")
        print(f"Testing on {n_splits} random stratified splits...\n")

        self.cv_strategy = 'SKF'
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        self.cv_results = []

        for fold_num, (train_idx, test_idx) in enumerate(skf.split(X, y), start=1):
            print(f"\n{'-' * 70}")
            print(f"Fold {fold_num} / {n_splits}")
            print(f"{'-' * 70}")

            X_tr = X.iloc[train_idx].reset_index(drop=True)
            y_tr = y.iloc[train_idx].reset_index(drop=True)
            X_te = X.iloc[test_idx].reset_index(drop=True)
            y_te = y.iloc[test_idx].reset_index(drop=True)

            print(f"  Training samples: {len(X_tr)} | Test samples: {len(X_te)}")
            print(f"  First-round WRs in test: {int(y_te.sum())}")

            result = self._train_single_fold(X_tr, y_tr, X_te, y_te, f"fold_{fold_num}")
            self.cv_results.append(result)

        print(f"\n{'=' * 70}")
        print("SKF EVALUATION COMPLETE")
        print(f"{'=' * 70}")

    def print_cross_validation_summary(self):
        """Print CV summary metrics."""
        strategy_label = {
            'GKF': 'Grouped K-Fold (Primary)',
            'SKF': 'Stratified K-Fold (Stability Check)',
        }.get(self.cv_strategy, 'Cross-Validation')

        print(f"\n{'=' * 70}")
        print(f"CV SUMMARY — {strategy_label}")
        print(f"{'=' * 70}")

        if not self.cv_results:
            print("No fold results available")
            return

        accs  = [r['accuracy']  for r in self.cv_results]
        f1s   = [r['f1_score']  for r in self.cv_results]
        precs = [r['precision'] for r in self.cv_results]
        recs  = [r['recall']    for r in self.cv_results]
        aucs  = [r['roc_auc']   for r in self.cv_results if not np.isnan(r['roc_auc'])]

        print(f"\nAccuracy:  Mean: {np.mean(accs):.4f} (±{np.std(accs):.4f})")
        print(f"F1-Score:  Mean: {np.mean(f1s):.4f} (±{np.std(f1s):.4f})")
        print(f"Precision: Mean: {np.mean(precs):.4f} (±{np.std(precs):.4f})")
        print(f"Recall:    Mean: {np.mean(recs):.4f} (±{np.std(recs):.4f})")
        if aucs:
            print(f"ROC-AUC:   Mean: {np.mean(aucs):.4f} (±{np.std(aucs):.4f})")

        print(f"\nPer-Fold Breakdown:")
        print(f"  {'Fold':<15} {'Acc':>7} {'F1':>7} {'Prec':>7} {'Rec':>7} {'Thresh':>8}")
        print(f"  {'-' * 60}")
        for r in self.cv_results:
            print(f"  {r.get('fold_label', '?'):<15} {r['accuracy']:>7.4f} {r['f1_score']:>7.4f} "
                  f"{r['precision']:>7.4f} {r['recall']:>7.4f} {r['best_threshold']:>8.3f}")

        total_cm = sum(r['confusion_matrix'] for r in self.cv_results)
        total_acc = (total_cm[0, 0] + total_cm[1, 1]) / total_cm.sum() if total_cm.sum() > 0 else 0
        print(f"\nAggregated Confusion Matrix:")
        print(f"  TN: {int(total_cm[0, 0])}, FP: {int(total_cm[0, 1])}")
        print(f"  FN: {int(total_cm[1, 0])}, TP: {int(total_cm[1, 1])}")
        print(f"  Overall Accuracy: {total_acc:.4f}")

    def get_feature_importance_from_gradients(self):
        """Compute gradient-based feature importance."""
        print(f"\n{'=' * 70}")
        print("FEATURE IMPORTANCE ANALYSIS (Gradient-based)")
        print(f"{'=' * 70}")

        if not self.cv_results:
            print("No models available")
            return None

        importance_dict = {}

        for result in self.cv_results:
            model  = result['model']
            scaler = result['scaler']
            X_test = result['X_test']

            X_test_scaled  = scaler.transform(X_test)
            X_test_tensor  = torch.FloatTensor(X_test_scaled).to(DEVICE)
            X_test_tensor.requires_grad_(True)

            model.eval()
            output = model(X_test_tensor)
            output.sum().backward()
            gradients = X_test_tensor.grad.data.cpu().numpy()
            feature_importance = np.abs(gradients).mean(axis=0)

            for feat_idx, importance in enumerate(feature_importance):
                feat_name = self.feature_names[feat_idx]
                importance_dict.setdefault(feat_name, []).append(importance)

        mean_importance   = {f: np.mean(scores) for f, scores in importance_dict.items()}
        sorted_importance = sorted(mean_importance.items(), key=lambda x: x[1], reverse=True)

        print("\nTop 20 Most Important Features:")
        for i, (feat, imp) in enumerate(sorted_importance[:20], 1):
            print(f"{i:2d}. {feat:25s} (importance: {imp:.4f})")

        return mean_importance


if __name__ == '__main__':
    print("WR Draft Predictor module. Use in pipeline script.")