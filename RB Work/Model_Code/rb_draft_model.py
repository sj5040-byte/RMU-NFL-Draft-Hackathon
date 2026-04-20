import pandas as pd
import numpy as np
import xgboost as xgb
import optuna
from optuna.pruners import MedianPruner
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    confusion_matrix, roc_auc_score,
    precision_recall_curve, f1_score, accuracy_score
)
from sklearn.model_selection import StratifiedKFold
import warnings

warnings.filterwarnings('ignore')

np.random.seed(42)


class RBDraftPredictor:
    """
    XGBoost model to predict first-round RB draft outcomes.
    Structure and logic mirrors QBDraftPredictor, adapted for RBs.
    """

    def __init__(self, csv_path: str):
        self.df = pd.read_csv(csv_path)
        self.label_encoders = {}
        self.feature_names = None
        self.cv_results = []
        self.best_params = None
        self.all_predictions = []
        self.cv_strategy = None


    def prepare_data(self):
        df = self.df.copy()
        df['first_round'] = (df['round'] == 1).astype(int)

        # Always drop these columns if present
        always_drop = ['Unnamed: 0', 'year']

        drop_cols = ['overall', 'round', 'pick', 'nfl_team',
        'name', 'position','college_team','vertical','broad_jump',
        'cone','shuttle'] + always_drop

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols = [c for c in numeric_cols if c not in drop_cols + ['first_round']]

        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        categorical_cols = [c for c in categorical_cols if c not in drop_cols + ['name']]

        for col in numeric_cols:
            if df[col].isnull().sum() > 0:
                df[col].fillna(df[col].median(), inplace=True)

        for col in categorical_cols:
            df[col].fillna(
                df[col].mode()[0] if len(df[col].mode()) > 0 else 'Unknown',
                inplace=True
            )

        for col in categorical_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            self.label_encoders[col] = le

        X = df[numeric_cols + categorical_cols].copy()
        y = df['first_round'].copy()
        self.feature_names = X.columns.tolist()

        print(f"Dataset shape: {X.shape}")
        print(f"Target distribution:\n{y.value_counts()}")
        print(f"First-round RBs: {y.sum()} ({y.mean() * 100:.1f}%)")

        return X, y, df

    def _objective(self, trial, X_train, y_train) -> float:
        neg = (y_train == 0).sum()
        pos = (y_train == 1).sum()
        params = {
            'objective':        'binary:logistic',
            'eval_metric':      'aucpr',
            'max_depth':        trial.suggest_int('max_depth', 3, 10),
            'learning_rate':    trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'subsample':        trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'gamma':            trial.suggest_float('gamma', 0, 5),
            'lambda':           trial.suggest_float('lambda', 0.001, 5, log=True),
            'alpha':            trial.suggest_float('alpha', 0.001, 5, log=True),
            'n_estimators':     trial.suggest_int('n_estimators', 100, 500),
            'scale_pos_weight': trial.suggest_float(
                'scale_pos_weight',
                max(1.0, (neg / pos) * 0.5),
                (neg / pos) * 2.0
            ),
            'random_state': 42,
            'verbosity':    0,
        }
        dtrain = xgb.DMatrix(X_train, label=y_train)
        cv = xgb.cv(
            params, dtrain,
            num_boost_round=params['n_estimators'],
            nfold=5, stratified=True,
            metrics=['aucpr'],
            early_stopping_rounds=10,
            verbose_eval=False
        )
        return cv['test-aucpr-mean'].max()

    def hyperparameter_tuning(self, X_train, y_train, n_trials: int = 50) -> dict:
        print("\n" + "=" * 60)
        print("HYPERPARAMETER TUNING")
        print("=" * 60)
        study = optuna.create_study(
            direction='maximize',
            pruner=MedianPruner(),
            sampler=optuna.samplers.TPESampler(seed=42)
        )
        study.optimize(
            lambda trial: self._objective(trial, X_train, y_train),
            n_trials=n_trials,
            show_progress_bar=True
        )
        self.best_params = study.best_params
        self.best_params['objective']    = 'binary:logistic'
        self.best_params['eval_metric']  = 'aucpr'
        self.best_params['random_state'] = 42
        self.best_params['verbosity']    = 0
        print(f"\nBest hyperparameters found:")
        for k, v in self.best_params.items():
            print(f"  {k}: {v}")
        print(f"Best validation score (aucpr): {study.best_value:.4f}")
        return self.best_params

    def _run_single_fold(self, X_train_fold, y_train_fold,
                         X_test_fold, y_test_fold,
                         fold_label: str,
                         val_X=None, val_y=None) -> dict:
        print(f"Training samples: {len(X_train_fold)} | Test samples: {len(X_test_fold)}")
        print(f"First-round RBs in test set: {int(y_test_fold.sum())}")
        dtrain = xgb.DMatrix(X_train_fold, label=y_train_fold)
        dtest  = xgb.DMatrix(X_test_fold,  label=y_test_fold)
        model = xgb.train(
            self.best_params, dtrain,
            num_boost_round=self.best_params.get('n_estimators', 300),
            evals=[(dtest, 'test')],
            early_stopping_rounds=10,
            verbose_eval=False
        )
        y_pred_proba = model.predict(dtest)
        if val_X is not None and val_y is not None and len(val_y) > 0 and val_y.sum() > 0:
            thresh_X, thresh_y = val_X, val_y
        else:
            skf_t = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            try:
                _, vi = next(skf_t.split(X_train_fold, y_train_fold))
                thresh_X = X_train_fold.iloc[vi] if hasattr(X_train_fold, 'iloc') else X_train_fold[vi]
                thresh_y = y_train_fold.iloc[vi] if hasattr(y_train_fold, 'iloc') else y_train_fold[vi]
            except Exception:
                thresh_X, thresh_y = None, None
        if thresh_X is not None and thresh_y is not None and int(thresh_y.sum()) > 0:
            val_proba = model.predict(xgb.DMatrix(thresh_X))
            prec, rec, thresholds = precision_recall_curve(thresh_y, val_proba)
            f1_t = 2 * prec * rec / (prec + rec + 1e-8)
            best_threshold = float(np.clip(
                thresholds[f1_t[:-1].argmax()], 0.1, 0.9
            )) if len(thresholds) > 0 else 0.5
        else:
            best_threshold = 0.5
        y_pred   = (y_pred_proba > best_threshold).astype(int)
        accuracy = accuracy_score(y_test_fold, y_pred)
        f1       = f1_score(y_test_fold, y_pred, zero_division=0)
        roc_auc  = (
            roc_auc_score(y_test_fold, y_pred_proba)
            if len(np.unique(y_test_fold)) > 1 else np.nan
        )
        cm = confusion_matrix(y_test_fold, y_pred)
        print(f"Optimal threshold: {best_threshold:.3f}")
        print(f"Accuracy: {accuracy:.4f} | F1: {f1:.4f}", end='')
        print(f" | ROC-AUC: {roc_auc:.4f}" if not np.isnan(roc_auc) else "")
        print(f"Confusion Matrix:\n{cm}")
        return {
            'fold_label':       fold_label,
            'train_size':       len(X_train_fold),
            'test_size':        len(X_test_fold),
            'accuracy':         accuracy,
            'f1_score':         f1,
            'roc_auc':          roc_auc,
            'best_threshold':   best_threshold,
            'model':            model,
            'y_test':           y_test_fold,
            'y_pred':           y_pred,
            'y_pred_proba':     y_pred_proba,
            'confusion_matrix': cm,
        }

    def train_and_evaluate_grouped(self, X, y, df, n_splits: int = 5):
        print("\n" + "=" * 60)
        print(f"GROUPED K-FOLD (PRIMARY BENCHMARK) — k={n_splits}, chronological year groups")
        print("=" * 60)
        self.cv_strategy = 'GKF'
        years    = np.array(sorted(df['year'].unique()))
        n_years  = len(years)
        year_to_fold = {yr: i * n_splits // n_years for i, yr in enumerate(years)}
        fold_col     = df['year'].map(year_to_fold).values
        print(f"\nYear to group assignments:")
        for fold_id in range(n_splits):
            group_years = [yr for yr, g in year_to_fold.items() if g == fold_id]
            print(f"  Group {fold_id + 1}: {group_years}")
        print(f"\nTuning hyperparameters on full dataset...")
        self.hyperparameter_tuning(X, y, n_trials=50)
        self.cv_results = []
        fold_results    = []
        for fold_id in range(n_splits):
            test_mask  = fold_col == fold_id
            train_mask = ~test_mask
            X_tr, X_te = X[train_mask], X[test_mask]
            y_tr, y_te = y[train_mask], y[test_mask]
            test_years = sorted(df['year'][test_mask].unique())
            print(f"\n{'−' * 60}")
            print(f"Group {fold_id + 1} / {n_splits} — Test years: {test_years}")
            print(f"{'−' * 60}")
            if y_tr.sum() == 0 or y_tr.sum() == len(y_tr):
                print("Skipping: insufficient class variation in training data")
                continue
            prev_folds = [f for f in range(fold_id) if f in fold_col]
            if prev_folds:
                latest_train_fold = max(prev_folds)
                val_mask   = fold_col == latest_train_fold
                val_X_fold = X[val_mask]
                val_y_fold = y[val_mask]
            else:
                val_X_fold, val_y_fold = None, None
            result = self._run_single_fold(
                X_tr, y_tr, X_te, y_te,
                fold_label=f"group_{fold_id + 1}",
                val_X=val_X_fold, val_y=val_y_fold
            )
            result['fold']       = fold_id + 1
            result['test_years'] = test_years
            fold_results.append(result)
            self.cv_results.append(result)
        return fold_results

    def train_and_evaluate_skf(self, X, y, n_splits: int = 5):
        print("\n" + "=" * 60)
        print(f"STRATIFIED K-FOLD (STABILITY CHECK) — k={n_splits}")
        print("=" * 60)
        self.cv_strategy = 'SKF'
        print(f"\nTuning hyperparameters on full dataset...")
        self.hyperparameter_tuning(X, y, n_trials=50)
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        self.cv_results = []
        fold_results    = []
        for fold_num, (train_idx, test_idx) in enumerate(skf.split(X, y), start=1):
            print(f"\n{'−' * 60}")
            print(f"Fold {fold_num} / {n_splits}")
            print(f"{'−' * 60}")
            X_tr = X.iloc[train_idx]
            y_tr = y.iloc[train_idx]
            X_te = X.iloc[test_idx]
            y_te = y.iloc[test_idx]
            result = self._run_single_fold(
                X_tr, y_tr, X_te, y_te,
                fold_label=f"fold_{fold_num}"
            )
            result['fold'] = fold_num
            fold_results.append(result)
            self.cv_results.append(result)
        return fold_results

    def print_cross_validation_summary(self):
        label_map = {
            'GKF': 'Grouped K-Fold — Primary Benchmark',
            'SKF': 'Stratified K-Fold — Stability Check',
        }
        strategy_label = label_map.get(self.cv_strategy, 'Cross-Validation')
        print("\n" + "=" * 60)
        print(f"CROSS-VALIDATION SUMMARY — {strategy_label}")
        print("=" * 60)
        if not self.cv_results:
            print("No valid fold results to summarize")
            return
        accuracies = [r['accuracy'] for r in self.cv_results]
        f1_scores  = [r['f1_score'] for r in self.cv_results]
        roc_aucs   = [r['roc_auc']  for r in self.cv_results if not np.isnan(r['roc_auc'])]
        print(f"\nAccuracy:")
        print(f"  Mean: {np.mean(accuracies):.4f} (+/- {np.std(accuracies):.4f})")
        print(f"  Min:  {np.min(accuracies):.4f}, Max: {np.max(accuracies):.4f}")
        print(f"\nF1-Score:")
        print(f"  Mean: {np.mean(f1_scores):.4f} (+/- {np.std(f1_scores):.4f})")
        print(f"  Min:  {np.min(f1_scores):.4f}, Max: {np.max(f1_scores):.4f}")
        if roc_aucs:
            print(f"\nROC-AUC:")
            print(f"  Mean: {np.mean(roc_aucs):.4f} (+/- {np.std(roc_aucs):.4f})")
            print(f"  Min:  {np.min(roc_aucs):.4f}, Max: {np.max(roc_aucs):.4f}")
        print(f"\nPer-Fold Breakdown:")
        print(f"  {'Label':<18} {'Acc':>6} {'F1':>6} {'AUC':>6} {'Threshold':>10}")
        print(f"  {'─' * 50}")
        for r in self.cv_results:
            label   = r.get('fold_label', str(r.get('fold', '?')))
            auc_str = f"{r['roc_auc']:.4f}" if not np.isnan(r['roc_auc']) else '  N/A'
            print(f"  {label:<18} {r['accuracy']:>6.4f} {r['f1_score']:>6.4f} "
                  f"{auc_str:>6} {r['best_threshold']:>10.3f}")
        total_cm  = sum(r['confusion_matrix'] for r in self.cv_results)
        total_acc = (total_cm[0, 0] + total_cm[1, 1]) / total_cm.sum()
        print(f"\nAggregated Confusion Matrix (across all folds):")
        print(f"  TN: {int(total_cm[0, 0])}, FP: {int(total_cm[0, 1])}")
        print(f"  FN: {int(total_cm[1, 0])}, TP: {int(total_cm[1, 1])}")
        print(f"  Overall Accuracy: {total_acc:.4f}")

    def feature_importance_analysis(self):
        print("\n" + "=" * 60)
        print("FEATURE IMPORTANCE ANALYSIS (from GKF models)")
        print("=" * 60)
        if not self.cv_results:
            print("No models available for feature importance")
            return
        importance_dict = {}
        for result in self.cv_results:
            for feat, score in result['model'].get_score(importance_type='weight').items():
                importance_dict.setdefault(feat, []).append(score)
        mean_importance   = {f: np.mean(s) for f, s in importance_dict.items()}
        sorted_importance = sorted(mean_importance.items(), key=lambda x: x[1], reverse=True)
        print("\nTop 20 Most Important Features:")
        for i, (feat, imp) in enumerate(sorted_importance[:20], 1):
            print(f"{i:2d}. {feat}: {imp:.2f}")
        print("\nFeature Mapping (Index -> Name):")
        for idx, name in enumerate(self.feature_names):
            f_key = f'f{idx}'
            if f_key in mean_importance:
                print(f"  f{idx}: {name} (Importance: {mean_importance[f_key]:.2f})")
        return mean_importance

    def generate_predictions_for_new_rbs(self, new_rbs_csv: str = None):
        print("\n" + "=" * 60)
        print("PREDICTION GENERATION FOR NEW RBs")
        print("=" * 60)
        if not self.cv_results:
            print("No trained models available")
            return None
        latest_model = self.cv_results[-1]['model']
        name_col = None
        if new_rbs_csv:
            new_df = pd.read_csv(new_rbs_csv)
            print(f"Loaded {len(new_df)} new RB records")
            # Save name column if present
            if 'name' in new_df.columns:
                name_col = new_df['name']
            for col in new_df.select_dtypes(include=[np.number]).columns:
                new_df[col].fillna(new_df[col].median(), inplace=True)
            for col in new_df.select_dtypes(include=['object']).columns:
                new_df[col].fillna(
                    new_df[col].mode()[0] if len(new_df[col].mode()) > 0 else 'Unknown',
                    inplace=True
                )
            for col in new_df.select_dtypes(include=['object']).columns:
                if col in self.label_encoders:
                    new_df[col] = self.label_encoders[col].transform(new_df[col].astype(str))
            # Only use columns present in both test and training features, and drop always_drop columns if present
            always_drop = ['Unnamed: 0', 'year']
            test_features = [col for col in self.feature_names if col in new_df.columns and col not in always_drop]
            X_new = new_df[test_features]
        else:
            X_new = self.cv_results[-1].get('X_test')
            if X_new is None:
                print("No new data provided and no test set available")
                return None
        predictions    = latest_model.predict(xgb.DMatrix(X_new))
        # Use mean threshold from all folds
        thresholds = [r.get('best_threshold', 0.5) for r in self.cv_results if 'best_threshold' in r]
        if thresholds:
            threshold = float(np.mean(thresholds))
        else:
            threshold = 0.5
        pred_first_round = (predictions > threshold).astype(int)
        # Compose output DataFrame with name column if available, always output required columns
        if name_col is not None:
            predictions_df = pd.DataFrame({
                'name': name_col,
                'probability_first_round': predictions,
                'predicted_first_round': pred_first_round,
            })
        else:
            # If no name column, just output probabilities and predictions
            predictions_df = pd.DataFrame({
                'name': [None]*len(predictions),
                'probability_first_round': predictions,
                'predicted_first_round': pred_first_round,
            })
        predictions_df = predictions_df[['name', 'probability_first_round', 'predicted_first_round']]
        predictions_df = predictions_df.sort_values('probability_first_round', ascending=False)
        print("\nTop predicted first-round RBs:")
        print(predictions_df.head(10).to_string())
        return predictions_df
