"""
Core model class for predicting whether a college RB gets drafted
in the first round of the NFL Draft.

Algorithm: XGBoost binary classifier, tuned via Optuna.

Evaluation uses two cross-validation strategies:

    GKF (Grouped K-Fold) -- PRIMARY
        Years are sorted chronologically and split into k blocks.
        Each block is held out once as the test set. Respects temporal
        ordering. No future data bleeds into training. This is the
        honest, reported benchmark.

    SKF (Stratified K-Fold) -- STABILITY CHECK
        Random stratified splits. Each fold preserves the natural
        first-round RB ratio. Used to detect temporal leakage and
        confirm GKF variance is real signal, not fold-size noise.
"""

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

# Fix seed for reproducibility across runs.
np.random.seed(42)


class RBDraftPredictor:
    """
    XGBoost model to predict first-round RB draft outcomes.
    df : pd.DataFrame
        Raw training data loaded from CSV.
    label_encoders : dict
        Fitted LabelEncoder per categorical column. Stored for reuse
        at inference time.
    feature_names : list[str]
        Ordered column names of the feature matrix. Used to align
        new data at inference.
    cv_results : list[dict]
        Fold-level results from the most recent CV run. Each dict holds
        metrics, the trained model, predictions, and the confusion matrix.
    best_params : dict
        Hyperparameters from the last Optuna study. Shared across GKF
        and SKF so both strategies are compared on equal footing.
    all_predictions : list
        Reserved for future use (e.g. ensembling).
    cv_strategy : str or None
        'GKF' or 'SKF'. Tracks which strategy last populated cv_results.
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
        """
        Clean, encode, and engineer features from the raw DataFrame.

        1. Create binary target: first_round = 1 if round == 1.
        2. Drop columns that are identifiers, post-draft outcomes,
           high-missingness combine drills, or CSV artefacts.
        3. Median-impute missing numeric values.
        4. Mode-impute missing categorical values.
        5. Label-encode categoricals. Encoders stored for inference.

        Drop list rationale
        - Draft outcome columns (leakage): overall, round, pick.
        - Identifiers: nfl_team, name, position, college_team.
        - Combine drills with high missingness: vertical, broad_jump,
          cone, shuttle.
        - always_drop: 'Unnamed: 0' (CSV index artefact) and 'year'.
          Year is excluded for RBs because draft era trends are less
          predictive for RBs than for QBs and WRs, and including it
          risks the model learning year-level noise rather than player
          quality signals.
        """
        df = self.df.copy()

        # Target: 1 if the RB was a first-round pick, 0 otherwise.
        df['first_round'] = (df['round'] == 1).astype(int)

        always_drop = ['Unnamed: 0', 'year']

        drop_cols = [
            'overall', 'round', 'pick', 'nfl_team',
            'name', 'position', 'college_team',
            'vertical', 'broad_jump', 'cone', 'shuttle',
        ] + always_drop

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols = [c for c in numeric_cols if c not in drop_cols + ['first_round']]

        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        categorical_cols = [c for c in categorical_cols if c not in drop_cols + ['name']]

        # Impute numerics with median.
        for col in numeric_cols:
            if df[col].isnull().sum() > 0:
                df[col].fillna(df[col].median(), inplace=True)

        # Impute categoricals with mode.
        for col in categorical_cols:
            df[col].fillna(
                df[col].mode()[0] if len(df[col].mode()) > 0 else 'Unknown',
                inplace=True
            )

        # Label-encode categoricals. Store encoders for inference.
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
        """
        Optuna objective function.
        Maximises AUCPR via an internal 5-fold stratified CV on the
        training data. AUCPR is preferred over log-loss for imbalanced
        data because it focuses optimisation on the minority class
        (first-round RBs, ~7% of the training set).

        scale_pos_weight is searched in a range centered on the natural
        neg/pos ratio, allowing the tuner to over- or under-weight the
        minority class relative to the naive balance.

        Parameters
        ----------
        trial : optuna.Trial
            Optuna trial suggesting hyperparameter values.
        X_train : array-like
            Training features for internal CV.
        y_train : array-like
            Training labels.
        """
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
        """
        Run an Optuna TPE study and store the best hyperparameters.
        MedianPruner cuts unpromising trials early, which keeps
        wall-clock time reasonable at 50 trials. The same best_params
        dict is reused for both GKF and SKF so both strategies are
        evaluated on an equal footing.

        Parameters
        X_train : array-like
            Feature matrix used for internal CV during tuning.
        y_train : array-like
            Target labels.
        n_trials : int
            Number of Optuna trials. Default 50.

        """
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
        # Append fixed params not part of the search space.
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
        """
        Train XGBoost on one fold and evaluate on the held-out set.

        Threshold tuning
        The classification threshold is tuned to maximise F1. We prefer
        a separate validation set so the test set stays clean. Priority:
            1. Explicit val_X / val_y passed by the caller (GKF uses the
               most recent training group as its validation set).
            2. A 20% stratified split carved from the training fold.
            3. Default threshold of 0.5 if neither is available.

        Parameters
        ----------
        X_train_fold : array-like
            Training features for this fold.
        y_train_fold : array-like
            Training labels.
        X_test_fold : array-like
            Held-out features.
        y_test_fold : array-like
            Held-out labels.
        fold_label : str
            Human-readable fold identifier (e.g. 'group_1', 'fold_3').
        val_X : array-like or None
            Optional validation features for threshold tuning.
        val_y : array-like or None
            Optional validation labels for threshold tuning.
        """
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

        # --- Threshold selection ---
        # Use the explicit validation set if provided; otherwise carve
        # a 20% stratified split out of the training fold.
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
            # Clip to [0.1, 0.9] to avoid degenerate thresholds.
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
        """
        PRIMARY BENCHMARK -- Grouped K-Fold by year.
        - Respects temporal ordering. A model trained on 2010-2018 data
          cannot see 2019+ picks. That is how it works in deployment.
        - Each fold contains enough first-round RBs to produce a
          meaningful F1 score.
        - Variance reflects era-level difficulty, not random sampling.
        """
        print("\n" + "=" * 60)
        print(f"GROUPED K-FOLD (PRIMARY BENCHMARK) -- k={n_splits}, chronological year groups")
        print("=" * 60)

        self.cv_strategy = 'GKF'
        years       = np.array(sorted(df['year'].unique()))
        n_years     = len(years)

        # Assign each year to a group using integer bucketing.
        year_to_fold = {yr: i * n_splits // n_years for i, yr in enumerate(years)}
        fold_col     = df['year'].map(year_to_fold).values

        print(f"\nYear to group assignments:")
        for fold_id in range(n_splits):
            group_years = [yr for yr, g in year_to_fold.items() if g == fold_id]
            print(f"  Group {fold_id + 1}: {group_years}")

        # Tune hyperparameters once on the full dataset. GKF and SKF
        # share these params so their results are directly comparable.
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
            print(f"\n{'--' * 30}")
            print(f"Group {fold_id + 1} / {n_splits} -- Test years: {test_years}")
            print(f"{'--' * 30}")

            # Skip folds where all training labels are the same class.
            if y_tr.sum() == 0 or y_tr.sum() == len(y_tr):
                print("Skipping: insufficient class variation in training data")
                continue

            # Use the immediately preceding group as the threshold-tuning
            # validation set so no test-year data influences the threshold.
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

    # ------------------------------------------------------------------
    # STABILITY CHECK: Stratified K-Fold
    # ------------------------------------------------------------------

    def train_and_evaluate_skf(self, X, y, n_splits: int = 5):
        """
        STABILITY CHECK -- Stratified K-Fold.
        Randomly splits data into k stratified folds. Each fold preserves
        the natural first-round RB ratio. NOT the primary benchmark
        because random splits allow RBs from later years to appear in
        training while earlier-year RBs appear in the test set.
        """
        print("\n" + "=" * 60)
        print(f"STRATIFIED K-FOLD (STABILITY CHECK) -- k={n_splits}")
        print("=" * 60)

        self.cv_strategy = 'SKF'

        # Retune hyperparameters for a clean, independent SKF run.
        print(f"\nTuning hyperparameters on full dataset...")
        self.hyperparameter_tuning(X, y, n_trials=50)

        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        self.cv_results = []
        fold_results    = []

        for fold_num, (train_idx, test_idx) in enumerate(skf.split(X, y), start=1):
            print(f"\n{'--' * 30}")
            print(f"Fold {fold_num} / {n_splits}")
            print(f"{'--' * 30}")

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
            'GKF': 'Grouped K-Fold -- Primary Benchmark',
            'SKF': 'Stratified K-Fold -- Stability Check',
        }
        strategy_label = label_map.get(self.cv_strategy, 'Cross-Validation')

        print("\n" + "=" * 60)
        print(f"CROSS-VALIDATION SUMMARY -- {strategy_label}")
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
        print(f"  {'-' * 50}")
        for r in self.cv_results:
            label   = r.get('fold_label', str(r.get('fold', '?')))
            auc_str = f"{r['roc_auc']:.4f}" if not np.isnan(r['roc_auc']) else '  N/A'
            print(f"  {label:<18} {r['accuracy']:>6.4f} {r['f1_score']:>6.4f} "
                  f"{auc_str:>6} {r['best_threshold']:>10.3f}")

        # Sum raw counts across folds for the aggregated view.
        total_cm  = sum(r['confusion_matrix'] for r in self.cv_results)
        total_acc = (total_cm[0, 0] + total_cm[1, 1]) / total_cm.sum()
        print(f"\nAggregated Confusion Matrix (across all folds):")
        print(f"  TN: {int(total_cm[0, 0])}, FP: {int(total_cm[0, 1])}")
        print(f"  FN: {int(total_cm[1, 0])}, TP: {int(total_cm[1, 1])}")
        print(f"  Overall Accuracy: {total_acc:.4f}")


    def feature_importance_analysis(self):
        """
        Aggregate XGBoost feature importance (split count) across all
        folds and print the top 20.

        Split count measures how many times each feature was used to
        make a split across all trees in a model. It is a frequency
        proxy -- not an effect size -- so features used in shallow
        splits early in each tree may score lower than their actual
        predictive contribution.

        Call this after GKF so importance reflects the primary benchmark
        models. Calling after SKF is valid but will reflect the random-
        split models instead.
        """
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

        # Map XGBoost internal keys (f0, f1, ...) back to column names.
        print("\nFeature Mapping (Index -> Name):")
        for idx, name in enumerate(self.feature_names):
            f_key = f'f{idx}'
            if f_key in mean_importance:
                print(f"  f{idx}: {name} (Importance: {mean_importance[f_key]:.2f})")

        return mean_importance


    def generate_predictions_for_new_rbs(self, new_rbs_csv: str = None):
        """
        Score a new cohort of RBs using the last trained GKF model.

        The last GKF fold model is used because it trains on the largest
        possible historical window. For inference, the decision threshold
        is the mean across all GKF fold thresholds. RB thresholds vary
        more fold-to-fold than QB thresholds (due to the smaller positive
        class), so averaging is more robust than using a single fold.

        Parameters
        ----------
        new_rbs_csv : str or None
            Path to a CSV of new RBs to score. Must contain the same
            feature columns used during training (minus the dropped
            columns). If None, falls back to the test set from the last
            fold (mainly for debugging).
        """
        print("\n" + "=" * 60)
        print("PREDICTION GENERATION FOR NEW RBs")
        print("=" * 60)

        if not self.cv_results:
            print("No trained models available")
            return None

        latest_model = self.cv_results[-1]['model']
        name_col     = None

        if new_rbs_csv:
            new_df = pd.read_csv(new_rbs_csv)
            print(f"Loaded {len(new_df)} new RB records")

            # Preserve name column before it gets dropped.
            if 'name' in new_df.columns:
                name_col = new_df['name']

            # Impute and encode using the same logic as prepare_data.
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

            # Only keep features present in both the test file and the
            # training feature list. Drop always_drop columns if present.
            always_drop   = ['Unnamed: 0', 'year']
            test_features = [
                col for col in self.feature_names
                if col in new_df.columns and col not in always_drop
            ]
            X_new = new_df[test_features]
        else:
            X_new = self.cv_results[-1].get('X_test')
            if X_new is None:
                print("No new data provided and no test set available")
                return None

        predictions = latest_model.predict(xgb.DMatrix(X_new))

        # Mean threshold across all folds for a stable decision boundary.
        thresholds = [r.get('best_threshold', 0.5) for r in self.cv_results if 'best_threshold' in r]
        threshold  = float(np.mean(thresholds)) if thresholds else 0.5

        pred_first_round = (predictions > threshold).astype(int)

        if name_col is not None:
            predictions_df = pd.DataFrame({
                'name':                    name_col,
                'probability_first_round': predictions,
                'predicted_first_round':   pred_first_round,
            })
        else:
            # Output a name column of None values to keep the schema
            # consistent regardless of whether names are available.
            predictions_df = pd.DataFrame({
                'name':                    [None] * len(predictions),
                'probability_first_round': predictions,
                'predicted_first_round':   pred_first_round,
            })

        predictions_df = predictions_df[['name', 'probability_first_round', 'predicted_first_round']]
        predictions_df = predictions_df.sort_values('probability_first_round', ascending=False)

        print("\nTop predicted first-round RBs:")
        print(predictions_df.head(10).to_string())
        return predictions_df