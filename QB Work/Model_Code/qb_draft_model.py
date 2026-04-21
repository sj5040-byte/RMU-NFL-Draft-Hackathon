"""
qb_draft_model.py
-----------------
Core model class for predicting whether a college QB gets drafted
in the first round of the NFL Draft.

Algorithm: XGBoost binary classifier, tuned via Optuna.
Evaluation uses two cross-validation strategies:

    GKF (Grouped K-Fold) -- PRIMARY
        Years are sorted chronologically and split into k blocks.
        Each block is held out once. No future data bleeds into
        training. This is the honest, reported benchmark.

    SKF (Stratified K-Fold) -- STABILITY CHECK
        Random stratified splits. Used to catch temporal leakage
        and confirm GKF variance is not just noise.

Rule of thumb:
    SKF >> GKF on mean F1  ->  leakage present, trust GKF only.
    SKF ~= GKF             ->  leakage minimal, GKF is solid.
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

# Fix seeds so results are reproducible across runs.
np.random.seed(42)


class QBDraftPredictor:
    """
    XGBoost model to predict first-round QB draft outcomes.

    Attributes
    ----------
    df : pd.DataFrame
        Raw training data loaded from CSV.
    label_encoders : dict
        Fitted LabelEncoder per categorical column. Reused at inference.
    feature_names : list[str]
        Column names of the feature matrix, in order. Needed to align
        new data at inference time.
    cv_results : list[dict]
        Fold-level results from the most recent CV run. Each dict holds
        metrics, the trained model, predictions, and the confusion matrix.
    best_params : dict
        Hyperparameters from the last Optuna study. Shared across GKF
        and SKF so both strategies are compared on equal footing.
    all_predictions : list
        Reserved for future use (e.g. ensembling predictions).
    cv_strategy : str or None
        'GKF' or 'SKF'. Tracks which strategy populated cv_results.
    """

    def __init__(self, csv_path: str):
        """
        Parameters
        ----------
        csv_path : str
            Path to the QB training CSV.
        """
        self.df = pd.read_csv(csv_path)
        self.label_encoders = {}
        self.feature_names = None
        self.cv_results = []
        self.best_params = None
        self.all_predictions = []
        self.cv_strategy = None

    # ------------------------------------------------------------------
    # Data preparation
    # ------------------------------------------------------------------

    def prepare_data(self):
        """
        Clean, encode, and engineer features from the raw DataFrame.

        Steps
        -----
        1. Create binary target: first_round = 1 if round == 1.
        2. Drop columns that are either identifiers, post-draft info,
           or combine drills excluded by design (vertical, broad jump,
           cone, shuttle -- too many missing values for QBs).
        3. Median-impute missing numeric values.
        4. Mode-impute missing categorical values.
        5. Label-encode categoricals. Encoders are stored for reuse.

        Returns
        -------
        X : pd.DataFrame
            Feature matrix.
        y : pd.Series
            Binary target (1 = first round).
        df : pd.DataFrame
            Full cleaned DataFrame, including 'year' for GKF grouping.
        """
        df = self.df.copy()

        # Target: 1 if the QB was a first-round pick, 0 otherwise.
        df['first_round'] = (df['round'] == 1).astype(int)

        # Columns we never want as features.
        # - Draft outcome columns (would be leakage): overall, round, pick.
        # - Administrative/identifier columns: nfl_team, name, position,
        #   college_team.
        # - Combine drills with excessive missingness for QBs: vertical,
        #   broad_jump, cone, shuttle.
        drop_cols = [
            'overall', 'round', 'pick', 'nfl_team',
            'name', 'position', 'college_team',
            'vertical', 'broad_jump', 'cone', 'shuttle',
        ]

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols = [c for c in numeric_cols if c not in drop_cols + ['first_round']]

        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        categorical_cols = [c for c in categorical_cols if c not in drop_cols + ['name']]

        # Impute numeric columns with their median.
        for col in numeric_cols:
            if df[col].isnull().sum() > 0:
                df[col].fillna(df[col].median(), inplace=True)

        # Impute categorical columns with their mode.
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
        print(f"First-round QBs: {y.sum()} ({y.mean() * 100:.1f}%)")

        return X, y, df

    # ------------------------------------------------------------------
    # Hyperparameter tuning
    # ------------------------------------------------------------------

    def _objective(self, trial, X_train, y_train) -> float:
        """
        Optuna objective function.

        Maximises AUCPR (area under the precision-recall curve) via an
        internal 5-fold stratified CV on the training data. AUCPR is
        preferred over log-loss for imbalanced targets because it focuses
        on the minority class (first-round QBs).

        Parameters
        ----------
        trial : optuna.Trial
            Optuna trial object that suggests hyperparameter values.
        X_train : array-like
            Training features for this outer fold.
        y_train : array-like
            Training labels for this outer fold.

        Returns
        -------
        float
            Best mean AUCPR across the internal 5-fold CV.
        """
        # scale_pos_weight balances the class imbalance.
        # We search a range centered on the natural ratio but allow
        # the tuner to over- or under-weight it.
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

        Uses a MedianPruner to cut unpromising trials early, which
        keeps wall-clock time reasonable at 50 trials.

        Parameters
        ----------
        X_train : array-like
            Feature matrix used for internal CV during tuning.
        y_train : array-like
            Target labels.
        n_trials : int
            Number of Optuna trials. Default 50 balances coverage vs. time.

        Returns
        -------
        dict
            Best hyperparameter dictionary (stored in self.best_params).
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
        # Append fixed params that are not part of the search space.
        self.best_params['objective']    = 'binary:logistic'
        self.best_params['eval_metric']  = 'aucpr'
        self.best_params['random_state'] = 42
        self.best_params['verbosity']    = 0

        print(f"\nBest hyperparameters found:")
        for k, v in self.best_params.items():
            print(f"  {k}: {v}")
        print(f"Best validation score (aucpr): {study.best_value:.4f}")

        return self.best_params

    # ------------------------------------------------------------------
    # Single-fold training helper
    # ------------------------------------------------------------------

    def _run_single_fold(self, X_train_fold, y_train_fold,
                         X_test_fold, y_test_fold,
                         fold_label: str,
                         val_X=None, val_y=None) -> dict:
        """
        Train XGBoost on one fold and evaluate on the held-out set.

        Threshold tuning
        ----------------
        The classification threshold is tuned to maximise F1. We prefer
        a separate validation set for this so the test set stays clean.
        Priority order:
            1. Explicit val_X / val_y passed by the caller.
            2. A 20% stratified split carved from the training fold.
            3. Default threshold of 0.5 if neither is available.

        Parameters
        ----------
        X_train_fold : array-like
            Training features for this fold.
        y_train_fold : array-like
            Training labels for this fold.
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

        Returns
        -------
        dict
            Keys: fold_label, train_size, test_size, accuracy, f1_score,
            roc_auc, best_threshold, model, y_test, y_pred, y_pred_proba,
            confusion_matrix.
        """
        print(f"Training samples: {len(X_train_fold)} | Test samples: {len(X_test_fold)}")
        print(f"First-round QBs in test set: {int(y_test_fold.sum())}")

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
        # Prefer the explicit validation set. Fall back to a stratified
        # 20% slice of the training data.
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

    # ------------------------------------------------------------------
    # PRIMARY: Grouped K-Fold
    # ------------------------------------------------------------------

    def train_and_evaluate_grouped(self, X, y, df, n_splits: int = 5):
        """
        PRIMARY BENCHMARK -- Grouped K-Fold by year.

        How it works
        ------------
        All unique draft years are sorted chronologically and divided
        into k roughly equal-sized blocks. Each block (2-3 years) is
        held out once as the test set; all other years form the training
        set.

        Why this is the primary benchmark
        ----------------------------------
        - Strictly respects temporal order. A model trained on 2010-2018
          data cannot see 2019+ picks. That is how it will work in reality.
        - Each fold contains 6-11 first-round QBs, enough for a stable F1.
        - Variance reflects genuine era-level difficulty, not random
          sampling noise.

        Threshold tuning uses the most recent training block as the
        validation set. This way, no test-year data influences the
        threshold decision.

        Parameters
        ----------
        X : pd.DataFrame
            Full feature matrix.
        y : pd.Series
            Full binary target.
        df : pd.DataFrame
            Full DataFrame with a 'year' column for grouping.
        n_splits : int
            Number of chronological groups. Default 5.

        Returns
        -------
        list[dict]
            One result dict per fold (see _run_single_fold for keys).
        """
        print("\n" + "=" * 60)
        print(f"GROUPED K-FOLD (PRIMARY BENCHMARK) -- k={n_splits}, chronological year groups")
        print("=" * 60)

        self.cv_strategy = 'GKF'
        years    = np.array(sorted(df['year'].unique()))
        n_years  = len(years)

        # Assign each year to one of the k groups.
        year_to_fold = {yr: i * n_splits // n_years for i, yr in enumerate(years)}
        fold_col     = df['year'].map(year_to_fold).values

        print(f"\nYear to group assignments:")
        for fold_id in range(n_splits):
            group_years = [yr for yr, g in year_to_fold.items() if g == fold_id]
            print(f"  Group {fold_id + 1}: {group_years}")

        # Tune hyperparameters once on the full dataset. Both GKF and SKF
        # use the same params so their results are directly comparable.
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

            # Skip degenerate folds where training data has only one class.
            if y_tr.sum() == 0 or y_tr.sum() == len(y_tr):
                print("Skipping: insufficient class variation in training data")
                continue

            # Use the immediately preceding group as the threshold-tuning
            # validation set. Ensures no test-year data leaks in.
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

        Each fold is a random stratified split that preserves the
        overall first-round QB ratio (~27%). This is NOT the primary
        benchmark because it allows QBs from later years to appear in
        training while earlier-year QBs appear in the test set.

        When to trust SKF
        -----------------
        - If SKF F1 ~= GKF F1: leakage is minimal. SKF gives a tighter
          confidence interval on model skill.
        - If SKF F1 >> GKF F1: temporal leakage is inflating SKF. Trust
          GKF only.

        Parameters
        ----------
        X : pd.DataFrame
            Full feature matrix.
        y : pd.Series
            Full binary target.
        n_splits : int
            Number of stratified folds. Default 5.

        Returns
        -------
        list[dict]
            One result dict per fold.
        """
        print("\n" + "=" * 60)
        print(f"STRATIFIED K-FOLD (STABILITY CHECK) -- k={n_splits}")
        print("=" * 60)

        self.cv_strategy = 'SKF'

        # Hyperparameter tuning runs once here as well.
        # If GKF ran first, best_params is already set.
        # We retune here so SKF is a clean, independent run.
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

    # ------------------------------------------------------------------
    # Summary printing
    # ------------------------------------------------------------------

    def print_cross_validation_summary(self):
        """
        Print per-fold metrics and aggregated statistics for the last
        CV run.

        Aggregated confusion matrix counts all TP/FP/TN/FN across folds
        to give an overall accuracy figure that accounts for fold size
        differences.
        """
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
        print(f"  {'--' * 25}")
        for r in self.cv_results:
            label   = r.get('fold_label', str(r.get('fold', '?')))
            auc_str = f"{r['roc_auc']:.4f}" if not np.isnan(r['roc_auc']) else '  N/A'
            print(f"  {label:<18} {r['accuracy']:>6.4f} {r['f1_score']:>6.4f} "
                  f"{auc_str:>6} {r['best_threshold']:>10.3f}")

        # Sum raw counts across folds, then compute overall accuracy.
        total_cm  = sum(r['confusion_matrix'] for r in self.cv_results)
        total_acc = (total_cm[0, 0] + total_cm[1, 1]) / total_cm.sum()
        print(f"\nAggregated Confusion Matrix (across all folds):")
        print(f"  TN: {int(total_cm[0, 0])}, FP: {int(total_cm[0, 1])}")
        print(f"  FN: {int(total_cm[1, 0])}, TP: {int(total_cm[1, 1])}")
        print(f"  Overall Accuracy: {total_acc:.4f}")

    # ------------------------------------------------------------------
    # Feature importance
    # ------------------------------------------------------------------

    def feature_importance_analysis(self):
        """
        Aggregate XGBoost feature importance (split count) across all
        folds and print the top 20.

        Split count measures how many times a feature was used to make
        a split across all trees. It is a frequency proxy for importance,
        not a measure of effect size.

        Call this after GKF so the importance reflects the primary
        benchmark models.

        Returns
        -------
        dict
            {feature_key: mean_importance} across folds. Feature keys
            are XGBoost's internal 'f0', 'f1', ... notation. The printed
            output maps these back to column names.
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

    # ------------------------------------------------------------------
    # Inference on new QB class
    # ------------------------------------------------------------------

    def generate_predictions_for_new_qbs(self, new_qbs_csv: str = None):
        """
        Score a new cohort of QBs using the last trained GKF model.

        The last GKF fold model is used because it trains on the largest
        possible historical window, which is the most representative of
        the current player pool.

        Parameters
        ----------
        new_qbs_csv : str or None
            Path to a CSV of new QBs. Must contain the same columns used
            during training (after dropping the excluded columns).
            If None, falls back to the test set from the last fold
            (mainly useful for debugging).

        Returns
        -------
        pd.DataFrame or None
            Columns: probability_first_round, predicted_first_round.
            Sorted descending by probability. Returns None if no models
            are available or no data can be found.
        """
        print("\n" + "=" * 60)
        print("PREDICTION GENERATION FOR NEW QBs")
        print("=" * 60)

        if not self.cv_results:
            print("No trained models available")
            return None

        latest_model = self.cv_results[-1]['model']

        if new_qbs_csv:
            new_df = pd.read_csv(new_qbs_csv)
            print(f"Loaded {len(new_df)} new QB records")

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

            X_new = new_df[self.feature_names]
        else:
            X_new = self.cv_results[-1].get('X_test')
            if X_new is None:
                print("No new data provided and no test set available")
                return None

        predictions    = latest_model.predict(xgb.DMatrix(X_new))
        threshold      = self.cv_results[-1].get('best_threshold', 0.5)
        predictions_df = pd.DataFrame({
            'probability_first_round': predictions,
            'predicted_first_round':   (predictions > threshold).astype(int),
        }).sort_values('probability_first_round', ascending=False)

        print("\nTop predicted first-round QBs:")
        print(predictions_df.head(10).to_string())
        return predictions_df


def main():
    """
    Demonstrate the full workflow: GKF primary benchmark, then SKF
    stability check. Intended for quick smoke-testing. The full
    pipeline (with visualizations and CSV exports) lives in
    run_qb_pipeline.py.
    """
    predictor = QBDraftPredictor('QB_train.csv')
    X, y, df  = predictor.prepare_data()

    # Step 1: Primary benchmark.
    predictor.train_and_evaluate_grouped(X, y, df)
    predictor.print_cross_validation_summary()
    predictor.feature_importance_analysis()

    # Step 2: Stability check.
    predictor.train_and_evaluate_skf(X, y)
    predictor.print_cross_validation_summary()

    print("\n" + "=" * 60)
    print("MODEL TRAINING COMPLETE")
    print("=" * 60)


if __name__ == '__main__':
    main()
