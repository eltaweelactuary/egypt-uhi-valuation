"""
Actuarial Loss Estimation - Machine Learning Models
Ensemble of gradient boosting models for claim cost prediction
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import warnings
warnings.filterwarnings('ignore')

# Optional imports
try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False

try:
    from catboost import CatBoostRegressor
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False


class ActuarialModel:
    """
    Ensemble model for actuarial loss prediction.
    Combines multiple gradient boosting models with different architectures.
    """
    
    def __init__(
        self,
        use_xgb: bool = True,
        use_lgb: bool = True,
        use_catboost: bool = True,
        n_folds: int = 5,
        random_state: int = 42
    ):
        """
        Initialize the ensemble model.
        
        Args:
            use_xgb: Whether to include XGBoost
            use_lgb: Whether to include LightGBM
            use_catboost: Whether to include CatBoost
            n_folds: Number of cross-validation folds
            random_state: Random seed
        """
        self.use_xgb = use_xgb and HAS_XGB
        self.use_lgb = use_lgb and HAS_LGB
        self.use_catboost = use_catboost and HAS_CATBOOST
        self.n_folds = n_folds
        self.random_state = random_state
        
        self.models = {}
        self.oof_predictions = {}
        self.feature_importance = {}
        
    def _get_xgb_params(self) -> dict:
        """Get XGBoost hyperparameters."""
        return {
            'objective': 'reg:absoluteerror',
            'eval_metric': 'mae',
            'max_depth': 6,
            'learning_rate': 0.05,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_weight': 3,
            'reg_alpha': 0.1,
            'reg_lambda': 0.1,
            'n_estimators': 1000,
            'early_stopping_rounds': 50,
            'random_state': self.random_state,
            'n_jobs': -1,
            'verbosity': 0
        }
    
    def _get_lgb_params(self) -> dict:
        """Get LightGBM hyperparameters."""
        return {
            'objective': 'mae',
            'metric': 'mae',
            'max_depth': 6,
            'learning_rate': 0.05,
            'num_leaves': 31,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_samples': 20,
            'reg_alpha': 0.1,
            'reg_lambda': 0.1,
            'n_estimators': 1000,
            'random_state': self.random_state,
            'n_jobs': -1,
            'verbose': -1
        }
    
    def _get_catboost_params(self) -> dict:
        """Get CatBoost hyperparameters."""
        return {
            'loss_function': 'MAE',
            'eval_metric': 'MAE',
            'depth': 6,
            'learning_rate': 0.05,
            'iterations': 1000,
            'early_stopping_rounds': 50,
            'random_state': self.random_state,
            'verbose': 0
        }
    
    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        cat_features: Optional[List[str]] = None
    ) -> 'ActuarialModel':
        """
        Fit the ensemble model using cross-validation.
        
        Args:
            X: Training features
            y: Target variable
            cat_features: List of categorical feature names (for CatBoost)
            
        Returns:
            self
        """
        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)
        
        # Initialize storage
        self.models = {'xgb': [], 'lgb': [], 'catboost': []}
        n_samples = len(X)
        
        self.oof_predictions = {
            'xgb': np.zeros(n_samples),
            'lgb': np.zeros(n_samples),
            'catboost': np.zeros(n_samples)
        }
        
        print(f"Training ensemble with {self.n_folds}-fold CV...")
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
            print(f"\nFold {fold + 1}/{self.n_folds}")
            
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            # XGBoost
            if self.use_xgb:
                print("  Training XGBoost...")
                xgb_params = self._get_xgb_params()
                early_stopping = xgb_params.pop('early_stopping_rounds')
                
                model_xgb = xgb.XGBRegressor(**xgb_params)
                model_xgb.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    verbose=False
                )
                
                self.models['xgb'].append(model_xgb)
                self.oof_predictions['xgb'][val_idx] = model_xgb.predict(X_val)
                
                xgb_mae = mean_absolute_error(y_val, self.oof_predictions['xgb'][val_idx])
                print(f"    XGB MAE: {xgb_mae:.4f}")
            
            # LightGBM
            if self.use_lgb:
                print("  Training LightGBM...")
                lgb_params = self._get_lgb_params()
                
                model_lgb = lgb.LGBMRegressor(**lgb_params)
                model_lgb.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    callbacks=[lgb.early_stopping(50, verbose=False)]
                )
                
                self.models['lgb'].append(model_lgb)
                self.oof_predictions['lgb'][val_idx] = model_lgb.predict(X_val)
                
                lgb_mae = mean_absolute_error(y_val, self.oof_predictions['lgb'][val_idx])
                print(f"    LGB MAE: {lgb_mae:.4f}")
            
            # CatBoost
            if self.use_catboost:
                print("  Training CatBoost...")
                catboost_params = self._get_catboost_params()
                
                model_cat = CatBoostRegressor(**catboost_params)
                model_cat.fit(
                    X_train, y_train,
                    eval_set=(X_val, y_val),
                    cat_features=cat_features,
                    verbose=0
                )
                
                self.models['catboost'].append(model_cat)
                self.oof_predictions['catboost'][val_idx] = model_cat.predict(X_val)
                
                cat_mae = mean_absolute_error(y_val, self.oof_predictions['catboost'][val_idx])
                print(f"    CatBoost MAE: {cat_mae:.4f}")
        
        # Calculate overall OOF scores
        print("\n" + "="*50)
        print("Overall Out-of-Fold MAE Scores:")
        
        active_models = []
        for name in ['xgb', 'lgb', 'catboost']:
            if len(self.models.get(name, [])) > 0:
                mae = mean_absolute_error(y, self.oof_predictions[name])
                print(f"  {name.upper()}: {mae:.4f}")
                active_models.append(name)
        
        # Ensemble prediction
        if len(active_models) > 0:
            ensemble_pred = np.mean([self.oof_predictions[m] for m in active_models], axis=0)
            ensemble_mae = mean_absolute_error(y, ensemble_pred)
            print(f"  ENSEMBLE: {ensemble_mae:.4f}")
        
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions using the ensemble.
        
        Args:
            X: Test features
            
        Returns:
            Ensemble predictions
        """
        predictions = []
        
        # XGBoost predictions
        if len(self.models.get('xgb', [])) > 0:
            xgb_preds = np.mean([m.predict(X) for m in self.models['xgb']], axis=0)
            predictions.append(xgb_preds)
        
        # LightGBM predictions
        if len(self.models.get('lgb', [])) > 0:
            lgb_preds = np.mean([m.predict(X) for m in self.models['lgb']], axis=0)
            predictions.append(lgb_preds)
        
        # CatBoost predictions
        if len(self.models.get('catboost', [])) > 0:
            cat_preds = np.mean([m.predict(X) for m in self.models['catboost']], axis=0)
            predictions.append(cat_preds)
        
        if len(predictions) == 0:
            raise ValueError("No models have been trained")
        
        return np.mean(predictions, axis=0)
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Get aggregated feature importance from all models."""
        
        importance_dfs = []
        
        # XGBoost importance
        if len(self.models.get('xgb', [])) > 0:
            xgb_imp = np.mean([m.feature_importances_ for m in self.models['xgb']], axis=0)
            importance_dfs.append(pd.DataFrame({
                'feature': self.models['xgb'][0].feature_names_in_,
                'importance': xgb_imp,
                'model': 'XGBoost'
            }))
        
        # LightGBM importance
        if len(self.models.get('lgb', [])) > 0:
            lgb_imp = np.mean([m.feature_importances_ for m in self.models['lgb']], axis=0)
            importance_dfs.append(pd.DataFrame({
                'feature': self.models['lgb'][0].feature_name_,
                'importance': lgb_imp,
                'model': 'LightGBM'
            }))
        
        if importance_dfs:
            return pd.concat(importance_dfs, ignore_index=True)
        
        return pd.DataFrame()


class SimpleBaselineModel:
    """
    Simple baseline models for comparison.
    Includes mean prediction and linear regression variants.
    """
    
    def __init__(self, model_type: str = 'ridge'):
        """
        Initialize baseline model.
        
        Args:
            model_type: One of 'mean', 'ridge', 'lasso', 'elasticnet', 'rf'
        """
        self.model_type = model_type
        self.model = None
        self.mean_target = None
        
    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'SimpleBaselineModel':
        """Fit the baseline model."""
        
        self.mean_target = y.mean()
        
        if self.model_type == 'mean':
            pass  # Just use mean
        elif self.model_type == 'ridge':
            self.model = Ridge(alpha=1.0)
            self.model.fit(X, y)
        elif self.model_type == 'lasso':
            self.model = Lasso(alpha=1.0)
            self.model.fit(X, y)
        elif self.model_type == 'elasticnet':
            self.model = ElasticNet(alpha=1.0, l1_ratio=0.5)
            self.model.fit(X, y)
        elif self.model_type == 'rf':
            self.model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
            self.model.fit(X, y)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions."""
        
        if self.model_type == 'mean':
            return np.full(len(X), self.mean_target)
        else:
            return self.model.predict(X)


def optimize_ensemble_weights(
    predictions: Dict[str, np.ndarray],
    y_true: np.ndarray,
    n_iter: int = 1000
) -> Dict[str, float]:
    """
    Find optimal ensemble weights using random search.
    
    Args:
        predictions: Dictionary of model predictions
        y_true: True target values
        n_iter: Number of random weight combinations to try
        
    Returns:
        Dictionary of optimal weights
    """
    model_names = list(predictions.keys())
    n_models = len(model_names)
    
    best_weights = None
    best_score = float('inf')
    
    for _ in range(n_iter):
        # Generate random weights
        weights = np.random.dirichlet(np.ones(n_models))
        
        # Calculate weighted ensemble prediction
        ensemble_pred = np.zeros_like(y_true, dtype=float)
        for i, name in enumerate(model_names):
            ensemble_pred += weights[i] * predictions[name]
        
        # Calculate MAE
        score = mean_absolute_error(y_true, ensemble_pred)
        
        if score < best_score:
            best_score = score
            best_weights = {name: w for name, w in zip(model_names, weights)}
    
    print(f"Best ensemble MAE: {best_score:.4f}")
    print("Optimal weights:")
    for name, weight in best_weights.items():
        print(f"  {name}: {weight:.4f}")
    
    return best_weights
