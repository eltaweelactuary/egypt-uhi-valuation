# Actuarial Pricing Engine - Core Module

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# DATA CLASSES FOR CONFIGURATION
# =============================================================================

@dataclass
class PricingConfig:
    """Dynamic pricing parameters - adjustable via dashboard sliders"""
    expense_loading: float = 0.25
    profit_margin: float = 0.10
    contingency_margin: float = 0.05
    reinsurance_cost: float = 0.03
    commission_rate: float = 0.15
    
    @property
    def total_loading(self) -> float:
        return (self.expense_loading + self.profit_margin + 
                self.contingency_margin + self.reinsurance_cost + 
                self.commission_rate)


@dataclass
class ModelConfig:
    """Model training configuration"""
    use_xgboost: bool = True
    use_lightgbm: bool = True
    use_catboost: bool = True
    n_folds: int = 5
    n_estimators: int = 500
    max_depth: int = 6
    learning_rate: float = 0.05
    random_state: int = 42


# =============================================================================
# FEATURE ENGINEERING (Fixes weakness #1: Date features)
# =============================================================================

def extract_date_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract actuarially-relevant features from date columns.
    Fixes: Previously dates were ignored entirely.
    """
    df = df.copy()
    
    # List of potential date column pairs
    date_columns = {
        'accident': ['DateTimeOfAccident', 'DateOfAccident'],
        'reported': ['DateReported'],
        'birth': ['DateOfBirth']
    }
    
    # Find actual columns in the dataframe
    def find_date_col(candidates):
        for col in candidates:
            if col in df.columns:
                return col
        return None
    
    accident_col = find_date_col(date_columns['accident'])
    reported_col = find_date_col(date_columns['reported'])
    birth_col = find_date_col(date_columns['birth'])
    
    # Parse dates
    if accident_col:
        df['_accident_date'] = pd.to_datetime(df[accident_col], errors='coerce')
        # Extract seasonality
        df['AccidentMonth'] = df['_accident_date'].dt.month
        df['AccidentQuarter'] = df['_accident_date'].dt.quarter
        df['AccidentDayOfWeek'] = df['_accident_date'].dt.dayofweek
        
    if reported_col:
        df['_reported_date'] = pd.to_datetime(df[reported_col], errors='coerce')
        
    if birth_col:
        df['_birth_date'] = pd.to_datetime(df[birth_col], errors='coerce')
    
    # Calculate Age at Accident
    if birth_col and accident_col:
        if '_birth_date' in df.columns and '_accident_date' in df.columns:
            df['AgeAtAccident'] = (
                (df['_accident_date'] - df['_birth_date']).dt.days / 365.25
            ).round(0)
            df['AgeAtAccident'] = df['AgeAtAccident'].clip(0, 120)  # Cap outliers
    
    # Calculate Reporting Lag (days between accident and report)
    if accident_col and reported_col:
        if '_accident_date' in df.columns and '_reported_date' in df.columns:
            df['ReportingLagDays'] = (
                (df['_reported_date'] - df['_accident_date']).dt.days
            )
            df['ReportingLagDays'] = df['ReportingLagDays'].clip(0, 365)  # Cap outliers
    
    # Drop temporary columns
    temp_cols = ['_accident_date', '_reported_date', '_birth_date']
    df = df.drop(columns=[c for c in temp_cols if c in df.columns], errors='ignore')
    
    return df


# =============================================================================
# DATA CLEANING (Fixes weakness #2: Advanced imputation)
# =============================================================================

def clean_data(df: pd.DataFrame, use_native_handling: bool = True) -> pd.DataFrame:
    """
    Clean data with improved missing value handling.
    
    Args:
        df: Input DataFrame
        use_native_handling: If True, use -999 for native model handling (XGBoost/LightGBM/CatBoost)
                            If False, use median imputation
    """
    df = df.copy()
    
    for col in df.select_dtypes(include=[np.number]).columns:
        if df[col].isnull().any():
            if use_native_handling:
                # Models like XGBoost/LightGBM handle missing values better than manual imputation
                df[col] = df[col].fillna(-999)
            else:
                df[col] = df[col].fillna(df[col].median())
    
    for col in df.select_dtypes(include=['object']).columns:
        if df[col].isnull().any():
            df[col] = df[col].fillna('MISSING')
    
    return df


# =============================================================================
# MODEL TRAINING
# =============================================================================

def prepare_features(train_df: pd.DataFrame, test_df: pd.DataFrame, 
                     target_col: str, exclude_cols: List[str] = None) -> Tuple:
    """Prepare features for modeling"""
    
    if exclude_cols is None:
        exclude_cols = [target_col, 'ClaimNumber', 'id', 'ClaimDescription', 
                       'AccidentDescription', 'DateTimeOfAccident', 
                       'DateReported', 'DateOfBirth', 'DateOfAccident']
    
    feature_cols = [c for c in train_df.columns 
                   if c not in exclude_cols and c in test_df.columns]
    
    train_clean = train_df.copy()
    test_clean = test_df.copy()
    
    # Label encode categorical columns
    for col in train_clean[feature_cols].select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        all_vals = pd.concat([train_clean[col], test_clean[col]]).astype(str).unique()
        le.fit(all_vals)
        train_clean[col] = le.transform(train_clean[col].astype(str))
        test_clean[col] = le.transform(test_clean[col].astype(str))
    
    X = train_clean[feature_cols].fillna(-999)
    X_test = test_clean[feature_cols].fillna(-999)
    y = np.log1p(train_clean[target_col])
    
    return X, X_test, y, feature_cols


def train_models(X: pd.DataFrame, y: pd.Series, X_test: pd.DataFrame,
                config: ModelConfig, progress_callback=None) -> Dict:
    """
    Train ensemble of models with cross-validation.
    
    Returns:
        Dictionary with model predictions and metrics
    """
    # Import here to avoid issues if not installed
    import xgboost as xgb
    import lightgbm as lgb
    from catboost import CatBoostRegressor
    
    kf = KFold(n_splits=config.n_folds, shuffle=True, random_state=config.random_state)
    results = {}
    
    models_to_train = []
    if config.use_xgboost:
        models_to_train.append('xgboost')
    if config.use_lightgbm:
        models_to_train.append('lightgbm')
    if config.use_catboost:
        models_to_train.append('catboost')
    
    for model_name in models_to_train:
        oof = np.zeros(len(X))
        pred = np.zeros(len(X_test))
        fold_scores = []
        
        for fold, (tr_idx, val_idx) in enumerate(kf.split(X)):
            X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
            y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]
            
            if model_name == 'xgboost':
                model = xgb.XGBRegressor(
                    n_estimators=config.n_estimators, 
                    max_depth=config.max_depth,
                    learning_rate=config.learning_rate,
                    random_state=config.random_state, 
                    verbosity=0
                )
                model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
                
            elif model_name == 'lightgbm':
                model = lgb.LGBMRegressor(
                    n_estimators=config.n_estimators,
                    max_depth=config.max_depth,
                    learning_rate=config.learning_rate,
                    random_state=config.random_state,
                    verbose=-1
                )
                model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)],
                         callbacks=[lgb.early_stopping(50, verbose=False)])
                
            else:  # catboost
                model = CatBoostRegressor(
                    iterations=config.n_estimators,
                    depth=config.max_depth,
                    learning_rate=config.learning_rate,
                    random_state=config.random_state,
                    verbose=0
                )
                model.fit(X_tr, y_tr, eval_set=(X_val, y_val), early_stopping_rounds=50)
            
            oof[val_idx] = model.predict(X_val)
            pred += model.predict(X_test) / config.n_folds
            fold_mae = mean_absolute_error(y_val, oof[val_idx])
            fold_scores.append(fold_mae)
            
            if progress_callback:
                progress_callback(model_name, fold + 1, config.n_folds, fold_mae)
        
        oof_mae = mean_absolute_error(y, oof)
        results[model_name] = {
            'oof': oof,
            'pred': pred,
            'fold_scores': fold_scores,
            'oof_mae': oof_mae,
            'model': model,
            'feature_importance': model.feature_importances_
        }
    
    return results


# =============================================================================
# PRICING CALCULATIONS (Fixes weakness #4: Dynamic pricing)
# =============================================================================

def calculate_premiums(predictions: np.ndarray, config: PricingConfig,
                      risk_scores: np.ndarray = None) -> Dict:
    """
    Calculate premiums with dynamic risk-based adjustments.
    
    Args:
        predictions: Log-transformed predicted losses
        config: Pricing configuration
        risk_scores: Optional per-claim risk scores for dynamic margin adjustment
    """
    expected_loss = np.expm1(predictions)
    
    # Base premium calculation
    base_premium = expected_loss * (1 + config.total_loading)
    
    # Dynamic margin adjustment based on prediction uncertainty (if risk scores provided)
    if risk_scores is not None:
        # Higher risk = higher contingency margin
        risk_adjustment = 1 + (risk_scores * config.contingency_margin)
        final_premium = base_premium * risk_adjustment
    else:
        final_premium = base_premium
    
    # Apply min/max bounds
    final_premium = np.clip(final_premium, 100, 1_000_000)
    
    return {
        'expected_loss': expected_loss,
        'base_premium': base_premium,
        'final_premium': final_premium,
        'mean_premium': np.mean(final_premium),
        'median_premium': np.median(final_premium),
        'min_premium': np.min(final_premium),
        'max_premium': np.max(final_premium),
        'total_loading_pct': config.total_loading * 100
    }


def sensitivity_analysis(predictions: np.ndarray, base_config: PricingConfig,
                        param_name: str, values: List[float]) -> pd.DataFrame:
    """
    Perform sensitivity analysis by varying a pricing parameter.
    
    Args:
        predictions: Log-transformed predicted losses
        base_config: Base pricing configuration
        param_name: Name of parameter to vary
        values: List of values to test
    """
    results = []
    
    for val in values:
        config = PricingConfig(
            expense_loading=base_config.expense_loading,
            profit_margin=base_config.profit_margin,
            contingency_margin=base_config.contingency_margin,
            reinsurance_cost=base_config.reinsurance_cost,
            commission_rate=base_config.commission_rate
        )
        setattr(config, param_name, val)
        
        premium_result = calculate_premiums(predictions, config)
        results.append({
            'parameter': param_name,
            'value': val,
            'mean_premium': premium_result['mean_premium'],
            'total_loading': config.total_loading * 100
        })
    
    return pd.DataFrame(results)
