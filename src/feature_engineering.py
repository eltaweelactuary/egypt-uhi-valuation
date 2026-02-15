"""
Actuarial Loss Estimation - Feature Engineering
Advanced feature generation for insurance claims prediction
"""

import pandas as pd
import numpy as np
from typing import List, Tuple, Optional
import re
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer


class FeatureEngineer:
    """Feature engineering pipeline for actuarial loss estimation."""
    
    def __init__(self, config: Optional[dict] = None):
        """
        Initialize feature engineer.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.tfidf = TfidfVectorizer(
            max_features=100,
            stop_words='english',
            ngram_range=(1, 2)
        )
        self.fitted = False
        
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fit and transform training data.
        
        Args:
            df: Training DataFrame
            
        Returns:
            Transformed DataFrame
        """
        df = df.copy()
        df = self._create_base_features(df)
        df = self._encode_categorical(df, fit=True)
        df = self._create_text_features(df, fit=True)
        df = self._create_interaction_features(df)
        df = self._create_aggregate_features(df)
        self.fitted = True
        return df
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform test data.
        
        Args:
            df: Test DataFrame
            
        Returns:
            Transformed DataFrame
        """
        if not self.fitted:
            raise ValueError("FeatureEngineer must be fitted before transform")
        
        df = df.copy()
        df = self._create_base_features(df)
        df = self._encode_categorical(df, fit=False)
        df = self._create_text_features(df, fit=False)
        df = self._create_interaction_features(df)
        df = self._create_aggregate_features(df)
        return df
    
    def _create_base_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create basic derived features."""
        
        # Date-based features (if date columns exist)
        date_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
        for col in date_cols:
            df[f'{col}_year'] = df[col].dt.year
            df[f'{col}_month'] = df[col].dt.month
            df[f'{col}_day'] = df[col].dt.day
            df[f'{col}_dayofweek'] = df[col].dt.dayofweek
            df[f'{col}_quarter'] = df[col].dt.quarter
        
        # Try to parse date columns that are strings
        potential_date_cols = ['DateTimeOfAccident', 'DateReported', 'DateOfBirth']
        for col in potential_date_cols:
            if col in df.columns and df[col].dtype == 'object':
                try:
                    df[col] = pd.to_datetime(df[col], errors='coerce')
                    df[f'{col}_year'] = df[col].dt.year
                    df[f'{col}_month'] = df[col].dt.month
                    df[f'{col}_dayofweek'] = df[col].dt.dayofweek
                except:
                    pass
        
        # Calculate age-related features
        if 'DateOfBirth' in df.columns and 'DateTimeOfAccident' in df.columns:
            df['DateOfBirth'] = pd.to_datetime(df['DateOfBirth'], errors='coerce')
            df['DateTimeOfAccident'] = pd.to_datetime(df['DateTimeOfAccident'], errors='coerce')
            df['AgeAtAccident'] = (df['DateTimeOfAccident'] - df['DateOfBirth']).dt.days / 365.25
        
        # Reporting delay
        if 'DateReported' in df.columns and 'DateTimeOfAccident' in df.columns:
            df['DateReported'] = pd.to_datetime(df['DateReported'], errors='coerce')
            df['ReportingDelay'] = (df['DateReported'] - df['DateTimeOfAccident']).dt.days
        
        # Weekly wages features (if exists)
        if 'WeeklyWages' in df.columns:
            df['WeeklyWages_log'] = np.log1p(df['WeeklyWages'])
            df['AnnualWages'] = df['WeeklyWages'] * 52
            df['DailyWages'] = df['WeeklyWages'] / 7
        
        # Part-time indicator
        if 'HoursWorkedPerWeek' in df.columns:
            df['IsPartTime'] = (df['HoursWorkedPerWeek'] < 35).astype(int)
            df['HourlyWage'] = df['WeeklyWages'] / df['HoursWorkedPerWeek'].replace(0, np.nan)
        
        # Initial case estimate features
        if 'InitialCaseEstimate' in df.columns:
            df['InitialCaseEstimate_log'] = np.log1p(df['InitialCaseEstimate'])
        
        # Initial incurred features
        if 'InitialIncurredCalimsCost' in df.columns:
            df['InitialIncurred_log'] = np.log1p(df['InitialIncurredCalimsCost'])
        
        return df
    
    def _encode_categorical(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Encode categorical variables."""
        
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Remove text columns from encoding
        text_cols = ['ClaimDescription', 'AccidentDescription', 'Description']
        categorical_cols = [c for c in categorical_cols if c not in text_cols]
        
        for col in categorical_cols:
            if fit:
                le = LabelEncoder()
                df[col] = df[col].fillna('MISSING')
                df[col] = le.fit_transform(df[col].astype(str))
                self.label_encoders[col] = le
            else:
                if col in self.label_encoders:
                    df[col] = df[col].fillna('MISSING')
                    # Handle unseen categories
                    le = self.label_encoders[col]
                    df[col] = df[col].astype(str).apply(
                        lambda x: le.transform([x])[0] if x in le.classes_ 
                        else -1
                    )
        
        return df
    
    def _create_text_features(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Extract features from text descriptions."""
        
        text_cols = ['ClaimDescription', 'AccidentDescription', 'Description']
        text_col = None
        
        for col in text_cols:
            if col in df.columns:
                text_col = col
                break
        
        if text_col is None:
            return df
        
        # Fill NaN with empty string
        df[text_col] = df[text_col].fillna('')
        
        # Basic text features
        df['text_length'] = df[text_col].apply(len)
        df['word_count'] = df[text_col].apply(lambda x: len(str(x).split()))
        df['avg_word_length'] = df[text_col].apply(
            lambda x: np.mean([len(w) for w in str(x).split()]) if len(str(x).split()) > 0 else 0
        )
        
        # Severity indicators from text
        severity_words = ['severe', 'serious', 'critical', 'permanent', 'fatal', 'death', 
                          'amputation', 'fracture', 'surgery', 'hospital']
        df['severity_score'] = df[text_col].apply(
            lambda x: sum(1 for w in severity_words if w.lower() in str(x).lower())
        )
        
        # Body part mentions
        body_parts = ['back', 'neck', 'shoulder', 'knee', 'head', 'hand', 'arm', 'leg', 
                      'foot', 'eye', 'spine', 'wrist', 'finger', 'ankle']
        df['body_parts_mentioned'] = df[text_col].apply(
            lambda x: sum(1 for bp in body_parts if bp.lower() in str(x).lower())
        )
        
        # TF-IDF features
        if fit:
            tfidf_matrix = self.tfidf.fit_transform(df[text_col])
        else:
            tfidf_matrix = self.tfidf.transform(df[text_col])
        
        tfidf_df = pd.DataFrame(
            tfidf_matrix.toarray(),
            columns=[f'tfidf_{i}' for i in range(tfidf_matrix.shape[1])],
            index=df.index
        )
        
        df = pd.concat([df, tfidf_df], axis=1)
        
        return df
    
    def _create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create interaction features between important variables."""
        
        # Age and wage interactions
        if 'AgeAtAccident' in df.columns and 'WeeklyWages' in df.columns:
            df['Age_Wage_Interaction'] = df['AgeAtAccident'] * df['WeeklyWages']
        
        # Reporting delay interactions
        if 'ReportingDelay' in df.columns:
            if 'InitialCaseEstimate' in df.columns:
                df['Delay_Estimate_Interaction'] = df['ReportingDelay'] * df['InitialCaseEstimate']
            if 'severity_score' in df.columns:
                df['Delay_Severity_Interaction'] = df['ReportingDelay'] * df['severity_score']
        
        return df
    
    def _create_aggregate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create aggregate/statistical features."""
        
        # Group by categorical columns and calculate statistics
        group_cols = ['Gender', 'MaritalStatus', 'DependentChildren', 'PartTimeFullTime']
        
        for col in group_cols:
            if col in df.columns and 'WeeklyWages' in df.columns:
                wage_stats = df.groupby(col)['WeeklyWages'].transform(['mean', 'std'])
                if isinstance(wage_stats, pd.DataFrame):
                    df[f'{col}_MeanWage'] = wage_stats['mean']
                    df[f'{col}_StdWage'] = wage_stats['std']
        
        return df
    
    def get_feature_names(self, df: pd.DataFrame) -> List[str]:
        """Get list of feature names (excluding target and ID columns)."""
        
        exclude_cols = ['ClaimNumber', 'UltimateIncurredClaimCost', 'ClaimDescription',
                        'AccidentDescription', 'Description', 'DateOfBirth', 
                        'DateTimeOfAccident', 'DateReported']
        
        return [c for c in df.columns if c not in exclude_cols]


def create_target_transform(y: pd.Series) -> Tuple[np.ndarray, float, float]:
    """
    Apply log transformation to target variable.
    
    Args:
        y: Target series
        
    Returns:
        Tuple of (transformed values, mean, std for inverse transform)
    """
    y_log = np.log1p(y)
    return y_log, y_log.mean(), y_log.std()


def inverse_target_transform(y_pred: np.ndarray) -> np.ndarray:
    """
    Inverse log transformation for predictions.
    
    Args:
        y_pred: Predicted values (log scale)
        
    Returns:
        Predictions in original scale
    """
    return np.expm1(y_pred)
