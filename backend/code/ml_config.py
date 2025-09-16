from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
import xgboost as xgb
from dataclasses import dataclass
from langchain_core.tools import tool


@dataclass
class MLConfig:
    """Configuration for ML models and parameters"""
    test_size: float = 0.2
    random_state: int = 42
    cv_folds: int = 5
    
    # Default models
    regression_models = {
        'LinearRegression': LinearRegression(),
        'RandomForestRegressor': RandomForestRegressor(random_state=42),
        'XGBRegressor': xgb.XGBRegressor(random_state=42, eval_metric='rmse')
    }
    
    classification_models = {
        'LogisticRegression': LogisticRegression(random_state=42, max_iter=1000),
        'RandomForestClassifier': RandomForestClassifier(random_state=42),
        'XGBClassifier': xgb.XGBClassifier(random_state=42, eval_metric='logloss')
    }