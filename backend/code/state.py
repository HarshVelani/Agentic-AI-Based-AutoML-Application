from typing import Dict, Any, List, Optional, Literal
from typing_extensions import TypedDict

class MLWorkflowState(TypedDict):
    """State management for the ML workflow"""
    # Input data
    session_id: str
    data_path: str
    target_column: str
    problem_type: Optional[str]
    user_comments: str
    
    
    # Analysis results
    data_schema: Dict[str, Any]
    inferred_problem_type: str 
    
    # Generated code and results
    pipeline_code: str
    pipeline_code_path: str
    execution_results: Dict[str, Any]
    model_path: str
    results_path: str
    summarized_result: str
    summarized_result_path: str
    best_model_name: str
    metrics: Dict[str, Any]
    
    # Tuning related
    tune_requested: bool
    tuning_code: str
    tuned_model_path: str
    
    # Messages and errors
    messages: List[str]
    errors: List[str]