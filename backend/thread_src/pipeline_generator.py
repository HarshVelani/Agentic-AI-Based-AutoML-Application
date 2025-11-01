"""
Thread-Safe Pipeline Generator with Session Tracking
"""

from thread_src.state import MLWorkflowState
from thread_src.llm_manager import ThreadSafeLLMManager
from thread_src.session_manager import ThreadSafeSessionManager, SessionStatus
from prompts.problem_classifier import problem_classifier_prompt
from prompts.pipeline_generator import pipeline_generator_prompt
from prompts.hyperparameter_tuning import hyperparameter_tuning_prompt
from prompts.result_summarizer import result_summarizer_prompt

import pandas as pd
import numpy as np
import os
import time
import json
import logging
import threading
from typing import Literal
from pathlib import Path

# ML Libraries
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix,
    mean_squared_error, mean_absolute_error, r2_score
)
from sklearn.preprocessing import LabelEncoder, StandardScaler
import xgboost as xgb
import pickle

logger = logging.getLogger(__name__)


class PipelineGenerator:
    """Thread-safe pipeline generator with session tracking"""
    
    def __init__(self, session_manager: ThreadSafeSessionManager):
        """
        Initialize with session manager
        
        Args:
            session_manager: ThreadSafeSessionManager instance
        """
        self.session_manager = session_manager
        self.MLWorkflowState = MLWorkflowState()
        self.llm = ThreadSafeLLMManager()
        
        # Thread lock for file operations
        self._file_lock = threading.Lock()

    def _update_session_progress(self, session_id: str, progress: float, step: str):
        """Helper to update session progress (thread-safe)"""
        self.session_manager.update_session(
            session_id=session_id,
            progress=progress,
            current_step=step
        )

    def _schema_analyzer_agent(self, state: MLWorkflowState) -> MLWorkflowState:
        """Analyze dataset schema and characteristics (thread-safe)"""
        session_id = state["session_id"]
        thread_name = threading.current_thread().name
        
        try:
            logger.info(f"[{thread_name}] Schema Analyzer invoked for session {session_id}")
            self._update_session_progress(session_id, 0.1, "Analyzing dataset schema")
            
            # Load data (support Excel now)
            file_path = state["data_path"]
            if file_path.endswith('.xlsx') or file_path.endswith('.xls'):
                df = pd.read_excel(file_path, engine='openpyxl')
            elif file_path.endswith('.csv'):
                df = pd.read_csv(file_path)
            else:
                raise ValueError("Unsupported file format. Use Excel or CSV.")
            
            # Analyze schema
            schema_info = {
                "shape": df.shape,
                "columns": list(df.columns),
                "dtypes": {str(k): str(v) for k, v in df.dtypes.to_dict().items()},
                "missing_values": {str(k): int(v) for k, v in df.isnull().sum().to_dict().items()},
                "numeric_columns": list(df.select_dtypes(include=[np.number]).columns),
                "categorical_columns": list(df.select_dtypes(include=['object', 'category']).columns),
                "target_column_info": {
                    "dtype": str(df[state["target_column"]].dtype),
                    "unique_values": int(len(df[state["target_column"]].unique())),
                    "sample_values": df[state["target_column"]].head().tolist()
                }
            }
            
            state["data_schema"] = schema_info
            state["messages"].append(f"✅ Dataset analyzed: {df.shape[0]} rows, {df.shape[1]} columns")
            
            self._update_session_progress(session_id, 0.2, "Schema analysis completed")
            logger.info(f"[{thread_name}] Schema analyzed for session {session_id}")

        except Exception as e:
            error_msg = f"Schema analysis failed: {str(e)}"
            state["errors"].append(error_msg)
            logger.error(f"[{thread_name}] Session {session_id}: {error_msg}")
            self.session_manager.update_session(
                session_id=session_id,
                error_message=error_msg
            )
        
        return state
    
    def _problem_classifier_agent(self, state: MLWorkflowState) -> MLWorkflowState:
        """Classify problem type (thread-safe)"""
        session_id = state["session_id"]
        thread_name = threading.current_thread().name
        
        try:
            logger.info(f"[{thread_name}] Problem Classifier invoked for session {session_id}")
            self._update_session_progress(session_id, 0.3, "Classifying problem type")

            if state.get("problem_type"):
                state["inferred_problem_type"] = state["problem_type"]
                state["messages"].append(f"✅ Using user-specified problem type: {state['problem_type']}")
            else:
                target_info = state["data_schema"]["target_column_info"]
                
                response = self.llm.invoke(problem_classifier_prompt, target_info=target_info)
                problem_type = response.strip().lower()

                if "regression" in problem_type:
                    state["inferred_problem_type"] = "regression"
                else:
                    state["inferred_problem_type"] = "classification"
                
                state["messages"].append(f"✅ Inferred problem type: {state['inferred_problem_type']}")
            
            self._update_session_progress(session_id, 0.35, f"Problem type identified: {state['inferred_problem_type']}")
            logger.info(f"[{thread_name}] Problem type for session {session_id}: {state['inferred_problem_type']}")
        
        except Exception as e:
            error_msg = f"Problem classification failed: {str(e)}"
            state["errors"].append(error_msg)
            state["inferred_problem_type"] = "classification"  # Default fallback
            logger.error(f"[{thread_name}] Session {session_id}: {error_msg}")
        
        return state
    
    def _pipeline_generator_agent(self, state: MLWorkflowState) -> MLWorkflowState:
        """Generate ML pipeline code using LLM (thread-safe)"""
        session_id = state["session_id"]
        thread_name = threading.current_thread().name
        
        try:
            logger.info(f"[{thread_name}] Pipeline Generator invoked for session {session_id}")
            self._update_session_progress(session_id, 0.4, "Generating ML pipeline code")

            schema = state["data_schema"]
            problem_type = state["inferred_problem_type"]
            target_col = state["target_column"]
            data_path = state["data_path"]
            user_comments = state["user_comments"]
            
            response = self.llm.invoke(
                pipeline_generator_prompt, 
                schema=schema, 
                problem_type=problem_type, 
                target_col=target_col, 
                data_path=data_path, 
                user_comments=user_comments,
                session_id=session_id
            )
            
            state["pipeline_code"] = response.strip()
            
            # Clean the code (remove markdown formatting if present)
            code = state["pipeline_code"]
            if "```python" in code:
                code = code.split("```python")[1].split("```")[0]
            elif "```" in code:
                code = code.split("```")[1].split("```")[0]
            
            # Save generated code (thread-safe file write)
            path = f'generated_code/{session_id}_code.py'
            with self._file_lock:
                Path('generated_code').mkdir(exist_ok=True)
                with open(path, 'w', encoding='utf-8') as f:
                    f.write(code)
            
            time.sleep(1)  # Reduced sleep time
            
            # Update state with cleaned code and paths
            state["model_path"] = f'model/{session_id}_{problem_type}_models.zip'
            state["results_path"] = f'results/{session_id}_{problem_type}_results.json'
            state["pipeline_code_path"] = path
            state["pipeline_code"] = code.strip()
            state["messages"].append("✅ ML pipeline code generated")
            
            self._update_session_progress(session_id, 0.5, "Pipeline code generated")
            logger.info(f"[{thread_name}] Pipeline code generated for session {session_id}")
            
        except Exception as e:
            error_msg = f"Pipeline generation failed: {str(e)}"
            state["errors"].append(error_msg)
            logger.error(f"[{thread_name}] Session {session_id}: {error_msg}")
        
        return state
    
    def _code_executor_node(self, state: MLWorkflowState) -> MLWorkflowState:
        """Execute the generated ML pipeline code (thread-safe)"""
        session_id = state["session_id"]
        thread_name = threading.current_thread().name
        
        try:
            logger.info(f"[{thread_name}] Code Executor invoked for session {session_id}")
            self._update_session_progress(session_id, 0.6, "Executing ML pipeline")

            # Execute code in subprocess to avoid threading issues
            code_file = f"generated_code/{session_id}_code.py"
            exit_code = os.system(f"python {code_file}")
            
            if exit_code != 0:
                raise Exception(f"Code execution failed with exit code {exit_code}")

            # Read results (thread-safe)
            filepath = state["results_path"]
            max_retries = 5
            for attempt in range(max_retries):
                try:
                    if os.path.exists(filepath):
                        with open(filepath, 'r') as f:
                            results = json.load(f)
                            state["execution_results"] = results
                            state["messages"].append("✅ Pipeline executed successfully")
                        break
                except (FileNotFoundError, json.JSONDecodeError) as e:
                    if attempt < max_retries - 1:
                        time.sleep(1)
                    else:
                        raise Exception(f"Failed to read results after {max_retries} attempts: {e}")
            
            self._update_session_progress(session_id, 0.7, "Pipeline execution completed")
            logger.info(f"[{thread_name}] Pipeline executed for session {session_id}")

        except Exception as e:
            error_msg = f"Code execution failed: {str(e)}"
            state["errors"].append(error_msg)
            logger.error(f"[{thread_name}] Session {session_id}: {error_msg}")

        return state

    def _best_model(self, state: MLWorkflowState) -> MLWorkflowState:
        """Identify the best model from results (thread-safe)"""
        session_id = state["session_id"]
        thread_name = threading.current_thread().name
        
        try:
            self._update_session_progress(session_id, 0.75, "Identifying best model")
            
            results = state["execution_results"]
            problem_type = state["inferred_problem_type"]

            if problem_type == "regression":
                best_model_name = min(results, key=lambda x: results[x].get("rmse") or results[x].get("RMSE", float('inf')))
            else:
                best_model_name = max(results, key=lambda x: results[x].get("accuracy", 0))
            
            state["best_model_name"] = best_model_name
            state["messages"].append(f"✅ Best model identified: {best_model_name}")
            
            logger.info(f"[{thread_name}] Best model for session {session_id}: {best_model_name}")

        except Exception as e:
            error_msg = f"Best model identification failed: {str(e)}"
            state["errors"].append(error_msg)
            logger.error(f"[{thread_name}] Session {session_id}: {error_msg}")

        return state

    def _result_summarizer(self, state: MLWorkflowState) -> MLWorkflowState:
        """Summarize key results from the workflow (thread-safe)"""
        session_id = state["session_id"]
        thread_name = threading.current_thread().name
        
        try:
            logger.info(f"[{thread_name}] Result Summarizer invoked for session {session_id}")
            self._update_session_progress(session_id, 0.85, "Summarizing results")

            data_schema = state["data_schema"]
            results = state["execution_results"]
            best_model = state["best_model_name"]
            
            response = self.llm.invoke(
                result_summarizer_prompt, 
                data_schema=data_schema,
                results=results,
                best_model=best_model
            )
            
            state["summarized_result"] = response.strip()

            # Save summary to .md file (thread-safe)
            path = f'ai_summary/{session_id}_summary.md'
            with self._file_lock:
                Path('ai_summary').mkdir(exist_ok=True)
                with open(path, 'w', encoding='utf-8') as f:
                    f.write(response)

            state["summarized_result_path"] = path
            state["messages"].append("✅ Results summarized")
            
            self._update_session_progress(session_id, 0.9, "Results summarization completed")
            logger.info(f"[{thread_name}] Results summarized for session {session_id}")
        
        except Exception as e:
            error_msg = f"Result summarization failed: {str(e)}"
            state["errors"].append(error_msg)
            logger.error(f"[{thread_name}] Session {session_id}: {error_msg}")

        return state
    
    def _tuning_agent(self, state: MLWorkflowState) -> MLWorkflowState:
        """Perform hyperparameter tuning if requested (thread-safe)"""
        session_id = state["session_id"]
        thread_name = threading.current_thread().name
        
        try:
            logger.info(f"[{thread_name}] Tuning Agent invoked for session {session_id}")
            self._update_session_progress(session_id, 0.95, "Performing hyperparameter tuning")

            problem_type = state["inferred_problem_type"]
            
            response = self.llm.invoke(hyperparameter_tuning_prompt, problem_type=problem_type)
            tuning_code = response.strip()
            
            # Clean code
            if "```python" in tuning_code:
                tuning_code = tuning_code.split("```python")[1].split("```")[0]
            elif "```" in tuning_code:
                tuning_code = tuning_code.split("```")[1].split("```")[0]
            
            state["tuning_code"] = tuning_code.strip()
            
            # Execute tuning code (each session has its own namespace)
            exec_globals = {
                'pd': pd, 'np': np, 'pickle': pickle,
                'GridSearchCV': GridSearchCV,
                'RandomizedSearchCV': RandomizedSearchCV,
                'train_test_split': train_test_split,
                'LinearRegression': LinearRegression,
                'LogisticRegression': LogisticRegression,
                'RandomForestClassifier': RandomForestClassifier,
                'RandomForestRegressor': RandomForestRegressor,
                'XGBRegressor': xgb.XGBRegressor,
                'XGBClassifier': xgb.XGBClassifier,
                'accuracy_score': accuracy_score,
                'mean_squared_error': mean_squared_error,
                'r2_score': r2_score,
                'LabelEncoder': LabelEncoder,
                'data_path': state["data_path"],
                'target_column': state["target_column"]
            }
            
            exec(state["tuning_code"], exec_globals)
            
            if 'tuned_results' in exec_globals:
                tuned_results = exec_globals['tuned_results']
                state["metrics"].update(tuned_results.get("metrics", {}))
                state["tuned_model_path"] = tuned_results.get("model_path", "tuned_model.pkl")
                state["messages"].append("✅ Hyperparameter tuning completed")
            
            logger.info(f"[{thread_name}] Tuning completed for session {session_id}")
            
        except Exception as e:
            error_msg = f"Hyperparameter tuning failed: {str(e)}"
            state["errors"].append(error_msg)
            logger.error(f"[{thread_name}] Session {session_id}: {error_msg}")
        
        return state
    
    def _should_tune(self, state: MLWorkflowState) -> Literal["tune", "end"]:
        """Decide whether to proceed with tuning (thread-safe)"""
        session_id = state["session_id"]
        thread_name = threading.current_thread().name
        logger.info(f"[{thread_name}] Checking if tuning requested for session {session_id}")
        return "tune" if state.get("tune_requested", False) else "end"