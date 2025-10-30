from langgraph.graph import StateGraph, END
from src.ml_config import MLConfig
from src.state import MLWorkflowState
from typing import Dict, Any, Optional
from src.pipeline_generator import PipelineGenerator



class AgenticMLWorkflow:
    """Main Agentic ML Workflow class"""
    
    def __init__(self):
        
        self.pg = PipelineGenerator()
        # Build the workflow graph
        self.workflow = self._build_workflow()
    
    def _build_workflow(self) -> StateGraph:
        """Build the LangGraph workflow"""
        workflow = StateGraph(MLWorkflowState)
        
        # Add nodes
        workflow.add_node("schema_analyzer", self.pg._schema_analyzer_agent)
        workflow.add_node("problem_classifier", self.pg._problem_classifier_agent)
        workflow.add_node("pipeline_generator", self.pg._pipeline_generator_agent)
        workflow.add_node("code_executor", self.pg._code_executor_node)
        workflow.add_node("best_model", self.pg._best_model)
        workflow.add_node("result_summarizer", self.pg._result_summarizer)
        workflow.add_node("tuning_agent", self.pg._tuning_agent)
        
        # Add edges
        workflow.add_edge("schema_analyzer", "problem_classifier")
        workflow.add_edge("problem_classifier", "pipeline_generator")
        workflow.add_edge("pipeline_generator", "code_executor")
        workflow.add_edge("code_executor", "best_model")
        workflow.add_edge("best_model", "result_summarizer")
        workflow.add_conditional_edges(
            "result_summarizer",
            self.pg._should_tune,
            {
                "tune": "tuning_agent",
                "end": END
            }
        )
        workflow.add_edge("tuning_agent", END)
        
        # Set entry point
        workflow.set_entry_point("schema_analyzer")
        
        return workflow.compile()
    

    def run_workflow(self, data_path: str, target_column: str, 
                    problem_type: Optional[str] = None, 
                    tune_model: bool = False,
                    user_comments: Optional[str] = None,
                    job_id: Optional[str] = "1") -> Dict[str, Any]:
        """Run the complete ML workflow"""
        
        # Initialize state
        initial_state = MLWorkflowState(
            session_id=job_id,
            data_path=data_path,
            target_column=target_column,
            problem_type=problem_type,
            user_comments=user_comments or "",
            data_schema={},
            inferred_problem_type="",
            pipeline_code="",
            pipeline_code_path="",
            execution_results={},
            model_path="",
            results_path="",
            best_model_name="",
            summarized_result="",
            summarized_result_path="",
            metrics={},
            tune_requested=tune_model,
            tuning_code="",
            tuned_model_path="",
            messages=[],
            errors=[]
        )
        
        # Run workflow
        final_state = self.workflow.invoke(initial_state)
        

        print(f"\n\n <<<<< Final State: \n{final_state} >>>>>")
        # Prepare results
        results = {
            "problem_type": final_state["inferred_problem_type"],
            "metrics": final_state["execution_results"],
            "model_path": final_state.get("tuned_model_path") or final_state["model_path"],
            "results_path": final_state["results_path"],
            "pipeline_code_path": final_state["pipeline_code_path"],
            "summarized_result_path": final_state["summarized_result_path"],
            "best_model_name": final_state["best_model_name"],
            "summarized_result": final_state["summarized_result"],
            "messages": final_state["messages"],
            "errors": final_state["errors"],
            "data_schema": final_state["data_schema"]
        }
        

        print(f"\n <<<<< Final State: {results} >>>>>")
        return results
    


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


from src.state import MLWorkflowState
from src.llm_manager import LLMManager
from src.prompts import (
    problem_classifier_prompt,
    pipeline_generator_prompt,
    hyperparameter_tuning_prompt,
    result_summarizer_prompt
)

import pandas as pd
import numpy as np
import os
import time

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
from langchain_core.prompts import ChatPromptTemplate


from typing import Literal

class PipelineGenerator:
    
    def __init__(self):
        self.MLWorkflowState = MLWorkflowState()
        self.llm = LLMManager()


    def _schema_analyzer_agent(self, state: MLWorkflowState) -> MLWorkflowState:
        """Analyze dataset schema and characteristics"""
        try:

            print(f"\n\n\n <<<<< Schema Analyzer Invoked >>>>>")
            # Load data
            if state["data_path"].endswith('.csv'):
                df = pd.read_csv(state["data_path"])
            elif state["data_path"].endswith('.xlsx'):
                df = pd.read_excel(state["data_path"])
            else:
                raise ValueError("Unsupported file format. Use CSV or Excel.")
            
            # Analyze schema
            schema_info = {
                "shape": df.shape,
                "columns": list(df.columns),
                "dtypes": df.dtypes.to_dict(),
                "missing_values": df.isnull().sum().to_dict(),
                "numeric_columns": list(df.select_dtypes(include=[np.number]).columns),
                "categorical_columns": list(df.select_dtypes(include=['object', 'category']).columns),
                "target_column_info": {
                    "dtype": str(df[state["target_column"]].dtype),
                    "unique_values": len(df[state["target_column"]].unique()),
                    "sample_values": df[state["target_column"]].head().tolist()
                }
            }
            
            state["data_schema"] = schema_info
            state["messages"].append(f"✅ Dataset analyzed: {df.shape[0]} rows, {df.shape[1]} columns")
            
            print(f"\n <<<<< Dataset schema: {schema_info} >>>>>")

        except Exception as e:
            state["errors"].append(f"Schema analysis failed: {str(e)}")
        
        return state
    
    def _problem_classifier_agent(self, state: MLWorkflowState) -> MLWorkflowState:
        """Classify problem type (Regression vs Classification)"""
        try:

            print(f"\n\n\n <<<<< Problem Classifier Invoked >>>>>")

            if state.get("problem_type"):
                # User provided problem type
                state["inferred_problem_type"] = state["problem_type"]
                state["messages"].append(f"✅ Using user-specified problem type: {state['problem_type']}")
            else:
                # Infer problem type using LLM
                target_info = state["data_schema"]["target_column_info"]
                
                response = self.llm.invoke(problem_classifier_prompt, target_info=target_info)
                problem_type = response.strip().lower()

                print(f"\n\n\n <<<<< LLM Problem Type Response: {problem_type} >>>>>")
                
                if "regression" in problem_type:
                    state["inferred_problem_type"] = "regression"
                else:
                    state["inferred_problem_type"] = "classification"
                
                state["messages"].append(f"✅ Inferred problem type: {state['inferred_problem_type']}")
        
        except Exception as e:
            state["errors"].append(f"Problem classification failed: {str(e)}")
            # Default fallback
            state["inferred_problem_type"] = "Classification"
        
        return state
    
    def _pipeline_generator_agent(self, state: MLWorkflowState) -> MLWorkflowState:
        """Generate ML pipeline code using LLM"""
        try:

            print(f"\n\n\n <<<<< Pipeline Generator Invoked >>>>>")

            schema = state["data_schema"]
            problem_type = state["inferred_problem_type"]
            target_col = state["target_column"]
            data_path = state["data_path"]
            # results_path = state["results_path"]
            # model_path = state["model_path"]
            user_comments = state["user_comments"]
            session_id = state["session_id"]
            
            response = self.llm.invoke(pipeline_generator_prompt, 
                                       schema=schema, 
                                       problem_type=problem_type, 
                                       target_col=target_col, 
                                       data_path=data_path, 
                                    #    results_path=results_path, 
                                    #    model_path=model_path,
                                       user_comments=user_comments,
                                       session_id=session_id)
            
            state["pipeline_code"] = response.strip()
            
            # Clean the code (remove markdown formatting if present)
            code = state["pipeline_code"]
            if "```python" in code:
                code = code.split("```python")[1].split("```")[0]
            elif "```" in code:
                code = code.split("```")[1].split("```")[0]
            

            print(f"\n <<<<< Generated Pipeline Code: \n{state['pipeline_code']} >>>>>")

            path = f'generated_code/{session_id}_code.py'
            with open(path, 'w', encoding='utf-8') as f:
                # codes = visual_code.split('\n')
                # code = "\n".join(codes[1:-1])
                f.write(code)
                print("\n====Cleaned Code====")

            time.sleep(2)
            # Update state with cleaned code and model_path and results_path
            state["model_path"] = f'model/{session_id}_{problem_type}_models.zip'
            state["results_path"] = f'results/{session_id}_{problem_type}_results.json'
            state["pipeline_code_path"] = path
            state["pipeline_code"] = code.strip()
            state["messages"].append("✅ ML pipeline code generated")

            # backend/generated_code/2025_09_01__10_30_41__bdc3fc8a-5147-4843-a7e0-8ef5ac8b0025_code.py
            
        except Exception as e:
            state["errors"].append(f"Pipeline generation failed: {str(e)}")
        
        return state
    

    
    def _code_executor_node(self, state: MLWorkflowState) -> MLWorkflowState:
        """Execute the generated ML pipeline code"""
        # try:

        print(f"\n\n\n <<<<< Code Executor Invoked >>>>>")

        session_id = state["session_id"]

        os.system(f"python generated_code/{session_id}_code.py")

        filepath = state["results_path"]
        with open(filepath, 'r') as f:
            import json
            results = json.load(f)
            state["execution_results"] = results
            state["messages"].append("✅ Pipeline executed successfully")
        
        print(f"\n <<<<< Execution Results: {results} >>>>>")

        return state

    def _best_model(self, state: MLWorkflowState) -> MLWorkflowState:
        
        """Identify the best model from results"""
        
        try:
            results = state["execution_results"]
            problem_type = state["inferred_problem_type"]

            if problem_type == "regression":
                # For regression, use RMSE or R²
                best_model_name = min(results, key=lambda x: results[x]["rmse"] or results[x]["RMSE"])
            else:
                best_model_name = max(results, key=lambda x: results[x]["accuracy"])
            
            state["best_model_name"] = best_model_name
            state["messages"].append(f"✅ Best model identified: {best_model_name}")

            # print(f"\n\n <<<<< Best Model: {state["best_model_name"]} >>>>>")

        except Exception as e:
            state["errors"].append(f"Best model identification failed: {str(e)}")

        return state


    def _result_summarizer(self, state: MLWorkflowState) -> MLWorkflowState:
        """Summarize key results from the workflow Using LLM"""
        try:

            print(f"\n\n\n <<<<< Result Summarizer Invoked >>>>>")

            data_schema = state["data_schema"]
            results = state["execution_results"]
            best_model = state["best_model_name"]
            
            response = self.llm.invoke(result_summarizer_prompt, 
                                       data_schema=data_schema,
                                        results=results,
                                        best_model=best_model
                                       )
            

            print(f"\n\n <<<<< Generated Result Summary: \n{response} >>>>>")
            state["summarized_result"] = response.strip()

            # save summary to .md file
            path = f'ai_summary/{state["session_id"]}_summary.md'

            with open(path, 'w', encoding='utf-8') as f:
                f.write(response)

            state["summarized_result_path"] = path
            # print(f"\n <<<<< Summary saved to: {state["summarized_result_path"]} >>>>>")

            state["messages"].append("✅ Results summarized")
        
        except Exception as e:
            state["errors"].append(f"Result summarization failed: {str(e)}")

        return state
    
    def _tuning_agent(self, state: MLWorkflowState) -> MLWorkflowState:
        """Perform hyperparameter tuning if requested"""
        try:

            print(f"\n\n\n <<<<< Tuning Agent Invoked >>>>>")

            # Generate tuning code using LLM
            problem_type = state["inferred_problem_type"]
            
            response = self.llm.invoke(hyperparameter_tuning_prompt, problem_type=problem_type)
            tuning_code = response.strip()
            
            # Clean code
            if "```python" in tuning_code:
                tuning_code = tuning_code.split("```python")[1].split("```")[0]
            elif "```" in tuning_code:
                tuning_code = tuning_code.split("```")[1].split("```")[0]
            
            state["tuning_code"] = tuning_code.strip()
            
            # Execute tuning code
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
            
        except Exception as e:
            state["errors"].append(f"Hyperparameter tuning failed: {str(e)}")
        
        return state
    
    def _should_tune(self, state: MLWorkflowState) -> Literal["tune", "end"]:
        """Decide whether to proceed with tuning"""

        print(f"\n\n\n <<<<< Should Tune Invoked >>>>>")

        return "tune" if state.get("tune_requested", False) else "end"
    

from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os
load_dotenv()  # Load environment variables from .env file

from langchain_core.prompts import ChatPromptTemplate


class LLMManager:
    """Manager for LLM interactions"""
        
    def __init__(self):
        self.llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0, api_key=os.getenv("GROQ_API_KEY"))

    def invoke(self, prompt: ChatPromptTemplate, **kwargs) -> str:
        messages = prompt.format_messages(**kwargs)
        response = self.llm.invoke(messages)
        return response.content
        