from code.state import MLWorkflowState
from code.llm_manager import LLMManager
from code.prompts import (
    problem_classifier_prompt,
    pipeline_generator_prompt,
    hyperparameter_tuning_prompt,
    result_summarizer_prompt
)

import pandas as pd
import numpy as np
import os

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

            path = f'backend/generated_code/{session_id}_code.py'
            with open(path, 'w') as f:
                # codes = visual_code.split('\n')
                # code = "\n".join(codes[1:-1])
                print("\n====Cleaned Code====")
                f.write(code)

            # Update state with cleaned code and model_path and results_path
            state["model_path"] = f'backend/model/{session_id}_{problem_type}_models.zip'
            state["results_path"] = f'backend/results/{session_id}_{problem_type}_results.json'
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

        os.system(f"python backend/generated_code/{session_id}_code.py")

        problem_type = state["inferred_problem_type"].lower()
        # filepath = f"results/{problem_type}/{session_id}_{problem_type}_results.json"

        filepath = state["results_path"]
        with open(filepath, 'r') as f:
            import json
            results = json.load(f)
            state["execution_results"] = results
            state["messages"].append("✅ Pipeline executed successfully")
        
        print(f"\n <<<<< Execution Results: {results} >>>>>")

        return state


        #     # Prepare execution environment
        #     exec_globals = {
        #         'pd': pd,
        #         'np': np,
        #         'pickle': pickle,
        #         'train_test_split': train_test_split,
        #         'LinearRegression': LinearRegression,
        #         'LogisticRegression': LogisticRegression,
        #         'RandomForestClassifier': RandomForestClassifier,
        #         'RandomForestRegressor': RandomForestRegressor,
        #         'XGBRegressor': xgb.XGBRegressor,
        #         'XGBClassifier': xgb.XGBClassifier,
        #         'accuracy_score': accuracy_score,
        #         'precision_score': precision_score,
        #         'recall_score': recall_score,
        #         'f1_score': f1_score,
        #         'confusion_matrix': confusion_matrix,
        #         'mean_squared_error': mean_squared_error,
        #         'mean_absolute_error': mean_absolute_error,
        #         'r2_score': r2_score,
        #         'LabelEncoder': LabelEncoder,
        #         'StandardScaler': StandardScaler,
        #         'data_path': state["data_path"],
        #         'target_column': state["target_column"]
        #     }
            
        #     # Execute the code
        #     exec(state["pipeline_code"], exec_globals)
            
        #     # Extract results
        #     if 'results' in exec_globals:
        #         state["execution_results"] = exec_globals['results']
        #         state["messages"].append("✅ Pipeline executed successfully")
        #     else:
        #         state["errors"].append("Code execution completed but no results found")
            
        # except Exception as e:
        #     state["errors"].append(f"Code execution failed: {str(e)}")
        
        # return state





        # try:
        #     # Prepare execution environment
        #     exec_globals = {
        #         'pd': pd,
        #         'np': np,
        #         'pickle': pickle,
        #         'joblib': pickle,  # Alternative for model saving
        #         'train_test_split': train_test_split,
        #         'LinearRegression': LinearRegression,
        #         'LogisticRegression': LogisticRegression,
        #         'RandomForestClassifier': RandomForestClassifier,
        #         'RandomForestRegressor': RandomForestRegressor,
        #         'XGBRegressor': xgb.XGBRegressor,
        #         'XGBClassifier': xgb.XGBClassifier,
        #         'accuracy_score': accuracy_score,
        #         'precision_score': precision_score,
        #         'recall_score': recall_score,
        #         'f1_score': f1_score,
        #         'confusion_matrix': confusion_matrix,
        #         'mean_squared_error': mean_squared_error,
        #         'mean_absolute_error': mean_absolute_error,
        #         'r2_score': r2_score,
        #         'LabelEncoder': LabelEncoder,
        #         'StandardScaler': StandardScaler,
        #         'data_path': state["data_path"],
        #         'target_column': state["target_column"],
        #         '__builtins__': __builtins__
        #     }
            
        #     # Create a local scope for execution
        #     exec_locals = {}
            
        #     # Execute the code with both global and local scopes
        #     exec(state["pipeline_code"], exec_globals, exec_locals)
            
        #     # Try to extract results from multiple possible locations
        #     results = None
            
        #     # Check local scope first
        #     if 'results' in exec_locals:
        #         results = exec_locals['results']
        #     # Check global scope
        #     elif 'results' in exec_globals:
        #         results = exec_globals['results']
        #     # Check for other common variable names
        #     elif 'final_results' in exec_locals:
        #         results = exec_locals['final_results']
        #     elif 'output' in exec_locals:
        #         results = exec_locals['output']
        #     elif 'model_results' in exec_locals:
        #         results = exec_locals['model_results']
            
        #     if results:
        #         state["execution_results"] = results
        #         state["messages"].append("✅ Pipeline executed successfully")
                
        #         # Debug: Print what we found
        #         state["messages"].append(f"📊 Results keys: {list(results.keys()) if isinstance(results, dict) else 'Not a dict'}")
        #     else:
        #         # If no results variable found, try to construct results from available variables
        #         constructed_results = {}
                
        #         # Look for common result patterns
        #         for var_name, var_value in exec_locals.items():
        #             if 'metric' in var_name.lower() or 'score' in var_name.lower():
        #                 constructed_results[var_name] = var_value
        #             elif var_name in ['best_model', 'model', 'trained_model']:
        #                 constructed_results['model'] = var_value
        #             elif var_name in ['model_path', 'saved_model_path']:
        #                 constructed_results['model_path'] = var_value
                
        #         if constructed_results:

        #             print(f"\n <<<<< Constructed Results: {constructed_results} >>>>>")
        #             state["execution_results"] = {'metrics': constructed_results, 'model_path': 'best_model.pkl'}
        #             state["messages"].append("✅ Pipeline executed, results constructed from variables")
        #         else:
        #             state["errors"].append("Code execution completed but no results found. Available variables: " + 
        #                                  str(list(exec_locals.keys())[:10]))  # Show first 10 variables for debugging
            
        # except SyntaxError as e:
        #     state["errors"].append(f"Code syntax error: {str(e)}")
        #     state["messages"].append("❌ Generated code has syntax errors")
        # except ImportError as e:
        #     state["errors"].append(f"Import error: {str(e)}")
        #     state["messages"].append("❌ Missing required libraries")
        # except FileNotFoundError as e:
        #     state["errors"].append(f"File not found: {str(e)}")
        #     state["messages"].append("❌ Data file not found")
        # except Exception as e:
        #     state["errors"].append(f"Code execution failed: {str(e)}")
        #     state["messages"].append(f"❌ Execution error: {type(e).__name__}")
        
        # return state


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

            print(f"\n\n <<<<< Best Model: {state["best_model_name"]} >>>>>")

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
            # problem_type = state["inferred_problem_type"]
            # target_col = state["target_column"]
            # data_path = state["data_path"]
            # results_path = state["results_path"]
            # model_path = state["model_path"]
            # user_comments = state["user_comments"]
            # session_id = state["session_id"]
            
            response = self.llm.invoke(result_summarizer_prompt, 
                                       data_schema=data_schema,
                                        results=results,
                                        best_model=best_model
                                       )
            

            print(f"\n\n <<<<< Generated Result Summary: \n{response} >>>>>")
            state["summarized_result"] = response.strip()

            # save summary to .md file
            path = f'backend/ai_summary/{state["session_id"]}_summary.md'
            with open(path, 'w') as f:
                f.write(response)

            state["summarized_result_path"] = path
            print(f"\n <<<<< Summary saved to: {state["summarized_result_path"]} >>>>>")

            state["messages"].append("✅ Results summarized")
        
        except Exception as e:
            state["errors"].append(f"Result summarization failed: {str(e)}")

        return state

    # def report_generator(self, state: MLWorkflowState) -> MLWorkflowState:
    #     """Generate a summary report of the workflow"""
    #     try:

    #         print(f"\n\n\n <<<<< Report Generator Invoked >>>>>")

    #         report_lines = [
    #             "### Workflow Summary Report",
    #             f"- **Data Path:** {state['data_path']}",
    #             f"- **Target Column:** {state['target_column']}",
    #             f"- **Inferred Problem Type:** {state.get('inferred_problem_type', 'N/A')}",
    #             f"- **Best Model:** {state.get('best_model_name', 'N/A')}",
    #             "- **Metrics:**"
    #         ]
            
    #         metrics = state.get("metrics", {})
    #         for metric, value in metrics.items():
    #             report_lines.append(f"  - {metric}: {value}")
            
    #         if state["errors"]:
    #             report_lines.append("\n- **Errors Encountered:**")
    #             for error in state["errors"]:
    #                 report_lines.append(f"  - {error}")
    #         else:
    #             report_lines.append("\n- No errors encountered.")
            
    #         state["report"] = "\n".join(report_lines)
    #         state["messages"].append("✅ Summary report generated")

    #         print(f"\n <<<<< Generated Report: \n{state['report']} >>>>>")

    #     except Exception as e:
    #         state["errors"].append(f"Report generation failed: {str(e)}")
        
    #     return state






    # def _evaluation_agent(self, state: MLWorkflowState) -> MLWorkflowState:
    #     """Evaluate model performance and prepare results"""
    #     # try:

    #     print(f"\n\n\n <<<<< Evaluation Agent Invoked >>>>>")

    #     if state.get("execution_results"):

    #         print(f"\n <<<<< Execution Results: {state['execution_results']} >>>>>")

    #         results = state["execution_results"]
    #         state["metrics"] = results.get("metrics", {})
    #         state["model_path"] = results.get("model_path", "trained_model/best_model.pkl")
            
    #         # Format metrics for display
    #         if state["inferred_problem_type"] == "Regression":
    #             metrics_str = f"RMSE: {state['metrics'].get('rmse', 'N/A')}, MAE: {state['metrics'].get('mae', 'N/A')}, R²: {state['metrics'].get('r2', 'N/A')}"
    #         else:
    #             metrics_str = f"Accuracy: {state['metrics'].get('accuracy', 'N/A')}, F1: {state['metrics'].get('f1', 'N/A')}"
            
    #         state["messages"].append(f"✅ Model evaluation completed - {metrics_str}")
    #     else:
    #         state["errors"].append("No execution results to evaluate")
        
    #     except Exception as e:
    #         state["errors"].append(f"Evaluation failed: {str(e)}")
        
    #     return state
    
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