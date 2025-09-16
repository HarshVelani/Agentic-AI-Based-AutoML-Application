from langgraph.graph import StateGraph, END
from code.ml_config import MLConfig
from code.state import MLWorkflowState
from typing import Dict, Any, Optional
from code.pipeline_generator import PipelineGenerator
from code.tool_manager import ToolManager



class AgenticMLWorkflow:
    """Main Agentic ML Workflow class"""
    
    def __init__(self):
        
        self.config = MLConfig()
        self.pg = PipelineGenerator()
        # self.Tool_Manager = ToolManager()
        #  # Build the workflow graph
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
        
        # # Add edges
        # workflow.add_edge("schema_analyzer", "problem_classifier")
        # workflow.add_edge("problem_classifier", "pipeline_generator")
        # workflow.add_edge("pipeline_generator", "code_executor")
        # workflow.add_edge("code_executor", "evaluation_agent")
        # workflow.add_conditional_edges(
        #     "evaluation_agent",
        #     self.pg._should_tune,
        #     {
        #         "tune": "tuning_agent",
        #         "end": END
        #     }
        # )
        # workflow.add_edge("tuning_agent", END)


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