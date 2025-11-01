"""
Thread-Safe Workflow Manager with Parallel Execution Support
"""

from thread_src.state import MLWorkflowState
from thread_src.pipeline_generator import PipelineGenerator
from thread_src.session_manager import ThreadSafeSessionManager, SessionStatus

from langgraph.graph import StateGraph, END
from typing import Dict, Any, Optional
import logging
import threading

logger = logging.getLogger(__name__)


class ThreadSafeAgenticMLWorkflow:
    """Thread-safe Agentic ML Workflow class for parallel execution"""
    
    # Class-level lock for workflow graph compilation
    _graph_lock = threading.Lock()
    _compiled_workflow = None
    
    def __init__(self, session_manager: ThreadSafeSessionManager):
        """
        Initialize workflow with session manager
        
        Args:
            session_manager: ThreadSafeSessionManager instance
        """
        self.session_manager = session_manager
        self.pg = PipelineGenerator(session_manager)
        
        # Thread-local storage for workflow state
        self._thread_local = threading.local()
        
        # Use shared compiled workflow or compile new one
        if ThreadSafeAgenticMLWorkflow._compiled_workflow is None:
            with ThreadSafeAgenticMLWorkflow._graph_lock:
                if ThreadSafeAgenticMLWorkflow._compiled_workflow is None:
                    ThreadSafeAgenticMLWorkflow._compiled_workflow = self._build_workflow()
        
        self.workflow = ThreadSafeAgenticMLWorkflow._compiled_workflow
    
    def _build_workflow(self) -> StateGraph:
        """Build the LangGraph workflow (thread-safe)"""
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
    
    def run_workflow(
        self,
        session_id: str,
        data_path: str,
        target_column: str, 
        problem_type: Optional[str] = None, 
        tune_model: bool = False,
        user_comments: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Run the complete ML workflow with session tracking (thread-safe)
        
        Args:
            session_id: Session ID
            data_path: Path to data file
            target_column: Target column name
            problem_type: Problem type (optional)
            tune_model: Whether to tune model
            user_comments: User comments
            
        Returns:
            Dictionary with workflow results
        """
        # Get current thread info
        current_thread = threading.current_thread()
        thread_name = current_thread.name
        thread_id = current_thread.ident
        
        logger.info(f"Starting workflow for session {session_id} on thread {thread_name} (ID: {thread_id})")
        
        try:
            # Update session status to running
            self.session_manager.update_session(
                session_id=session_id,
                status=SessionStatus.RUNNING,
                progress=0.0,
                current_step="Initializing workflow",
                thread_id=str(thread_id),
                thread_name=thread_name
            )
            
            # Initialize state
            initial_state = MLWorkflowState(
                session_id=session_id,
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
            
            # Run workflow (thread-safe as each session has its own state)
            logger.info(f"Invoking workflow for session {session_id}")
            final_state = self.workflow.invoke(initial_state)
            
            logger.info(f"Workflow completed for session {session_id} on thread {thread_name}")
            
            # Update session with final results
            self.session_manager.update_session(
                session_id=session_id,
                status=SessionStatus.COMPLETED if not final_state["errors"] else SessionStatus.FAILED,
                progress=1.0,
                current_step="Workflow completed",
                inferred_problem_type=final_state["inferred_problem_type"],
                best_model_name=final_state["best_model_name"],
                metrics=final_state.get("metrics", {}),
                generated_code_path=final_state["pipeline_code_path"],
                results_path=final_state["results_path"],
                model_path=final_state.get("tuned_model_path") or final_state["model_path"],
                ai_summary_path=final_state["summarized_result_path"],
                error_message="; ".join(final_state["errors"]) if final_state["errors"] else None
            )
            
            # Prepare results
            results = {
                "session_id": session_id,
                "thread_name": thread_name,
                "thread_id": thread_id,
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
            
            logger.info(f"Session {session_id} completed successfully on thread {thread_name}")
            return results
            
        except Exception as e:
            logger.error(f"Workflow failed for session {session_id} on thread {thread_name}: {str(e)}")
            
            # Update session with error
            self.session_manager.update_session(
                session_id=session_id,
                status=SessionStatus.FAILED,
                error_message=str(e),
                current_step="Workflow failed"
            )
            
            raise