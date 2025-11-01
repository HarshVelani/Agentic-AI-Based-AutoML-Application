# from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks, Form, Request
# from fastapi.middleware.cors import CORSMiddleware
# from fastapi.responses import JSONResponse, FileResponse
# from contextlib import asynccontextmanager
# import uvicorn
# import os
# import json
# import logging
# from typing import Dict, Any, Optional
# from datetime import datetime, timedelta
# import aiofiles
# import uuid
# from pathlib import Path
# import pandas as pd
# from pydantic import BaseModel, Field, validator
# import traceback
# from slowapi import Limiter, _rate_limit_exceeded_handler
# from slowapi.util import get_remote_address
# from slowapi.errors import RateLimitExceeded
# from slowapi.middleware import SlowAPIMiddleware
# from enum import Enum

# # Import session management and workflow
# from src.session_manager import SessionManager, SessionStatus, SessionRecord
# from src.workflow_manager import AgenticMLWorkflow

# # Configure logging
# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
#     handlers=[
#         logging.FileHandler('logs/app.log'),
#         logging.StreamHandler()
#     ]
# )
# logger = logging.getLogger(__name__)

# # Rate limiting with optimized defaults
# limiter = Limiter(
#     key_func=get_remote_address, 
#     default_limits=["200/minute"],  # More permissive default for read operations
#     storage_uri="memory://"  # Better performance than default
# )

# # Global session manager
# session_manager: Optional[SessionManager] = None
# workflow_instances: Dict[str, AgenticMLWorkflow] = {}

# # Simple in-memory cache for session status
# session_status_cache: Dict[str, tuple[Any, datetime]] = {}
# CACHE_TTL_SECONDS = 2  # Cache for 2 seconds to handle rapid polling


# class ProblemType(str, Enum):
#     CLASSIFICATION = "classification"
#     REGRESSION = "regression"
#     AUTO = "auto"


# class TrainingRequest(BaseModel):
#     target_column: str = Field(..., description="Name of the target column")
#     problem_type: Optional[ProblemType] = Field(ProblemType.AUTO, description="Problem type")
#     tune_model: bool = Field(False, description="Whether to perform hyperparameter tuning")
#     user_comments: Optional[str] = Field(None, description="Optional description of the dataset/problem")
    
#     @validator('target_column')
#     def validate_target_column(cls, v):
#         if not v or not v.strip():
#             raise ValueError('Target column cannot be empty')
#         return v.strip()


# class TrainingResponse(BaseModel):
#     session_id: str
#     job_id: str
#     status: SessionStatus
#     message: str
#     created_at: datetime


# class SessionStatusResponse(BaseModel):
#     session_id: str
#     job_id: str
#     status: SessionStatus
#     progress: float
#     current_step: Optional[str]
#     filename: str
#     target_column: str
#     created_at: datetime
#     updated_at: datetime
#     completed_at: Optional[datetime]
#     inferred_problem_type: Optional[str]
#     best_model_name: Optional[str]
#     metrics: Optional[Dict[str, Any]]
#     error_message: Optional[str]


# def make_json_safe(obj):
#     """Convert numpy types to native Python types"""
#     import numpy as np
    
#     if isinstance(obj, (np.integer, np.floating)):
#         return obj.item()
#     if isinstance(obj, (np.ndarray, pd.Series)):
#         return obj.tolist()
#     if isinstance(obj, (pd.Int64Dtype, pd.StringDtype, pd.CategoricalDtype)):
#         return str(obj)
#     return str(obj)


# def get_cached_session_status(session_id: str) -> Optional[SessionStatusResponse]:
#     """Get session status from cache if available and not expired"""
#     if session_id in session_status_cache:
#         cached_response, cached_time = session_status_cache[session_id]
#         if (datetime.utcnow() - cached_time).total_seconds() < CACHE_TTL_SECONDS:
#             return cached_response
#     return None


# def cache_session_status(session_id: str, response: SessionStatusResponse):
#     """Cache session status response"""
#     session_status_cache[session_id] = (response, datetime.utcnow())
    
#     # Simple cache cleanup: remove old entries if cache gets too large
#     if len(session_status_cache) > 1000:
#         current_time = datetime.utcnow()
#         expired_keys = [
#             key for key, (_, cached_time) in session_status_cache.items()
#             if (current_time - cached_time).total_seconds() > CACHE_TTL_SECONDS * 10
#         ]
#         for key in expired_keys:
#             del session_status_cache[key]


# @asynccontextmanager
# async def lifespan(app: FastAPI):
#     """Handle application startup and shutdown"""
#     global session_manager
    
#     # Startup
#     logger.info("Starting FastAPI application...")
    
#     # Initialize session manager
#     session_manager = SessionManager(storage_dir="session_data")
#     logger.info("Session manager initialized")
    
#     # Create necessary directories
#     directories = [
#         "uploads", "model", "results", "generated_code",
#         "ai_summary", "workflow_info", "logs", "backups", "session_data"
#     ]
#     for directory in directories:
#         os.makedirs(directory, exist_ok=True)
    
#     logger.info("FastAPI application started successfully")
#     yield
    
#     # Shutdown
#     logger.info("Shutting down FastAPI application...")
    
#     # Save all sessions before shutdown
#     if session_manager:
#         session_manager._save_sessions()
    
#     # Clear cache
#     session_status_cache.clear()
    
#     logger.info("FastAPI application shut down")


# # Create FastAPI app
# app = FastAPI(
#     title="Agentic ML Workflow API with Session Management",
#     description="Production-grade API for automated machine learning workflows with session tracking",
#     version="2.0.0",
#     docs_url="/docs",
#     redoc_url="/redoc",
#     lifespan=lifespan
# )

# # Add middleware
# app.add_middleware(SlowAPIMiddleware)
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["GET", "POST", "PUT", "DELETE"],
#     allow_headers=["*"],
# )

# # Add rate limit handler
# app.state.limiter = limiter
# app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


# async def save_uploaded_file(file: UploadFile, session_id: str) -> str:
#     """Save uploaded file and return path"""
#     if not file.filename:
#         raise HTTPException(status_code=400, detail="No file provided")
    
#     # Validate file type
#     allowed_extensions = {'.csv', '.xlsx', '.xls'}
#     file_extension = Path(file.filename).suffix.lower()
#     if file_extension not in allowed_extensions:
#         raise HTTPException(
#             status_code=400, 
#             detail=f"Invalid file type. Allowed: {', '.join(allowed_extensions)}"
#         )
    
#     # Create filename with session ID
#     filename = f"{session_id}_{file.filename}"
#     file_path = f"uploads/{filename}"
    
#     try:
#         async with aiofiles.open(file_path, 'wb') as f:
#             content = await file.read()
#             await f.write(content)
        
#         # Validate file content
#         try:
#             if file_extension == '.csv':
#                 pd.read_csv(file_path, nrows=5)
#             else:
#                 pd.read_excel(file_path, nrows=5)
#         except Exception as e:
#             os.remove(file_path)
#             raise HTTPException(status_code=400, detail=f"Invalid file format: {str(e)}")
        
#         return file_path
#     except Exception as e:
#         if os.path.exists(file_path):
#             os.remove(file_path)
#         raise HTTPException(status_code=500, detail=f"Failed to save file: {str(e)}")


# async def run_ml_workflow(session_id: str):
#     """Background task to run ML workflow for a session"""
#     try:
#         # Get session record
#         session = session_manager.get_session(session_id)
#         if not session:
#             logger.error(f"Session {session_id} not found")
#             return
        
#         # Initialize workflow
#         workflow = AgenticMLWorkflow(session_manager)
#         workflow_instances[session_id] = workflow
        
#         # Run workflow
#         results = workflow.run_workflow(
#             session_id=session_id,
#             data_path=session.file_path,
#             target_column=session.target_column,
#             problem_type=session.problem_type if session.problem_type != "auto" else None,
#             tune_model=session.tune_model,
#             user_comments=session.user_comments
#         )
        
#         # Save workflow results
#         workflow_path = f"workflow_info/{session_id}_workflow_results.json"
#         async with aiofiles.open(workflow_path, 'w') as f:
#             await f.write(json.dumps(results, default=make_json_safe, indent=2))
        
#         # Update session with workflow path
#         session_manager.update_session(
#             session_id=session_id,
#             workflow_info_path=workflow_path
#         )
        
#         # Clear cache for this session since it's updated
#         if session_id in session_status_cache:
#             del session_status_cache[session_id]
        
#         logger.info(f"Workflow completed successfully for session {session_id}")
        
#     except Exception as e:
#         logger.error(f"Workflow failed for session {session_id}: {str(e)}")
#         logger.error(traceback.format_exc())
        
#         # Update session with error
#         session_manager.update_session(
#             session_id=session_id,
#             status=SessionStatus.FAILED,
#             error_message=str(e),
#             error_traceback=traceback.format_exc()
#         )
        
#         # Clear cache for this session
#         if session_id in session_status_cache:
#             del session_status_cache[session_id]
#     finally:
#         # Cleanup
#         if session_id in workflow_instances:
#             del workflow_instances[session_id]


# # API Endpoints

# @app.get("/", tags=["Health"])
# async def root():
#     """Health check endpoint"""
#     return {
#         "message": "Agentic ML Workflow API with Session Management",
#         "status": "healthy",
#         "timestamp": datetime.utcnow(),
#         "version": "2.0.0"
#     }


# @app.get("/health", tags=["Health"])
# async def health_check():
#     """Detailed health check"""
#     stats = session_manager.get_session_statistics() if session_manager else {}
#     return {
#         "status": "healthy",
#         "active_workflows": len(workflow_instances),
#         "session_statistics": stats,
#         "cache_size": len(session_status_cache),
#         "timestamp": datetime.utcnow()
#     }


# @app.post("/train", response_model=TrainingResponse, tags=["Machine Learning"])
# @limiter.limit("10/minute")  # Strict limit for resource-intensive operations
# async def train_model(
#     request: Request,
#     background_tasks: BackgroundTasks,
#     file: UploadFile = File(...),
#     target_column: str = Form(...),
#     problem_type: Optional[str] = Form(ProblemType.AUTO),
#     tune_model: bool = Form(False),
#     user_comments: Optional[str] = Form(None)
# ):
#     """
#     Train a machine learning model with session tracking
#     Creates a new session for each training request
#     """
#     try:
#         # Create training request
#         training_request = TrainingRequest(
#             target_column=target_column,
#             problem_type=problem_type,
#             tune_model=tune_model,
#             user_comments=user_comments
#         )
        
#         # Create new session
#         session = session_manager.create_session(
#             filename=file.filename,
#             file_path="",  # Will be updated after file save
#             target_column=training_request.target_column,
#             problem_type=training_request.problem_type.value,
#             tune_model=training_request.tune_model,
#             user_comments=training_request.user_comments
#         )
        
#         # Save uploaded file
#         file_path = await save_uploaded_file(file, session.session_id)
        
#         # Update session with file path
#         session_manager.update_session(
#             session_id=session.session_id,
#             file_path=file_path
#         )
        
#         # Start background task
#         background_tasks.add_task(run_ml_workflow, session.session_id)
        
#         logger.info(f"Started training session {session.session_id} for file {file.filename}")
        
#         return TrainingResponse(
#             session_id=session.session_id,
#             job_id=session.job_id,
#             status=session.status,
#             message="Training session created successfully",
#             created_at=session.created_at
#         )
        
#     except Exception as e:
#         logger.error(f"Failed to create training session: {str(e)}")
#         raise HTTPException(status_code=500, detail=f"Failed to create training session: {str(e)}")


# @app.get("/sessions/{session_id}", response_model=SessionStatusResponse, tags=["Session Management"])
# # Using default rate limit of 200/minute - suitable for polling
# async def get_session_status(
#     request: Request,
#     session_id: str
# ):
#     """
#     Get detailed status of a training session
#     Optimized with caching for frequent polling
#     """
#     # Check cache first
#     cached_response = get_cached_session_status(session_id)
#     if cached_response:
#         return cached_response
    
#     # Get fresh data
#     session = session_manager.get_session(session_id)
    
#     if not session:
#         raise HTTPException(status_code=404, detail="Session not found")
    
#     response = SessionStatusResponse(
#         session_id=session.session_id,
#         job_id=session.job_id,
#         status=session.status,
#         progress=session.progress,
#         current_step=session.current_step,
#         filename=session.filename,
#         target_column=session.target_column,
#         created_at=session.created_at,
#         updated_at=session.updated_at,
#         completed_at=session.completed_at,
#         inferred_problem_type=session.inferred_problem_type,
#         best_model_name=session.best_model_name,
#         metrics=session.metrics,
#         error_message=session.error_message
#     )
    
#     # Cache the response
#     cache_session_status(session_id, response)
    
#     return response


# @app.get("/sessions", tags=["Session Management"])
# @limiter.limit("50/minute")  # Moderate limit for list operations
# async def list_sessions(
#     request: Request,
#     limit: int = 50,
#     offset: int = 0,
#     status: Optional[SessionStatus] = None
# ):
#     """List all sessions with pagination and filtering"""
#     try:
#         sessions, total = session_manager.list_sessions(
#             status=status,
#             limit=limit,
#             offset=offset
#         )
        
#         return {
#             "sessions": [
#                 {
#                     "session_id": s.session_id,
#                     "job_id": s.job_id,
#                     "filename": s.filename,
#                     "status": s.status.value,
#                     "progress": s.progress,
#                     "created_at": s.created_at.isoformat(),
#                     "updated_at": s.updated_at.isoformat(),
#                     "completed_at": s.completed_at.isoformat() if s.completed_at else None,
#                     "inferred_problem_type": s.inferred_problem_type,
#                     "best_model_name": s.best_model_name
#                 }
#                 for s in sessions
#             ],
#             "total": total,
#             "limit": limit,
#             "offset": offset,
#             "has_next": offset + limit < total
#         }
        
#     except Exception as e:
#         logger.error(f"Failed to list sessions: {str(e)}")
#         raise HTTPException(status_code=500, detail="Failed to retrieve sessions")


# @app.delete("/sessions/{session_id}", tags=["Session Management"])
# @limiter.limit("20/minute")  # Moderate limit for delete operations
# async def delete_session(
#     request: Request,
#     session_id: str
# ):
#     """Delete a session and its artifacts"""
#     session = session_manager.get_session(session_id)
    
#     if not session:
#         raise HTTPException(status_code=404, detail="Session not found")
    
#     if session.status in [SessionStatus.RUNNING]:
#         raise HTTPException(status_code=400, detail="Cannot delete running session")
    
#     # Delete session
#     success = session_manager.delete_session(session_id)
    
#     if not success:
#         raise HTTPException(status_code=500, detail="Failed to delete session")
    
#     # Cancel workflow if active
#     if session_id in workflow_instances:
#         del workflow_instances[session_id]
    
#     # Clear cache
#     if session_id in session_status_cache:
#         del session_status_cache[session_id]
    
#     logger.info(f"Session {session_id} deleted")
    
#     return {"message": "Session deleted successfully", "session_id": session_id}


# @app.get("/sessions/{session_id}/code", tags=["Artifacts"])
# # Using default rate limit - file downloads are read operations
# async def download_code(session_id: str):
#     """Download generated code for a session"""
#     session = session_manager.get_session(session_id)
    
#     if not session:
#         raise HTTPException(status_code=404, detail="Session not found")
    
#     if session.status != SessionStatus.COMPLETED:
#         raise HTTPException(status_code=400, detail="Session not completed")
    
#     if not session.generated_code_path or not os.path.exists(session.generated_code_path):
#         raise HTTPException(status_code=404, detail="Code file not found")
    
#     return FileResponse(
#         session.generated_code_path,
#         media_type="text/x-python",
#         filename=f"ml_code_{session_id}.py"
#     )


# @app.get("/sessions/{session_id}/results", tags=["Artifacts"])
# async def download_results(session_id: str):
#     """Download results JSON for a session"""
#     session = session_manager.get_session(session_id)
    
#     if not session:
#         raise HTTPException(status_code=404, detail="Session not found")
    
#     if session.status != SessionStatus.COMPLETED:
#         raise HTTPException(status_code=400, detail="Session not completed")
    
#     if not session.results_path or not os.path.exists(session.results_path):
#         raise HTTPException(status_code=404, detail="Results file not found")
    
#     return FileResponse(
#         session.results_path,
#         media_type="application/json",
#         filename=f"ml_results_{session_id}.json"
#     )


# @app.get("/sessions/{session_id}/summary", tags=["Artifacts"])
# async def download_summary(session_id: str):
#     """Download AI-generated summary for a session"""
#     session = session_manager.get_session(session_id)
    
#     if not session:
#         raise HTTPException(status_code=404, detail="Session not found")
    
#     if session.status != SessionStatus.COMPLETED:
#         raise HTTPException(status_code=400, detail="Session not completed")
    
#     if not session.ai_summary_path or not os.path.exists(session.ai_summary_path):
#         raise HTTPException(status_code=404, detail="Summary file not found")
    
#     return FileResponse(
#         session.ai_summary_path,
#         media_type="text/markdown",
#         filename=f"ai_summary_{session_id}.md"
#     )


# @app.get("/sessions/{session_id}/model", tags=["Artifacts"])
# async def download_model(session_id: str):
#     """Download trained model for a session"""
#     session = session_manager.get_session(session_id)
    
#     if not session:
#         raise HTTPException(status_code=404, detail="Session not found")
    
#     if session.status != SessionStatus.COMPLETED:
#         raise HTTPException(status_code=400, detail="Session not completed")
    
#     if not session.model_path or not os.path.exists(session.model_path):
#         raise HTTPException(status_code=404, detail="Model file not found")
    
#     return FileResponse(
#         session.model_path,
#         media_type="application/zip",
#         filename=f"model_{session_id}.zip"
#     )


# @app.get("/sessions/export/excel", tags=["Export"])
# @limiter.limit("10/minute")  # Stricter limit for export operations
# async def export_sessions_excel(request: Request):
#     """Export all sessions to Excel file"""
#     try:
#         if not session_manager.sessions_file.exists():
#             raise HTTPException(status_code=404, detail="No sessions found")
        
#         return FileResponse(
#             session_manager.sessions_file,
#             media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
#             filename=f"sessions_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
#         )
#     except Exception as e:
#         logger.error(f"Failed to export sessions: {str(e)}")
#         raise HTTPException(status_code=500, detail="Failed to export sessions")


# @app.get("/statistics", tags=["Monitoring"])
# async def get_statistics():
#     """Get system statistics"""
#     try:
#         stats = session_manager.get_session_statistics()
        
#         return {
#             "session_statistics": stats,
#             "active_workflows": len(workflow_instances),
#             "cache_statistics": {
#                 "cached_sessions": len(session_status_cache),
#                 "cache_ttl_seconds": CACHE_TTL_SECONDS
#             },
#             "storage_info": {
#                 "sessions_file_size": os.path.getsize(session_manager.sessions_file) if session_manager.sessions_file.exists() else 0,
#                 "upload_directory_size": sum(
#                     os.path.getsize(os.path.join("uploads", f)) 
#                     for f in os.listdir("uploads") 
#                     if os.path.isfile(os.path.join("uploads", f))
#                 ) if os.path.exists("uploads") else 0,
#             },
#             "timestamp": datetime.utcnow()
#         }
#     except Exception as e:
#         logger.error(f"Failed to get statistics: {str(e)}")
#         raise HTTPException(status_code=500, detail="Failed to retrieve statistics")


# @app.post("/admin/cleanup", tags=["Admin"])
# @limiter.limit("5/hour")  # Very strict limit for admin operations
# async def cleanup_old_sessions(request: Request, days: int = 30):
#     """Cleanup sessions older than specified days"""
#     try:
#         removed = session_manager.cleanup_old_sessions(days=days)
        
#         # Clear entire cache after cleanup
#         session_status_cache.clear()
        
#         return {
#             "message": f"Cleaned up {removed} sessions older than {days} days",
#             "removed_count": removed
#         }
#     except Exception as e:
#         logger.error(f"Failed to cleanup sessions: {str(e)}")
#         raise HTTPException(status_code=500, detail="Failed to cleanup sessions")


# @app.post("/admin/clear-cache", tags=["Admin"])
# @limiter.limit("10/minute")
# async def clear_cache(request: Request):
#     """Clear the session status cache"""
#     cache_size = len(session_status_cache)
#     session_status_cache.clear()
#     return {
#         "message": f"Cache cleared successfully",
#         "cleared_entries": cache_size
#     }


# # Error handlers
# @app.exception_handler(HTTPException)
# async def http_exception_handler(request, exc):
#     logger.error(f"HTTP {exc.status_code} error on {request.url}: {exc.detail}")
#     return JSONResponse(
#         status_code=exc.status_code,
#         content={"error": exc.detail, "timestamp": datetime.utcnow().isoformat()}
#     )


# @app.exception_handler(Exception)
# async def general_exception_handler(request, exc):
#     logger.error(f"Unhandled error on {request.url}: {str(exc)}", exc_info=True)
#     return JSONResponse(
#         status_code=500,
#         content={
#             "error": "Internal server error",
#             "timestamp": datetime.utcnow().isoformat()
#         }
#     )


# # if __name__ == "__main__":
# #     uvicorn.run(
# #         "main:app",
# #         host="0.0.0.0",
# #         port=8000,
# #         reload=True,
# #         log_level="info"
# #     )