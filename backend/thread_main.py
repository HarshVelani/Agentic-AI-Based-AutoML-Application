"""
Thread-Safe FastAPI Main Application with Parallel Session Support
"""

from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks, Form, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from contextlib import asynccontextmanager
import uvicorn
import os
import json
import logging
from typing import Dict, Any, Optional
from datetime import datetime
import aiofiles
import uuid
from pathlib import Path
import pandas as pd
from pydantic import BaseModel, Field, validator
import traceback
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from enum import Enum
import threading
import concurrent.futures

# Import session management and workflow
from thread_src.session_manager import ThreadSafeSessionManager, SessionStatus, SessionRecord
from thread_src.workflow_manager import ThreadSafeAgenticMLWorkflow

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(threadName)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Rate limiting
limiter = Limiter(key_func=get_remote_address, default_limits=["30/minute"])

# Global session manager and thread pool
session_manager: Optional[ThreadSafeSessionManager] = None
thread_pool: Optional[concurrent.futures.ThreadPoolExecutor] = None
workflow_instances: Dict[str, ThreadSafeAgenticMLWorkflow] = {}


class ProblemType(str, Enum):
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    AUTO = "auto"


class TrainingRequest(BaseModel):
    target_column: str = Field(..., description="Name of the target column")
    problem_type: Optional[ProblemType] = Field(ProblemType.AUTO, description="Problem type")
    tune_model: bool = Field(False, description="Whether to perform hyperparameter tuning")
    user_comments: Optional[str] = Field(None, description="Optional description of the dataset/problem")
    
    @validator('target_column')
    def validate_target_column(cls, v):
        if not v or not v.strip():
            raise ValueError('Target column cannot be empty')
        return v.strip()


class TrainingResponse(BaseModel):
    session_id: str
    job_id: str
    status: SessionStatus
    message: str
    created_at: datetime
    thread_info: Optional[str] = None


class SessionStatusResponse(BaseModel):
    session_id: str
    job_id: str
    status: SessionStatus
    progress: float
    current_step: Optional[str]
    filename: str
    target_column: str
    created_at: datetime
    updated_at: datetime
    completed_at: Optional[datetime]
    inferred_problem_type: Optional[str]
    best_model_name: Optional[str]
    metrics: Optional[Dict[str, Any]]
    error_message: Optional[str]
    thread_id: Optional[str]
    thread_name: Optional[str]


def make_json_safe(obj):
    """Convert numpy types to native Python types"""
    import numpy as np
    
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    if isinstance(obj, (np.ndarray, pd.Series)):
        return obj.tolist()
    if isinstance(obj, (pd.Int64Dtype, pd.StringDtype, pd.CategoricalDtype)):
        return str(obj)
    return str(obj)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle application startup and shutdown"""
    global session_manager, thread_pool
    
    # Startup
    logger.info("Starting FastAPI application with parallel execution support...")
    
    # Initialize session manager
    session_manager = ThreadSafeSessionManager(storage_dir="session_data")
    logger.info("Thread-safe session manager initialized")
    
    # Initialize thread pool for parallel execution
    max_workers = int(os.getenv("MAX_PARALLEL_SESSIONS", "5"))
    thread_pool = concurrent.futures.ThreadPoolExecutor(
        max_workers=max_workers,
        thread_name_prefix="MLWorkflow"
    )
    logger.info(f"Thread pool initialized with {max_workers} workers")
    
    # Create necessary directories
    directories = [
        "uploads", "model", "results", "generated_code",
        "ai_summary", "workflow_info", "logs", "backups", "session_data"
    ]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    
    logger.info("FastAPI application started successfully")
    yield
    
    # Shutdown
    logger.info("Shutting down FastAPI application...")
    
    # Shutdown thread pool
    if thread_pool:
        logger.info("Waiting for active threads to complete...")
        thread_pool.shutdown(wait=True, cancel_futures=False)
        logger.info("Thread pool shut down")
    
    # Save all sessions before shutdown
    if session_manager:
        session_manager._save_sessions()
    
    logger.info("FastAPI application shut down")


# Create FastAPI app
app = FastAPI(
    title="Parallel Agentic ML Workflow API",
    description="Production-grade API for automated machine learning workflows with parallel execution",
    version="3.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)

# Add rate limit handler
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


async def save_uploaded_file(file: UploadFile, session_id: str) -> str:
    """Save uploaded file and return path (supports Excel)"""
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")
    
    # Validate file type (now supports Excel)
    allowed_extensions = {'.xlsx', '.xls', '.csv'}
    file_extension = Path(file.filename).suffix.lower()
    if file_extension not in allowed_extensions:
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid file type. Allowed: {', '.join(allowed_extensions)}"
        )
    
    # Create filename with session ID
    filename = f"{session_id}_{file.filename}"
    file_path = f"uploads/{filename}"
    
    try:
        async with aiofiles.open(file_path, 'wb') as f:
            content = await file.read()
            await f.write(content)
        
        # Validate file content
        try:
            if file_extension == '.csv':
                pd.read_csv(file_path, nrows=5)
            else:
                pd.read_excel(file_path, nrows=5, engine='openpyxl')
        except Exception as e:
            os.remove(file_path)
            raise HTTPException(status_code=400, detail=f"Invalid file format: {str(e)}")
        
        return file_path
    except Exception as e:
        if os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(status_code=500, detail=f"Failed to save file: {str(e)}")


def run_ml_workflow_sync(session_id: str):
    """
    Synchronous workflow runner for thread pool execution
    This runs in a separate thread
    """
    try:
        # Get session record
        session = session_manager.get_session(session_id)
        if not session:
            logger.error(f"Session {session_id} not found")
            return
        
        # Get current thread info
        current_thread = threading.current_thread()
        logger.info(f"Starting workflow for session {session_id} on thread {current_thread.name}")
        
        # Register thread
        session_manager.register_thread(session_id, current_thread)
        
        # Initialize workflow
        workflow = ThreadSafeAgenticMLWorkflow(session_manager)
        workflow_instances[session_id] = workflow
        
        # Run workflow
        results = workflow.run_workflow(
            session_id=session_id,
            data_path=session.file_path,
            target_column=session.target_column,
            problem_type=session.problem_type if session.problem_type != "auto" else None,
            tune_model=session.tune_model,
            user_comments=session.user_comments
        )
        
        # Save workflow results
        workflow_path = f"workflow_info/{session_id}_workflow_results.json"
        with open(workflow_path, 'w') as f:
            json.dump(results, f, default=make_json_safe, indent=2)
        
        # Update session with workflow path
        session_manager.update_session(
            session_id=session_id,
            workflow_info_path=workflow_path
        )
        
        logger.info(f"Workflow completed successfully for session {session_id} on thread {current_thread.name}")
        
    except Exception as e:
        logger.error(f"Workflow failed for session {session_id}: {str(e)}")
        logger.error(traceback.format_exc())
        
        # Update session with error
        session_manager.update_session(
            session_id=session_id,
            status=SessionStatus.FAILED,
            error_message=str(e),
            error_traceback=traceback.format_exc()
        )
    finally:
        # Cleanup
        if session_id in workflow_instances:
            del workflow_instances[session_id]
        
        # Unregister thread
        session_manager.unregister_thread(session_id)


async def run_ml_workflow(session_id: str):
    """
    Async wrapper to submit workflow to thread pool
    This allows parallel execution of multiple workflows
    """
    try:
        # Submit to thread pool for parallel execution
        future = thread_pool.submit(run_ml_workflow_sync, session_id)
        logger.info(f"Submitted session {session_id} to thread pool")
    except Exception as e:
        logger.error(f"Failed to submit workflow to thread pool: {str(e)}")
        session_manager.update_session(
            session_id=session_id,
            status=SessionStatus.FAILED,
            error_message=f"Failed to start workflow: {str(e)}"
        )


# API Endpoints

@app.get("/", tags=["Health"])
async def root():
    """Health check endpoint"""
    return {
        "message": "Parallel Agentic ML Workflow API",
        "status": "healthy",
        "timestamp": datetime.utcnow(),
        "version": "3.0.0",
        "features": ["parallel_execution", "thread_safe", "excel_support"]
    }


@app.get("/health", tags=["Health"])
async def health_check():
    """Detailed health check"""
    stats = session_manager.get_session_statistics() if session_manager else {}
    
    # Get thread pool status
    if thread_pool:
        thread_info = {
            "max_workers": thread_pool._max_workers,
            "active_threads": threading.active_count(),
            "thread_names": [t.name for t in threading.enumerate()]
        }
    else:
        thread_info = None
    
    return {
        "status": "healthy",
        "active_workflows": len(workflow_instances),
        "session_statistics": stats,
        "thread_pool_info": thread_info,
        "timestamp": datetime.utcnow()
    }


@app.post("/train", response_model=TrainingResponse, tags=["Machine Learning"])
@limiter.limit("10/minute")
async def train_model(
    request: Request,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    target_column: str = Form(...),
    problem_type: Optional[str] = Form(ProblemType.AUTO),
    tune_model: bool = Form(False),
    user_comments: Optional[str] = Form(None)
):
    """
    Train a machine learning model with parallel execution support
    Creates a new session for each training request
    Multiple requests can be processed simultaneously
    """
    try:
        # Create training request
        training_request = TrainingRequest(
            target_column=target_column,
            problem_type=problem_type,
            tune_model=tune_model,
            user_comments=user_comments
        )
        
        # Create new session
        session = session_manager.create_session(
            filename=file.filename,
            file_path="",  # Will be updated after file save
            target_column=training_request.target_column,
            problem_type=training_request.problem_type.value,
            tune_model=training_request.tune_model,
            user_comments=training_request.user_comments
        )
        
        # Save uploaded file
        file_path = await save_uploaded_file(file, session.session_id)
        
        # Update session with file path
        session_manager.update_session(
            session_id=session.session_id,
            file_path=file_path
        )
        
        # Start workflow in thread pool (parallel execution)
        background_tasks.add_task(run_ml_workflow, session.session_id)
        
        logger.info(f"Started training session {session.session_id} for file {file.filename}")
        
        return TrainingResponse(
            session_id=session.session_id,
            job_id=session.job_id,
            status=session.status,
            message="Training session created and submitted to parallel execution queue",
            created_at=session.created_at,
            thread_info=f"Will run on thread pool (max {thread_pool._max_workers} parallel sessions)"
        )
        
    except Exception as e:
        logger.error(f"Failed to create training session: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to create training session: {str(e)}")


@app.get("/sessions/{session_id}", response_model=SessionStatusResponse, tags=["Session Management"])
@limiter.limit("30/minute")
async def get_session_status(
    request: Request,
    session_id: str
):
    """Get detailed status of a training session"""
    session = session_manager.get_session(session_id)
    
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    return SessionStatusResponse(
        session_id=session.session_id,
        job_id=session.job_id,
        status=session.status,
        progress=session.progress,
        current_step=session.current_step,
        filename=session.filename,
        target_column=session.target_column,
        created_at=session.created_at,
        updated_at=session.updated_at,
        completed_at=session.completed_at,
        inferred_problem_type=session.inferred_problem_type,
        best_model_name=session.best_model_name,
        metrics=session.metrics,
        error_message=session.error_message,
        thread_id=session.thread_id,
        thread_name=session.thread_name
    )


@app.get("/sessions", tags=["Session Management"])
@limiter.limit("20/minute")
async def list_sessions(
    request: Request,
    limit: int = 50,
    offset: int = 0,
    status: Optional[SessionStatus] = None
):
    """List all sessions with pagination and filtering"""
    try:
        sessions, total = session_manager.list_sessions(
            status=status,
            limit=limit,
            offset=offset
        )
        
        return {
            "sessions": [
                {
                    "session_id": s.session_id,
                    "job_id": s.job_id,
                    "filename": s.filename,
                    "status": s.status.value,
                    "progress": s.progress,
                    "created_at": s.created_at.isoformat(),
                    "updated_at": s.updated_at.isoformat(),
                    "completed_at": s.completed_at.isoformat() if s.completed_at else None,
                    "inferred_problem_type": s.inferred_problem_type,
                    "best_model_name": s.best_model_name,
                    "thread_name": s.thread_name
                }
                for s in sessions
            ],
            "total": total,
            "limit": limit,
            "offset": offset,
            "has_next": offset + limit < total
        }
        
    except Exception as e:
        logger.error(f"Failed to list sessions: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to retrieve sessions")


@app.delete("/sessions/{session_id}", tags=["Session Management"])
@limiter.limit("10/minute")
async def delete_session(
    request: Request,
    session_id: str
):
    """Delete a session and its artifacts"""
    session = session_manager.get_session(session_id)
    
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    if session.status in [SessionStatus.RUNNING]:
        raise HTTPException(status_code=400, detail="Cannot delete running session. Cancel it first.")
    
    # Delete session
    success = session_manager.delete_session(session_id)
    
    if not success:
        raise HTTPException(status_code=500, detail="Failed to delete session")
    
    # Cancel workflow if active
    if session_id in workflow_instances:
        del workflow_instances[session_id]
    
    logger.info(f"Session {session_id} deleted")
    
    return {"message": "Session deleted successfully", "session_id": session_id}


@app.get("/sessions/{session_id}/code", tags=["Artifacts"])
async def download_code(session_id: str):
    """Download generated code for a session"""
    session = session_manager.get_session(session_id)
    
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    if session.status != SessionStatus.COMPLETED:
        raise HTTPException(status_code=400, detail="Session not completed")
    
    if not session.generated_code_path or not os.path.exists(session.generated_code_path):
        raise HTTPException(status_code=404, detail="Code file not found")
    
    return FileResponse(
        session.generated_code_path,
        media_type="text/x-python",
        filename=f"ml_code_{session_id}.py"
    )


@app.get("/sessions/{session_id}/results", tags=["Artifacts"])
async def download_results(session_id: str):
    """Download results JSON for a session"""
    session = session_manager.get_session(session_id)
    
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    if session.status != SessionStatus.COMPLETED:
        raise HTTPException(status_code=400, detail="Session not completed")
    
    if not session.results_path or not os.path.exists(session.results_path):
        raise HTTPException(status_code=404, detail="Results file not found")
    
    return FileResponse(
        session.results_path,
        media_type="application/json",
        filename=f"ml_results_{session_id}.json"
    )


@app.get("/sessions/{session_id}/summary", tags=["Artifacts"])
async def download_summary(session_id: str):
    """Download AI-generated summary for a session"""
    session = session_manager.get_session(session_id)
    
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    if session.status != SessionStatus.COMPLETED:
        raise HTTPException(status_code=400, detail="Session not completed")
    
    if not session.ai_summary_path or not os.path.exists(session.ai_summary_path):
        raise HTTPException(status_code=404, detail="Summary file not found")
    
    return FileResponse(
        session.ai_summary_path,
        media_type="text/markdown",
        filename=f"ai_summary_{session_id}.md"
    )


@app.get("/sessions/{session_id}/model", tags=["Artifacts"])
async def download_model(session_id: str):
    """Download trained model for a session"""
    session = session_manager.get_session(session_id)
    
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    if session.status != SessionStatus.COMPLETED:
        raise HTTPException(status_code=400, detail="Session not completed")
    
    if not session.model_path or not os.path.exists(session.model_path):
        raise HTTPException(status_code=404, detail="Model file not found")
    
    return FileResponse(
        session.model_path,
        media_type="application/zip",
        filename=f"model_{session_id}.zip"
    )


@app.get("/sessions/export/excel", tags=["Export"])
async def export_sessions_excel():
    """Export all sessions to Excel file"""
    try:
        if not session_manager.sessions_file.exists():
            raise HTTPException(status_code=404, detail="No sessions found")
        
        return FileResponse(
            session_manager.sessions_file,
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            filename=f"sessions_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
        )
    except Exception as e:
        logger.error(f"Failed to export sessions: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to export sessions")


@app.get("/statistics", tags=["Monitoring"])
async def get_statistics():
    """Get system statistics including thread pool status"""
    try:
        stats = session_manager.get_session_statistics()
        
        # Thread pool statistics
        thread_stats = {
            "max_workers": thread_pool._max_workers if thread_pool else 0,
            "active_count": threading.active_count(),
            "active_threads": [
                {"name": t.name, "daemon": t.daemon, "alive": t.is_alive()}
                for t in threading.enumerate()
            ]
        }
        
        return {
            "session_statistics": stats,
            "active_workflows": len(workflow_instances),
            "thread_statistics": thread_stats,
            "storage_info": {
                "sessions_file_size": os.path.getsize(session_manager.sessions_file) if session_manager.sessions_file.exists() else 0,
                "upload_directory_size": sum(
                    os.path.getsize(os.path.join("uploads", f)) 
                    for f in os.listdir("uploads") 
                    if os.path.isfile(os.path.join("uploads", f))
                ) if os.path.exists("uploads") else 0,
            },
            "timestamp": datetime.utcnow()
        }
    except Exception as e:
        logger.error(f"Failed to get statistics: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to retrieve statistics")


@app.post("/admin/cleanup", tags=["Admin"])
async def cleanup_old_sessions(days: int = 30):
    """Cleanup sessions older than specified days"""
    try:
        removed = session_manager.cleanup_old_sessions(days=days)
        return {
            "message": f"Cleaned up {removed} sessions older than {days} days",
            "removed_count": removed
        }
    except Exception as e:
        logger.error(f"Failed to cleanup sessions: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to cleanup sessions")


# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    logger.error(f"HTTP {exc.status_code} error on {request.url}: {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail, "timestamp": datetime.utcnow().isoformat()}
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Unhandled error on {request.url}: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "timestamp": datetime.utcnow().isoformat()
        }
    )


# if __name__ == "__main__":
#     uvicorn.run(
#         "thread_main:app",
#         host="0.0.0.0",
#         port=8000,
#         reload=False,
#         access_log=True,
#         workers=1  # Important: Keep workers=1 for thread pool management
#     )