from fastapi import FastAPI, HTTPException, Depends, UploadFile, File, BackgroundTasks, Security, Form, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse, FileResponse
from contextlib import asynccontextmanager
import uvicorn
import os
import json
import logging
from typing import Dict, Any, Optional, List
import asyncio
from datetime import datetime, timedelta
import aiofiles
import uuid
from pathlib import Path
import pandas as pd
import numpy as np
from pydantic import BaseModel, Field, validator
import traceback
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
import redis.asyncio as redis
from enum import Enum
import time


# Import your existing workflow
from code.workflow_manager import AgenticMLWorkflow

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('backend/logs/app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Rate limiting
limiter = Limiter(key_func=get_remote_address, default_limits=["30  /minute"])

# Security
security = HTTPBearer()

# Redis for caching and session management
redis_client = None

class ProblemType(str, Enum):
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    AUTO = "auto"

class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"

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

class BatchTrainingRequest(BaseModel):
    jobs: List[TrainingRequest] = Field(..., description="List of training jobs")
    
    @validator('jobs')
    def validate_jobs(cls, v):
        if len(v) == 0:
            raise ValueError('At least one job is required')
        if len(v) > 10:  # Limit batch size
            raise ValueError('Maximum 10 jobs allowed per batch')
        return v

class TrainingResponse(BaseModel):
    job_id: str
    status: JobStatus
    message: str
    submitted_at: datetime

class JobStatusResponse(BaseModel):
    job_id: str
    status: JobStatus
    progress: Optional[float] = None
    submitted_at: datetime
    completed_at: Optional[datetime] = None
    results: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

# class ModelPredictionRequest(BaseModel):
#     model_id: str = Field(..., description="ID of the trained model")
#     data: Dict[str, Any] = Field(..., description="Input data for prediction")

# Global variables for job management
jobs_db = {}
workflow_instances = {}



def make_json_safe(obj):
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
    # Startup
    global redis_client
    try:
        redis_client = redis.from_url(
            os.getenv("REDIS_URL", "redis://localhost:6379"),
            decode_responses=True
        )
        await redis_client.ping()
        logger.info("Connected to Redis successfully")
    except Exception as e:
        logger.warning(f"Failed to connect to Redis: {e}. Using in-memory storage.")
        redis_client = None
    
    # Create necessary directories
    os.makedirs("backend/uploads", exist_ok=True)
    os.makedirs("backend/model", exist_ok=True)
    os.makedirs("backend/results", exist_ok=True)
    os.makedirs("backend/generated_code", exist_ok=True)
    os.makedirs("backend/ai_summary", exist_ok=True)
    os.makedirs("backend/workflow_info", exist_ok=True)
    os.makedirs("backend/logs", exist_ok=True)
    os.makedirs("backend/backups", exist_ok=True)
    
    logger.info("FastAPI application started successfully")
    yield
    
    # Shutdown
    if redis_client:
        await redis_client.close()
    logger.info("FastAPI application shut down")

# Create FastAPI app
app = FastAPI(
    title="Agentic ML Workflow API",
    description="Production-grade API for automated machine learning workflows",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Add middleware
app.add_middleware(SlowAPIMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],# os.getenv("ALLOWED_ORIGINS", "http://localhost:3000").split(","),
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)
# app.add_middleware(
#     TrustedHostMiddleware, 
#     allowed_hosts=os.getenv("ALLOWED_HOSTS", "localhost,127.0.0.1").split(",")
# )

# Add rate limit handler
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# # Security functions
# async def verify_token(credentials: HTTPAuthorizationCredentials = Security(security)):
#     """Verify API token"""
#     token = credentials.credentials
#     valid_tokens = os.getenv("API_TOKENS", "").split(",")
    
#     if not valid_tokens or token not in valid_tokens:
#         raise HTTPException(
#             status_code=401,
#             detail="Invalid authentication token",
#             headers={"WWW-Authenticate": "Bearer"},
#         )
#     return token

async def save_uploaded_file(file: UploadFile) -> str:
    """Save uploaded file and return path"""
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")
    
    # Validate file type
    allowed_extensions = {'.csv', '.xlsx', '.xls'}
    file_extension = Path(file.filename).suffix.lower()
    if file_extension not in allowed_extensions:
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid file type. Allowed: {', '.join(allowed_extensions)}"
        )
    
    # Create unique filename
    file_id = str(uuid.uuid4())
    current_time = datetime.now().strftime("%Y_%m_%d__%H_%M_%S")
    filename = f"{current_time}__{file_id}_{file.filename}"
    file_path = f"backend/uploads/{filename}"
    
    try:
        async with aiofiles.open(file_path, 'wb') as f:
            content = await file.read()
            await f.write(content)
        
        # Validate file content
        try:
            if file_extension == '.csv':
                pd.read_csv(file_path, nrows=5)
            else:
                pd.read_excel(file_path, nrows=5)
        except Exception as e:
            os.remove(file_path)
            raise HTTPException(status_code=400, detail=f"Invalid file format: {str(e)}")
        
        return file_path
    except Exception as e:
        if os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(status_code=500, detail=f"Failed to save file: {str(e)}")

async def save_job_status(job_id: str, status: JobStatus, **kwargs):
    """Save job status to Redis or memory"""
    job_data = {
        "job_id": job_id,
        "status": status.value,
        "updated_at": datetime.utcnow().isoformat(),
        **kwargs
    }
    
    # Convert any non-string values to strings for Redis
    for key, value in job_data.items():
        if isinstance(value, dict):
            job_data[key] = json.dumps(value)
        elif not isinstance(value, (str, int, float, bool)):
            job_data[key] = str(value)
    
    if redis_client:
        try:
            await redis_client.hset(f"job:{job_id}", mapping=job_data)
            await redis_client.expire(f"job:{job_id}", 86400)  # 24 hours TTL
        except Exception as e:
            logger.warning(f"Failed to save to Redis: {e}, falling back to memory")
            jobs_db[job_id] = job_data
    else:
        jobs_db[job_id] = job_data



    # if redis_client:
    #     await redis_client.hset(f"job:{job_id}", mapping=job_data)
    #     await redis_client.expire(f"job:{job_id}", 86400)  # 24 hours TTL
    # else:
    #     jobs_db[job_id] = job_data

async def get_job_status(job_id: str) -> Optional[Dict]:
    """Get job status from Redis or memory"""
    # if redis_client:
    #     job_data = await redis_client.hgetall(f"job:{job_id}")
    #     return job_data if job_data else None
    # else:
    #     return jobs_db.get(job_id)
    if redis_client:
        try:
            job_data = await redis_client.hgetall(f"job:{job_id}")
            if job_data:
                # Convert back JSON strings to objects where needed
                for key in ['results', 'request']:
                    if key in job_data and isinstance(job_data[key], str):
                        try:
                            job_data[key] = json.loads(job_data[key])
                        except (json.JSONDecodeError, TypeError):
                            pass
                return job_data
        except Exception as e:
            logger.warning(f"Failed to get from Redis: {e}, trying memory")
    
    return jobs_db.get(job_id)

async def run_ml_workflow(job_id: str, data_path: str, request: TrainingRequest):
    """Background task to run ML workflow"""
    try:
        # Update status to running
        await save_job_status(job_id, JobStatus.RUNNING, progress=0.1)
        
        # Initialize workflow
        workflow = AgenticMLWorkflow()
        workflow_instances[job_id] = workflow
        
        await save_job_status(job_id, JobStatus.RUNNING, progress=0.2)
        
        # Run workflow
        results = workflow.run_workflow(
            data_path=data_path,
            target_column=request.target_column,
            problem_type=request.problem_type.value if request.problem_type != ProblemType.AUTO else None,
            tune_model=request.tune_model,
            user_comments=request.user_comments,
            job_id=job_id
        )
        
        await save_job_status(job_id, JobStatus.RUNNING, progress=0.9)
        
        # Save results
        workflow_path = f"backend/workflow_info/{job_id}_workflow_results.json"
        # async with aiofiles.open(workflow_path, 'w') as f:
        #     # await f.write(json.dumps(results, default=str, indent=2))
        #     await f.write(json.dumps(results, default=make_json_safe, indent=2))
        async with aiofiles.open(workflow_path, 'w') as f:
            await f.write(json.dumps(results, default=make_json_safe, indent=2))
        
        # Update status to completed
        # await save_job_status(
        #     job_id, 
        #     JobStatus.COMPLETED,
        #     progress=1.0,
        #     completed_at=datetime.utcnow().isoformat(),
        #     results=results,
        #     workflow_path=workflow_path
        # )
        
        safe_results = json.loads(json.dumps(results, default=make_json_safe))
        await save_job_status(
            job_id, 
            JobStatus.COMPLETED,
            progress=1.0,
            completed_at=datetime.utcnow().isoformat(),
            results=safe_results,
            workflow_path=workflow_path
        )

        logger.info(f"Job {job_id} completed successfully")
        
    except Exception as e:
        logger.error(f"Job {job_id} failed: {str(e)}")
        await save_job_status(
            job_id,
            JobStatus.FAILED,
            error=str(e),
            traceback=traceback.format_exc(),
            completed_at=datetime.utcnow().isoformat()
        )
    finally:
        # Cleanup
        if job_id in workflow_instances:
            del workflow_instances[job_id]

# API Endpoints

@app.get("/", tags=["Health"])
async def root():
    """Health check endpoint"""
    return {
        "message": "Agentic ML Workflow API",
        "status": "healthy",
        "timestamp": datetime.utcnow(),
        "version": "1.0.0"
    }

@app.get("/health", tags=["Health"])
async def health_check():
    """Detailed health check"""
    redis_status = "connected" if redis_client else "disconnected"
    return {
        "status": "healthy",
        "redis": redis_status,
        "active_jobs": len(workflow_instances),
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
    # token: str = Depends(verify_token)
):
    """Train a machine learning model"""
    try:
        # Validate and save file
        file_path = await save_uploaded_file(file)
        
        # Create training request
        training_request = TrainingRequest(
            target_column=target_column,
            problem_type=problem_type,
            tune_model=tune_model,
            user_comments=user_comments
        )
        
        # Generate job ID
        current_time = datetime.now().strftime("%Y_%m_%d__%H_%M_%S")
        id = str(uuid.uuid4())
        job_id = current_time + "__" + id
        
        # Save initial job status
        await save_job_status(
            job_id,
            JobStatus.PENDING,
            submitted_at=datetime.utcnow().isoformat(),
            file_path=file_path,
            request=training_request.dict()
        )
        
        # Start background task
        background_tasks.add_task(run_ml_workflow, job_id, file_path, training_request)
        
        logger.info(f"Started training job {job_id} for file {file.filename}")
        
        return TrainingResponse(
            job_id=job_id,
            status=JobStatus.PENDING,
            message="Training job submitted successfully",
            submitted_at=datetime.utcnow()
        )
        
    except Exception as e:
        logger.error(f"Failed to submit training job: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to submit training job: {str(e)}")

@app.get("/jobs/{job_id}", response_model=JobStatusResponse, tags=["Job Management"])
@limiter.limit("30/minute")
async def get_job_status_endpoint(
    request: Request,
    job_id: str
    # token: str = Depends(verify_token)
):
    """Get status of a training job"""
    job_data = await get_job_status(job_id)
    
    if not job_data:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return JobStatusResponse(
        job_id=job_id,
        status=JobStatus(job_data.get("status", "unknown")),
        progress=job_data.get("progress"),
        submitted_at=datetime.fromisoformat(job_data.get("submitted_at", datetime.utcnow().isoformat())),
        completed_at=datetime.fromisoformat(job_data["completed_at"]) if job_data.get("completed_at") else None,
        results=json.loads(job_data["results"]) if job_data.get("results") and isinstance(job_data["results"], str) else job_data.get("results"),
        error=job_data.get("error")
    )

@app.get("/jobs", tags=["Job Management"])
@limiter.limit("20/minute")
async def list_jobs(
    request: Request,
    limit: int = 50,
    offset: int = 0,
    status: Optional[JobStatus] = None
    # token: str = Depends(verify_token)
):
    """List all jobs with pagination and filtering"""
    try:
        if redis_client:
            # Get job keys from Redis
            keys = await redis_client.keys("job:*")
            all_jobs = []
            for key in keys:
                job_data = await redis_client.hgetall(key)
                if job_data and (not status or job_data.get("status") == status.value):
                    all_jobs.append(job_data)
        else:
            # Get jobs from memory
            all_jobs = [job for job in jobs_db.values() 
                       if not status or job.get("status") == status.value]
        
        # Sort by submission time (most recent first)
        all_jobs.sort(key=lambda x: x.get("submitted_at", ""), reverse=True)
        
        # Apply pagination
        total = len(all_jobs)
        jobs = all_jobs[offset:offset + limit]
        
        return {
            "jobs": jobs,
            "total": total,
            "limit": limit,
            "offset": offset,
            "has_next": offset + limit < total
        }
        
    except Exception as e:
        logger.error(f"Failed to list jobs: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to retrieve jobs")

@app.delete("/jobs/{job_id}", tags=["Job Management"])
@limiter.limit("10/minute")
async def cancel_job(
    request: Request,
    job_id: str
    # token: str = Depends(verify_token)
):
    """Cancel a running job"""
    job_data = await get_job_status(job_id)
    
    if not job_data:
        raise HTTPException(status_code=404, detail="Job not found")
    
    if job_data.get("status") in [JobStatus.COMPLETED.value, JobStatus.FAILED.value]:
        raise HTTPException(status_code=400, detail="Cannot cancel completed or failed job")
    
    # Cancel the job
    if job_id in workflow_instances:
        del workflow_instances[job_id]
    
    await save_job_status(
        job_id,
        JobStatus.FAILED,
        error="Job cancelled by user",
        completed_at=datetime.utcnow().isoformat()
    )
    
    logger.info(f"Job {job_id} cancelled by user")
    
    return {"message": "Job cancelled successfully", "job_id": job_id}



@app.get("/jobs/{job_id}/code", tags=["Code"])
async def download_code(
    job_id: str
    # token: str = Depends(verify_token)
):
    """Download job Code as .py file"""
    job_data = await get_job_status(job_id)
    
    if not job_data:
        raise HTTPException(status_code=404, detail="Job not found")
    
    if job_data.get("status") != JobStatus.COMPLETED.value:
        raise HTTPException(status_code=400, detail="Job not completed")
    
    code_path = job_data.get("generated_code_path", f"backend/generated_code/{job_id}_code.py")
    
    if not os.path.exists(code_path):
        raise HTTPException(status_code=404, detail="Code file not found")
    
    return FileResponse(
        code_path,
        media_type="text/x-python",
        filename=f"ml_code_{job_id}.py"
    )

@app.get("/jobs/{job_id}/results", tags=["Results"])
async def download_results(
    job_id: str
    # token: str = Depends(verify_token)
):
    """Download job results as JSON file"""
    job_data = await get_job_status(job_id)
    
    if not job_data:
        raise HTTPException(status_code=404, detail="Job not found")
    
    if job_data.get("status") != JobStatus.COMPLETED.value:
        raise HTTPException(status_code=400, detail="Job not completed")
    

    print(f"\n\n <<<<< Job Data: {job_data} >>>>>")

    problem_type = job_data["results"]["problem_type"]
    # if problem_type == "classification":
    #     results_path = job_data.get("results_path", f"results/{job_id}_classification_results.json")
    # else:
    results_path = job_data.get("results_path", f"backend/results/{job_id}_{problem_type}_results.json")

    print(f"\n\n <<<<< Results Path: {results_path} >>>>>")

    if not os.path.exists(results_path):
        raise HTTPException(status_code=404, detail="Results file not found")
    
    return FileResponse(
        results_path,
        media_type="application/json",
        filename=f"ml_results_{job_id}.json"
    )



# import json
# import os
# from fastapi import HTTPException
# from fastapi.responses import FileResponse

# @app.get("/jobs/{job_id}/results", tags=["Results"])
# async def download_results(job_id: str):
#     """Download filtered job results as JSON file"""
#     job_data = await get_job_status(job_id)
    
#     if not job_data:
#         raise HTTPException(status_code=404, detail="Job not found")
    
#     if job_data.get("status") != JobStatus.COMPLETED.value:
#         raise HTTPException(status_code=400, detail="Job not completed")
    
#     results_path = job_data.get("results_path", f"results/{job_id}_results.json")

#     if not os.path.exists(results_path):
#         raise HTTPException(status_code=404, detail="Results file not found")
    
#     # Load the full results file
#     with open(results_path, "r") as f:
#         results = json.load(f)
    
#     # Keep only the required fields
#     filtered_results = {
#         "problem_type": results.get("problem_type"),
#         "metrics": results.get("metrics"),
#         "data_schema": results.get("data_schema"),
#     }
    
#     # Overwrite or create a filtered file
#     filtered_path = f"results/filtered/{job_id}_filtered_results.json"
#     with open(filtered_path, "w") as f:
#         json.dump(filtered_results, f, indent=4)
    
#     return FileResponse(
#         filtered_path,
#         media_type="application/json",
#         filename=f"ml_results_{job_id}.json"
#     )



@app.get("/jobs/{job_id}/ai_summary", tags=["AI Summary"])
async def download_ai_summary(job_id: str):
    """Download AI Summarized Results as Markdown file"""
    job_data = await get_job_status(job_id)
    
    if not job_data:
        raise HTTPException(status_code=404, detail="Job not found")
    
    if job_data.get("status") != JobStatus.COMPLETED.value:
        raise HTTPException(status_code=400, detail="Job not completed")

    summary_path = job_data.get("summarized_result_path")
    print(f"\n\n <<<<< Job Data Summary Path: {summary_path} >>>>>")

    summary_path = job_data.get("summarized_result_path", f'backend/ai_summary/{job_id}_summary.md')

    print(f"\n\n <<<<< Summary Path: {summary_path} >>>>>")

    if not summary_path or not os.path.exists(summary_path):
        raise HTTPException(status_code=404, detail="Summarized result file not found")
    
    # Inline View
    return FileResponse(
        summary_path,
        media_type="text/markdown",
        filename=f"ai_summary_{job_id}.md"
    )

    # # Force download
    # return FileResponse(
    #     summary_path,
    #     media_type="application/octet-stream",  # makes browser download instead of open
    #     filename=f"ai_summary_{job_id}.md"
    # )




@app.get("/jobs/{job_id}/model", tags=["Models"])
async def download_model(
    job_id: str
    # token: str = Depends(verify_token)
):
    """Download trained model file"""
    job_data = await get_job_status(job_id)
    
    if not job_data:
        raise HTTPException(status_code=404, detail="Job not found")
    
    if job_data.get("status") != JobStatus.COMPLETED.value:
        raise HTTPException(status_code=400, detail="Job not completed")
    
    results = job_data.get("results")
    if not results:
        raise HTTPException(status_code=404, detail="Job results not found")
    
    if isinstance(results, str):
        results = json.loads(results)
    
    model_path = results.get("model_path")
    if not model_path or not os.path.exists(model_path):
        raise HTTPException(status_code=404, detail="Model file not found")
    
    # return FileResponse(
    #     model_path,
    #     media_type="application/octet-stream",
    #     filename=f"model_{job_id}."
    # )
    return FileResponse(
    model_path,
    media_type="application/zip",
    filename=f"model_{job_id}.zip"
)




@app.post("/batch-train", tags=["Machine Learning"])
@limiter.limit("5/hour")
async def batch_train(
    request: Request,
    background_tasks: BackgroundTasks,
    batch_request: BatchTrainingRequest,
    files: List[UploadFile] = File(...)
    # token: str = Depends(verify_token)
):
    """Submit multiple training jobs"""
    if len(files) != len(batch_request.jobs):
        raise HTTPException(
            status_code=400, 
            detail="Number of files must match number of jobs"
        )
    
    job_ids = []
    
    for i, (file, job_request) in enumerate(zip(files, batch_request.jobs)):
        try:
            # Save file
            file_path = await save_uploaded_file(file)
            
            # Generate job ID
            job_id = str(uuid.uuid4())
            job_ids.append(job_id)
            
            # Save job status
            await save_job_status(
                job_id,
                JobStatus.PENDING,
                submitted_at=datetime.utcnow().isoformat(),
                file_path=file_path,
                request=job_request.dict(),
                batch_index=i
            )
            
            # Start background task
            background_tasks.add_task(run_ml_workflow, job_id, file_path, job_request)
            
        except Exception as e:
            logger.error(f"Failed to submit batch job {i}: {str(e)}")
            # Clean up already submitted jobs could be added here
            raise HTTPException(
                status_code=500, 
                detail=f"Failed to submit batch job {i}: {str(e)}"
            )
    
    logger.info(f"Submitted batch training with {len(job_ids)} jobs")
    
    return {
        "message": f"Batch training submitted successfully with {len(job_ids)} jobs",
        "job_ids": job_ids,
        "submitted_at": datetime.utcnow()
    }

@app.get("/metrics", tags=["Monitoring"])
async def get_system_metrics():  # token: str = Depends(verify_token)
    """Get system metrics and statistics"""
    try:
        # Job statistics
        if redis_client:
            keys = await redis_client.keys("job:*")
            all_jobs = []
            for key in keys:
                job_data = await redis_client.hgetall(key)
                if job_data:
                    all_jobs.append(job_data)
        else:
            all_jobs = list(jobs_db.values())
        
        status_counts = {}
        for status in JobStatus:
            status_counts[status.value] = sum(1 for job in all_jobs if job.get("status") == status.value)
        
        # System info
        return {
            "total_jobs": len(all_jobs),
            "active_jobs": len(workflow_instances),
            "job_status_distribution": status_counts,
            "system_info": {
                "redis_connected": redis_client is not None,
                "upload_directory_size": sum(
                    os.path.getsize(os.path.join("uploads", f)) 
                    for f in os.listdir("uploads") 
                    if os.path.isfile(os.path.join("uploads", f))
                ) if os.path.exists("uploads") else 0,
                "models_count": len([f for f in os.listdir("models") if f.endswith('.pkl')]) if os.path.exists("models") else 0
            },
            "timestamp": datetime.utcnow()
        }
        
    except Exception as e:
        logger.error(f"Failed to get metrics: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to retrieve metrics")

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

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        access_log=True,
        workers=1  # Use 1 worker for development, increase for production
    )