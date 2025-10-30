# Session Management System for Agentic ML Workflow

## Overview

The updated system now includes comprehensive session management that tracks each ML training as a separate session. Every session is uniquely identified and all related data, artifacts, and state are managed independently.

## Key Features

### 1. **Session-Based Architecture**
- Each CSV upload creates a new session
- Unique session ID generated for every training
- Complete isolation between different training sessions
- Persistent storage of session metadata

### 2. **State Management**
- Real-time progress tracking
- Current step visibility
- Error tracking and reporting
- Session status lifecycle (Created → Running → Completed/Failed)

### 3. **Artifact Management**
Each session maintains references to:
- Uploaded data file
- Generated Python code
- Training results (JSON)
- Trained model files (ZIP)
- AI-generated summary (Markdown)
- Workflow information

### 4. **Persistent Storage**
- Sessions stored in Excel file (`session_data/sessions.xlsx`)
- JSON backup for redundancy
- In-memory cache for active sessions
- Automatic persistence on updates

## Project Structure

```
.
├── src/
│   ├── session_manager.py          # Core session management
│   ├── workflow_manager.py          # Updated workflow with session support
│   ├── pipeline_generator.py       # Pipeline generator with tracking
│   ├── session_utils.py            # Utility functions
│   └── ...
├── main.py                          # FastAPI application
├── session_data/
│   ├── sessions.xlsx               # Main session storage
│   └── sessions_backup.json        # JSON backup
├── uploads/                         # Uploaded files (named with session ID)
├── generated_code/                  # Generated Python code
├── results/                         # Training results
├── model/                           # Trained models
├── ai_summary/                      # AI summaries
├── workflow_info/                   # Workflow results
└── exports/                         # Exported reports
```

## API Endpoints

### Training and Session Management

#### Create New Training Session
```http
POST /train
Content-Type: multipart/form-data

Parameters:
  - file: CSV/Excel file
  - target_column: Target column name
  - problem_type: "classification" | "regression" | "auto"
  - tune_model: boolean
  - user_comments: Optional description

Response:
{
  "session_id": "2025_01_15__10_30_45__abc123",
  "job_id": "2025_01_15__10_30_45__abc123",
  "status": "created",
  "message": "Training session created successfully",
  "created_at": "2025-01-15T10:30:45"
}
```

#### Get Session Status
```http
GET /sessions/{session_id}

Response:
{
  "session_id": "2025_01_15__10_30_45__abc123",
  "job_id": "2025_01_15__10_30_45__abc123",
  "status": "running",
  "progress": 0.65,
  "current_step": "Executing ML pipeline",
  "filename": "iris.csv",
  "target_column": "species",
  "created_at": "2025-01-15T10:30:45",
  "updated_at": "2025-01-15T10:32:10",
  "completed_at": null,
  "inferred_problem_type": "classification",
  "best_model_name": "RandomForest",
  "metrics": {...},
  "error_message": null
}
```

#### List Sessions
```http
GET /sessions?limit=50&offset=0&status=completed

Response:
{
  "sessions": [...],
  "total": 150,
  "limit": 50,
  "offset": 0,
  "has_next": true
}
```

#### Delete Session
```http
DELETE /sessions/{session_id}

Response:
{
  "message": "Session deleted successfully",
  "session_id": "2025_01_15__10_30_45__abc123"
}
```

### Artifact Download

#### Download Generated Code
```http
GET /sessions/{session_id}/code
Response: Python file (.py)
```

#### Download Results
```http
GET /sessions/{session_id}/results
Response: JSON file
```

#### Download AI Summary
```http
GET /sessions/{session_id}/summary
Response: Markdown file (.md)
```

#### Download Model
```http
GET /sessions/{session_id}/model
Response: ZIP file containing trained models
```

### Export and Analytics

#### Export Sessions to Excel
```http
GET /sessions/export/excel
Response: Excel file with multiple sheets
  - Sessions: Main session data
  - Artifacts: File paths
  - Errors: Error logs
  - Metrics: Model metrics
  - Statistics: Summary statistics
```

#### Get System Statistics
```http
GET /statistics

Response:
{
  "session_statistics": {
    "total_sessions": 150,
    "status_distribution": {...},
    "successful_sessions": 120,
    "failed_sessions": 5,
    "active_sessions": 2
  },
  "active_workflows": 2,
  "storage_info": {...}
}
```

#### Cleanup Old Sessions
```http
POST /admin/cleanup?days=30

Response:
{
  "message": "Cleaned up 25 sessions older than 30 days",
  "removed_count": 25
}
```

## SessionManager Class

### Core Methods

```python
from src.session_manager import SessionManager, SessionStatus

# Initialize
session_manager = SessionManager(storage_dir="session_data")

# Create new session
session = session_manager.create_session(
    filename="data.csv",
    file_path="/path/to/file.csv",
    target_column="target",
    problem_type="classification",
    tune_model=False,
    user_comments="Optional description"
)

# Update session
session_manager.update_session(
    session_id=session.session_id,
    status=SessionStatus.RUNNING,
    progress=0.5,
    current_step="Training models"
)

# Get session
session = session_manager.get_session(session_id)

# List sessions
sessions, total = session_manager.list_sessions(
    status=SessionStatus.COMPLETED,
    limit=50,
    offset=0
)

# Delete session
session_manager.delete_session(session_id)

# Get statistics
stats = session_manager.get_session_statistics()

# Cleanup old sessions
removed = session_manager.cleanup_old_sessions(days=30)
```

## Session Record Structure

```python
@dataclass
class SessionRecord:
    session_id: str                      # Unique identifier
    job_id: str                          # Job identifier
    filename: str                        # Original filename
    file_path: str                       # Path to uploaded file
    target_column: str                   # Target column
    problem_type: Optional[str]          # Problem type
    tune_model: bool                     # Tuning flag
    user_comments: Optional[str]         # User comments
    status: SessionStatus                # Current status
    created_at: datetime                 # Creation time
    updated_at: datetime                 # Last update
    completed_at: Optional[datetime]     # Completion time
    
    # Artifact paths
    generated_code_path: Optional[str]
    results_path: Optional[str]
    model_path: Optional[str]
    ai_summary_path: Optional[str]
    workflow_info_path: Optional[str]
    
    # Results
    inferred_problem_type: Optional[str]
    best_model_name: Optional[str]
    metrics: Optional[Dict[str, Any]]
    
    # Error tracking
    error_message: Optional[str]
    error_traceback: Optional[str]
    
    # Progress
    progress: float                      # 0.0 to 1.0
    current_step: Optional[str]
```

## Session Status Lifecycle

```
CREATED → RUNNING → COMPLETED
                  ↘ FAILED
                  ↘ CANCELLED
```

### Status Definitions

- **CREATED**: Session initialized, waiting to start
- **RUNNING**: Workflow actively processing
- **COMPLETED**: Successfully finished
- **FAILED**: Encountered error during processing
- **CANCELLED**: Manually cancelled by user

## Progress Tracking

The system tracks progress through workflow stages:

| Progress | Stage                          |
|----------|--------------------------------|
| 0.0      | Session created                |
| 0.1      | Analyzing dataset schema       |
| 0.2      | Schema analysis completed      |
| 0.3      | Classifying problem type       |
| 0.35     | Problem type identified        |
| 0.4      | Generating ML pipeline code    |
| 0.5      | Pipeline code generated        |
| 0.6      | Executing ML pipeline          |
| 0.7      | Pipeline execution completed   |
| 0.75     | Identifying best model         |
| 0.85     | Summarizing results            |
| 0.9      | Results summarization completed|
| 0.95     | Performing hyperparameter tuning (if enabled) |
| 1.0      | Workflow completed             |

## Utility Functions

### SessionExporter

```python
from src.session_utils import SessionExporter

# Export to Excel with multiple sheets
path = SessionExporter.export_to_excel(
    sessions_file=Path("session_data/sessions.xlsx"),
    include_metrics=True
)

# Export to CSV files
paths = SessionExporter.export_to_csv(
    sessions_file=Path("session_data/sessions.xlsx"),
    output_dir=Path("exports/csv")
)

# Generate summary report
report = SessionExporter.generate_summary_report(
    sessions_file=Path("session_data/sessions.xlsx")
)
```

### SessionCleaner

```python
from src.session_utils import SessionCleaner

# Cleanup artifacts for a specific session
status = SessionCleaner.cleanup_artifacts(session_record)

# Cleanup old artifacts (dry run)
results = SessionCleaner.cleanup_old_artifacts(
    sessions_file=Path("session_data/sessions.xlsx"),
    days=30,
    dry_run=True
)
```

### SessionAnalyzer

```python
from src.session_utils import SessionAnalyzer

# Analyze performance
analysis = SessionAnalyzer.analyze_performance(
    sessions_file=Path("session_data/sessions.xlsx")
)

# Find similar sessions
similar = SessionAnalyzer.find_similar_sessions(
    sessions_file=Path("session_data/sessions.xlsx"),
    reference_session_id="session_123",
    similarity_threshold=0.7
)
```

## Excel Export Structure

The exported Excel file contains multiple sheets:

### 1. Sessions Sheet
Main session information including:
- session_id, job_id, filename
- target_column, problem_type, status
- created_at, updated_at, completed_at
- inferred_problem_type, best_model_name
- progress

### 2. Artifacts Sheet
Paths to all generated artifacts:
- generated_code_path
- results_path
- model_path
- ai_summary_path
- workflow_info_path

### 3. Errors Sheet
Failed sessions with:
- session_id
- error_message
- created_at
- status

### 4. Metrics Sheet
Detailed model metrics for each session

### 5. Statistics Sheet
Summary statistics:
- Total sessions
- Completed/Failed/Running counts
- Average progress
- Problem type distribution

## Best Practices

### 1. Session Management
- Always create a new session for each training
- Check session status before accessing artifacts
- Handle failed sessions appropriately
- Clean up old sessions periodically

### 2. Error Handling
- Check error_message field for failed sessions
- Use error_traceback for detailed debugging
- Implement retry logic for transient failures

### 3. Performance
- Use pagination when listing sessions
- Filter by status to reduce data retrieval
- Cache active session data in memory
- Export to Excel periodically for backup

### 4. Cleanup
- Schedule regular cleanup of old sessions
- Use dry_run mode first to verify cleanup
- Archive important sessions before deletion
- Monitor storage usage

## Migration from Old System

If migrating from the old job-based system:

1. **Data Migration**
   ```python
   # Old jobs_db to new session format
   for job_id, job_data in jobs_db.items():
       session_manager.create_session(
           filename=job_data['filename'],
           file_path=job_data['file_path'],
           target_column=job_data['target_column'],
           ...
       )
   ```

2. **API Updates**
   - Change `/jobs/{job_id}` to `/sessions/{session_id}`
   - Update client code to use session_id
   - Update artifact download endpoints

3. **Configuration**
   - Set `storage_dir` for session data
   - Configure cleanup schedule
   - Set up export automation

## Troubleshooting

### Session Not Found
```python
session = session_manager.get_session(session_id)
if not session:
    # Session may have been deleted or doesn't exist
    # Check Excel file directly
```

### Progress Stuck
```python
# Check current step
session = session_manager.get_session(session_id)
print(f"Stuck at: {session.current_step}")
print(f"Progress: {session.progress}")
```

### Failed Session Recovery
```python
# Get error details
session = session_manager.get_session(session_id)
if session.status == SessionStatus.FAILED:
    print(f"Error: {session.error_message}")
    print(f"Traceback: {session.error_traceback}")
```

## Configuration

### Environment Variables
```bash
# Redis (optional, falls back to in-memory)
REDIS_URL=redis://localhost:6379

# LLM Configuration
GROQ_API_KEY=your_api_key

# Storage
SESSION_STORAGE_DIR=session_data
MAX_SESSION_AGE_DAYS=30
```

### Logging
```python
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/app.log'),
        logging.StreamHandler()
    ]
)
```

## Performance Considerations

- **In-Memory Cache**: Active sessions cached for fast access
- **Lazy Loading**: Completed sessions loaded only when needed
- **Batch Operations**: Use batch endpoints for multiple operations
- **Pagination**: Always use pagination for large result sets
- **Indexing**: Excel files indexed by session_id for quick lookups

## Security

- Validate all input data
- Sanitize file names
- Check file types before processing
- Implement rate limiting
- Use authentication for sensitive endpoints
- Audit session access logs

## Future Enhancements

1. **Session Collaboration**: Share sessions between users
2. **Session Templates**: Save and reuse session configurations
3. **Advanced Analytics**: ML-powered session insights
4. **Real-time Updates**: WebSocket support for progress
5. **Automated Reporting**: Scheduled email reports
6. **Session Comparison**: Side-by-side session comparison

## Support

For issues or questions:
- Check logs in `logs/app.log`
- Review session Excel file
- Check artifact paths
- Verify file permissions
- Contact system administrator

## License

[Your License Here]