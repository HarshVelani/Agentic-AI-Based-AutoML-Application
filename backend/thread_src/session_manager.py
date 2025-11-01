"""
Thread-Safe Session Manager Module with File Locking
Manages ML training sessions with parallel execution support
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Any, Optional, List
from pathlib import Path
import json
import uuid
import logging
from dataclasses import dataclass, asdict
from enum import Enum
import threading
import time
from filelock import FileLock
import os

logger = logging.getLogger(__name__)


class SessionStatus(str, Enum):
    """Session status enumeration"""
    CREATED = "created"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class SessionRecord:
    """Data class for session records"""
    session_id: str
    job_id: str
    filename: str
    file_path: str
    target_column: str
    problem_type: Optional[str]
    tune_model: bool
    user_comments: Optional[str]
    status: SessionStatus
    created_at: datetime
    updated_at: datetime
    completed_at: Optional[datetime]
    
    # Thread information
    thread_id: Optional[str] = None
    thread_name: Optional[str] = None
    
    # Paths to generated artifacts
    generated_code_path: Optional[str] = None
    results_path: Optional[str] = None
    model_path: Optional[str] = None
    ai_summary_path: Optional[str] = None
    workflow_info_path: Optional[str] = None
    
    # Results summary
    inferred_problem_type: Optional[str] = None
    best_model_name: Optional[str] = None
    metrics: Optional[Dict[str, Any]] = None
    
    # Error tracking
    error_message: Optional[str] = None
    error_traceback: Optional[str] = None
    
    # Progress tracking
    progress: float = 0.0
    current_step: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary with datetime serialization"""
        data = asdict(self)
        data['created_at'] = self.created_at.isoformat()
        data['updated_at'] = self.updated_at.isoformat()
        data['completed_at'] = self.completed_at.isoformat() if self.completed_at else None
        data['status'] = self.status.value
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SessionRecord':
        """Create SessionRecord from dictionary"""
        data['created_at'] = datetime.fromisoformat(data['created_at'])
        data['updated_at'] = datetime.fromisoformat(data['updated_at'])
        if data.get('completed_at'):
            data['completed_at'] = datetime.fromisoformat(data['completed_at'])
        data['status'] = SessionStatus(data['status'])
        return cls(**data)


class ThreadSafeSessionManager:
    """Thread-safe session manager with file locking for parallel execution"""
    
    def __init__(self, storage_dir: str = "session_data"):
        """
        Initialize Thread-Safe SessionManager
        
        Args:
            storage_dir: Directory to store session data
        """
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(exist_ok=True)
        
        self.sessions_file = self.storage_dir / "sessions.xlsx"
        self.sessions_json = self.storage_dir / "sessions_backup.json"
        self.lock_file = self.storage_dir / "sessions.lock"
        
        # Thread-safe lock for in-memory operations
        self._memory_lock = threading.RLock()
        
        # File lock for Excel operations
        self._file_lock = FileLock(str(self.lock_file), timeout=30)
        
        # In-memory cache for active sessions (thread-safe)
        self.active_sessions: Dict[str, SessionRecord] = {}
        
        # Thread tracking
        self.active_threads: Dict[str, threading.Thread] = {}
        self._thread_lock = threading.RLock()
        
        # Load existing sessions
        self._load_sessions()
        
        logger.info(f"ThreadSafeSessionManager initialized with {len(self.active_sessions)} existing sessions")
    
    def _load_sessions(self):
        """Load existing sessions from storage with file locking"""
        try:
            with self._file_lock:
                if self.sessions_file.exists():
                    df = pd.read_excel(self.sessions_file, engine='openpyxl')
                    
                    with self._memory_lock:
                        for _, row in df.iterrows():
                            try:
                                record = SessionRecord(
                                    session_id=row['session_id'],
                                    job_id=row['job_id'],
                                    filename=row['filename'],
                                    file_path=row['file_path'],
                                    target_column=row['target_column'],
                                    problem_type=row.get('problem_type'),
                                    tune_model=bool(row.get('tune_model', False)),
                                    user_comments=row.get('user_comments'),
                                    status=SessionStatus(row['status']),
                                    created_at=pd.to_datetime(row['created_at']).to_pydatetime(),
                                    updated_at=pd.to_datetime(row['updated_at']).to_pydatetime(),
                                    completed_at=pd.to_datetime(row['completed_at']).to_pydatetime() if pd.notna(row.get('completed_at')) else None,
                                    thread_id=row.get('thread_id'),
                                    thread_name=row.get('thread_name'),
                                    generated_code_path=row.get('generated_code_path'),
                                    results_path=row.get('results_path'),
                                    model_path=row.get('model_path'),
                                    ai_summary_path=row.get('ai_summary_path'),
                                    workflow_info_path=row.get('workflow_info_path'),
                                    inferred_problem_type=row.get('inferred_problem_type'),
                                    best_model_name=row.get('best_model_name'),
                                    metrics=json.loads(row['metrics']) if pd.notna(row.get('metrics')) else None,
                                    error_message=row.get('error_message'),
                                    error_traceback=row.get('error_traceback'),
                                    progress=float(row.get('progress', 0.0)),
                                    current_step=row.get('current_step')
                                )
                                
                                # Only load active sessions into memory
                                if record.status in [SessionStatus.CREATED, SessionStatus.RUNNING]:
                                    self.active_sessions[record.session_id] = record
                            except Exception as e:
                                logger.error(f"Error loading session record: {e}")
                                continue
                            
                    logger.info(f"Loaded {len(df)} sessions from Excel")
        except Exception as e:
            logger.warning(f"Could not load sessions from Excel: {e}")
    
    def create_session(
        self,
        filename: str,
        file_path: str,
        target_column: str,
        problem_type: Optional[str] = None,
        tune_model: bool = False,
        user_comments: Optional[str] = None,
        job_id: Optional[str] = None
    ) -> SessionRecord:
        """
        Create a new training session (thread-safe)
        
        Args:
            filename: Original filename
            file_path: Path to uploaded file
            target_column: Target column name
            problem_type: Problem type (classification/regression/auto)
            tune_model: Whether to perform hyperparameter tuning
            user_comments: Optional user comments
            job_id: Optional job ID (generated if not provided)
            
        Returns:
            SessionRecord: Created session record
        """
        # Generate session ID
        current_time = datetime.now().strftime("%Y_%m_%d__%H_%M_%S")
        session_uuid = str(uuid.uuid4())
        session_id = f"{current_time}__{session_uuid}"
        
        # Use provided job_id or create new one
        if not job_id:
            job_id = session_id
        
        # Get thread information
        current_thread = threading.current_thread()
        thread_id = str(current_thread.ident)
        thread_name = current_thread.name
        
        # Create session record
        record = SessionRecord(
            session_id=session_id,
            job_id=job_id,
            filename=filename,
            file_path=file_path,
            target_column=target_column,
            problem_type=problem_type,
            tune_model=tune_model,
            user_comments=user_comments,
            status=SessionStatus.CREATED,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            completed_at=None,
            thread_id=thread_id,
            thread_name=thread_name,
            progress=0.0,
            current_step="Session created"
        )
        
        # Store in active sessions (thread-safe)
        with self._memory_lock:
            self.active_sessions[session_id] = record
        
        # Persist to storage
        self._save_sessions()
        
        logger.info(f"Created new session: {session_id} on thread {thread_name}")
        return record
    
    def update_session(
        self,
        session_id: str,
        status: Optional[SessionStatus] = None,
        progress: Optional[float] = None,
        current_step: Optional[str] = None,
        **kwargs
    ) -> Optional[SessionRecord]:
        """
        Update session record (thread-safe)
        
        Args:
            session_id: Session ID
            status: New status
            progress: Progress percentage (0.0 to 1.0)
            current_step: Current processing step
            **kwargs: Additional fields to update
            
        Returns:
            Updated SessionRecord or None if not found
        """
        with self._memory_lock:
            record = self.get_session(session_id)
            if not record:
                logger.warning(f"Session {session_id} not found for update")
                return None
            
            # Update fields
            if status:
                record.status = status
            if progress is not None:
                record.progress = progress
            if current_step:
                record.current_step = current_step
            
            # Update additional fields
            for key, value in kwargs.items():
                if hasattr(record, key):
                    setattr(record, key, value)
            
            # Update timestamp
            record.updated_at = datetime.now()
            
            # Mark as completed if status is completed or failed
            if status in [SessionStatus.COMPLETED, SessionStatus.FAILED, SessionStatus.CANCELLED]:
                record.completed_at = datetime.now()
                # Remove from active sessions
                if session_id in self.active_sessions:
                    # Keep in memory for a bit to allow final reads
                    pass
        
        # Persist changes (outside memory lock to avoid deadlock)
        self._save_sessions()
        
        logger.debug(f"Updated session {session_id}: status={status}, progress={progress}")
        return record
    
    def get_session(self, session_id: str) -> Optional[SessionRecord]:
        """
        Get session record by ID (thread-safe)
        
        Args:
            session_id: Session ID
            
        Returns:
            SessionRecord or None if not found
        """
        # Check active sessions first (thread-safe)
        with self._memory_lock:
            if session_id in self.active_sessions:
                return self.active_sessions[session_id]
        
        # Check persisted storage with file lock
        try:
            with self._file_lock:
                if self.sessions_file.exists():
                    df = pd.read_excel(self.sessions_file, engine='openpyxl')
                    session_df = df[df['session_id'] == session_id]
                    if not session_df.empty:
                        row = session_df.iloc[0]
                        return SessionRecord(
                            session_id=row['session_id'],
                            job_id=row['job_id'],
                            filename=row['filename'],
                            file_path=row['file_path'],
                            target_column=row['target_column'],
                            problem_type=row.get('problem_type'),
                            tune_model=bool(row.get('tune_model', False)),
                            user_comments=row.get('user_comments'),
                            status=SessionStatus(row['status']),
                            created_at=pd.to_datetime(row['created_at']).to_pydatetime(),
                            updated_at=pd.to_datetime(row['updated_at']).to_pydatetime(),
                            completed_at=pd.to_datetime(row['completed_at']).to_pydatetime() if pd.notna(row.get('completed_at')) else None,
                            thread_id=row.get('thread_id'),
                            thread_name=row.get('thread_name'),
                            generated_code_path=row.get('generated_code_path'),
                            results_path=row.get('results_path'),
                            model_path=row.get('model_path'),
                            ai_summary_path=row.get('ai_summary_path'),
                            workflow_info_path=row.get('workflow_info_path'),
                            inferred_problem_type=row.get('inferred_problem_type'),
                            best_model_name=row.get('best_model_name'),
                            metrics=json.loads(row['metrics']) if pd.notna(row.get('metrics')) else None,
                            error_message=row.get('error_message'),
                            error_traceback=row.get('error_traceback'),
                            progress=float(row.get('progress', 0.0)),
                            current_step=row.get('current_step')
                        )
        except Exception as e:
            logger.error(f"Error retrieving session {session_id}: {e}")
        
        return None
    
    def list_sessions(
        self,
        status: Optional[SessionStatus] = None,
        limit: int = 50,
        offset: int = 0
    ) -> tuple[List[SessionRecord], int]:
        """
        List sessions with optional filtering (thread-safe)
        
        Args:
            status: Filter by status
            limit: Maximum number of records
            offset: Offset for pagination
            
        Returns:
            Tuple of (list of SessionRecords, total count)
        """
        try:
            with self._file_lock:
                if self.sessions_file.exists():
                    df = pd.read_excel(self.sessions_file, engine='openpyxl')
                    
                    # Filter by status if provided
                    if status:
                        df = df[df['status'] == status.value]
                    
                    total = len(df)
                    
                    # Sort by created_at descending
                    df = df.sort_values('created_at', ascending=False)
                    
                    # Apply pagination
                    df = df.iloc[offset:offset + limit]
                    
                    # Convert to SessionRecords
                    records = []
                    for _, row in df.iterrows():
                        try:
                            record = SessionRecord(
                                session_id=row['session_id'],
                                job_id=row['job_id'],
                                filename=row['filename'],
                                file_path=row['file_path'],
                                target_column=row['target_column'],
                                problem_type=row.get('problem_type'),
                                tune_model=bool(row.get('tune_model', False)),
                                user_comments=row.get('user_comments'),
                                status=SessionStatus(row['status']),
                                created_at=pd.to_datetime(row['created_at']).to_pydatetime(),
                                updated_at=pd.to_datetime(row['updated_at']).to_pydatetime(),
                                completed_at=pd.to_datetime(row['completed_at']).to_pydatetime() if pd.notna(row.get('completed_at')) else None,
                                thread_id=row.get('thread_id'),
                                thread_name=row.get('thread_name'),
                                generated_code_path=row.get('generated_code_path'),
                                results_path=row.get('results_path'),
                                model_path=row.get('model_path'),
                                ai_summary_path=row.get('ai_summary_path'),
                                workflow_info_path=row.get('workflow_info_path'),
                                inferred_problem_type=row.get('inferred_problem_type'),
                                best_model_name=row.get('best_model_name'),
                                metrics=json.loads(row['metrics']) if pd.notna(row.get('metrics')) else None,
                                error_message=row.get('error_message'),
                                progress=float(row.get('progress', 0.0)),
                                current_step=row.get('current_step')
                            )
                            records.append(record)
                        except Exception as e:
                            logger.error(f"Error parsing session record: {e}")
                            continue
                    
                    return records, total
        except Exception as e:
            logger.error(f"Error listing sessions: {e}")
        
        return [], 0
    
    def delete_session(self, session_id: str) -> bool:
        """
        Delete session (thread-safe)
        
        Args:
            session_id: Session ID
            
        Returns:
            True if deleted, False otherwise
        """
        record = self.get_session(session_id)
        if not record:
            return False
        
        # Remove from active sessions
        with self._memory_lock:
            if session_id in self.active_sessions:
                del self.active_sessions[session_id]
        
        # Persist changes
        self._save_sessions()
        
        logger.info(f"Deleted session: {session_id}")
        return True
    
    def get_session_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about all sessions (thread-safe)
        
        Returns:
            Dictionary with statistics
        """
        try:
            with self._file_lock:
                if self.sessions_file.exists():
                    df = pd.read_excel(self.sessions_file, engine='openpyxl')
                    
                    with self._thread_lock:
                        active_thread_count = len([t for t in self.active_threads.values() if t.is_alive()])
                    
                    stats = {
                        'total_sessions': len(df),
                        'status_distribution': df['status'].value_counts().to_dict(),
                        'problem_type_distribution': df['inferred_problem_type'].value_counts().to_dict() if 'inferred_problem_type' in df.columns else {},
                        'average_progress': float(df['progress'].mean()) if 'progress' in df.columns else 0.0,
                        'successful_sessions': len(df[df['status'] == SessionStatus.COMPLETED.value]),
                        'failed_sessions': len(df[df['status'] == SessionStatus.FAILED.value]),
                        'active_sessions': len(df[df['status'].isin([SessionStatus.CREATED.value, SessionStatus.RUNNING.value])]),
                        'active_threads': active_thread_count,
                        'total_threads': len(self.active_threads)
                    }
                    
                    return stats
        except Exception as e:
            logger.error(f"Error calculating statistics: {e}")
        
        return {}
    
    def _save_sessions(self):
        """Persist sessions to storage with file locking"""
        max_retries = 3
        retry_delay = 1
        
        for attempt in range(max_retries):
            try:
                with self._file_lock:
                    # Load all existing sessions
                    all_sessions = []
                    existing_ids = set()
                    
                    if self.sessions_file.exists():
                        try:
                            df = pd.read_excel(self.sessions_file, engine='openpyxl')
                            existing_ids = set(df['session_id'].tolist())
                            
                            # Add existing sessions not in active memory
                            for _, row in df.iterrows():
                                with self._memory_lock:
                                    if row['session_id'] not in self.active_sessions:
                                        all_sessions.append(row.to_dict())
                        except Exception as e:
                            logger.error(f"Error loading existing sessions: {e}")
                    
                    # Add/update active sessions
                    with self._memory_lock:
                        for session_id, record in self.active_sessions.items():
                            session_dict = record.to_dict()
                            
                            # Convert metrics to JSON string
                            if session_dict.get('metrics'):
                                session_dict['metrics'] = json.dumps(session_dict['metrics'])
                            
                            all_sessions.append(session_dict)
                    
                    # Create DataFrame and save to Excel
                    if all_sessions:
                        df = pd.DataFrame(all_sessions)
                        
                        # Remove duplicates, keeping the latest
                        df = df.drop_duplicates(subset=['session_id'], keep='last')
                        
                        # Save to Excel with openpyxl engine
                        df.to_excel(self.sessions_file, index=False, engine='openpyxl')
                        
                        # Also save JSON backup
                        with self._memory_lock:
                            with open(self.sessions_json, 'w') as f:
                                json.dump([record.to_dict() for record in self.active_sessions.values()], f, indent=2)
                        
                        logger.debug(f"Saved {len(df)} sessions to storage")
                
                # Success - break retry loop
                break
                
            except Exception as e:
                logger.error(f"Error saving sessions (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                else:
                    logger.error("Failed to save sessions after all retries")
    
    def register_thread(self, session_id: str, thread: threading.Thread):
        """Register a thread for a session"""
        with self._thread_lock:
            self.active_threads[session_id] = thread
        logger.info(f"Registered thread {thread.name} for session {session_id}")
    
    def unregister_thread(self, session_id: str):
        """Unregister a thread for a session"""
        with self._thread_lock:
            if session_id in self.active_threads:
                del self.active_threads[session_id]
        logger.info(f"Unregistered thread for session {session_id}")
    
    def cleanup_old_sessions(self, days: int = 30) -> int:
        """
        Remove sessions older than specified days (thread-safe)
        
        Args:
            days: Number of days to keep
            
        Returns:
            Number of sessions removed
        """
        try:
            with self._file_lock:
                if not self.sessions_file.exists():
                    return 0
                
                df = pd.read_excel(self.sessions_file, engine='openpyxl')
                cutoff_date = datetime.now() - pd.Timedelta(days=days)
                
                # Filter sessions
                old_sessions = df[pd.to_datetime(df['created_at']) < cutoff_date]
                removed_count = len(old_sessions)
                
                # Keep recent sessions
                df = df[pd.to_datetime(df['created_at']) >= cutoff_date]
                
                # Save updated data
                df.to_excel(self.sessions_file, index=False, engine='openpyxl')
                
                logger.info(f"Cleaned up {removed_count} sessions older than {days} days")
                return removed_count
        except Exception as e:
            logger.error(f"Error cleaning up old sessions: {e}")
            return 0