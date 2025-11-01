"""
Session Utilities Module
Helper functions for session management and data export
"""

import pandas as pd
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class SessionExporter:
    """Utility class for exporting session data"""
    
    @staticmethod
    def export_to_excel(
        sessions_file: Path,
        output_path: Optional[Path] = None,
        include_metrics: bool = True
    ) -> Path:
        """
        Export sessions to Excel with formatted output
        
        Args:
            sessions_file: Path to sessions Excel file
            output_path: Output path (generated if not provided)
            include_metrics: Whether to include detailed metrics
            
        Returns:
            Path to exported file
        """
        try:
            # Load sessions
            df = pd.read_excel(sessions_file)
            
            # Create output path if not provided
            if not output_path:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_path = Path(f"exports/sessions_export_{timestamp}.xlsx")
                output_path.parent.mkdir(exist_ok=True)
            
            # Create Excel writer
            with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
                # Main sessions sheet
                main_df = df[[
                    'session_id', 'job_id', 'filename', 'target_column',
                    'problem_type', 'status', 'created_at', 'updated_at',
                    'inferred_problem_type', 'best_model_name', 'progress'
                ]].copy()
                main_df.to_excel(writer, sheet_name='Sessions', index=False)
                
                # Artifacts sheet
                artifacts_df = df[[
                    'session_id', 'generated_code_path', 'results_path',
                    'model_path', 'ai_summary_path', 'workflow_info_path'
                ]].copy()
                artifacts_df.to_excel(writer, sheet_name='Artifacts', index=False)
                
                # Errors sheet
                errors_df = df[df['error_message'].notna()][[
                    'session_id', 'error_message', 'created_at', 'status'
                ]].copy()
                errors_df.to_excel(writer, sheet_name='Errors', index=False)
                
                # Metrics sheet (if requested)
                if include_metrics:
                    metrics_data = []
                    for _, row in df.iterrows():
                        if pd.notna(row.get('metrics')):
                            try:
                                metrics = json.loads(row['metrics']) if isinstance(row['metrics'], str) else row['metrics']
                                for model_name, model_metrics in metrics.items():
                                    metrics_data.append({
                                        'session_id': row['session_id'],
                                        'model_name': model_name,
                                        **model_metrics
                                    })
                            except Exception as e:
                                logger.warning(f"Could not parse metrics for session {row['session_id']}: {e}")
                    
                    if metrics_data:
                        metrics_df = pd.DataFrame(metrics_data)
                        metrics_df.to_excel(writer, sheet_name='Metrics', index=False)
                
                # Statistics sheet
                stats_data = {
                    'Total Sessions': [len(df)],
                    'Completed': [len(df[df['status'] == 'completed'])],
                    'Failed': [len(df[df['status'] == 'failed'])],
                    'Running': [len(df[df['status'] == 'running'])],
                    'Average Progress': [df['progress'].mean()],
                    'Classification Problems': [len(df[df['inferred_problem_type'] == 'classification'])],
                    'Regression Problems': [len(df[df['inferred_problem_type'] == 'regression'])]
                }
                stats_df = pd.DataFrame(stats_data)
                stats_df.to_excel(writer, sheet_name='Statistics', index=False)
            
            logger.info(f"Exported sessions to {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Failed to export sessions: {e}")
            raise
    
    @staticmethod
    def export_to_csv(
        sessions_file: Path,
        output_dir: Optional[Path] = None
    ) -> List[Path]:
        """
        Export sessions to multiple CSV files
        
        Args:
            sessions_file: Path to sessions Excel file
            output_dir: Output directory (created if not exists)
            
        Returns:
            List of paths to exported CSV files
        """
        try:
            # Load sessions
            df = pd.read_excel(sessions_file)
            
            # Create output directory
            if not output_dir:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_dir = Path(f"exports/csv_export_{timestamp}")
            output_dir.mkdir(parents=True, exist_ok=True)
            
            exported_files = []
            
            # Export main sessions
            main_path = output_dir / "sessions.csv"
            df.to_csv(main_path, index=False)
            exported_files.append(main_path)
            
            # Export by status
            for status in df['status'].unique():
                if pd.notna(status):
                    status_df = df[df['status'] == status]
                    status_path = output_dir / f"sessions_{status}.csv"
                    status_df.to_csv(status_path, index=False)
                    exported_files.append(status_path)
            
            logger.info(f"Exported {len(exported_files)} CSV files to {output_dir}")
            return exported_files
            
        except Exception as e:
            logger.error(f"Failed to export to CSV: {e}")
            raise
    
    @staticmethod
    def generate_summary_report(sessions_file: Path) -> Dict[str, Any]:
        """
        Generate a summary report of all sessions
        
        Args:
            sessions_file: Path to sessions Excel file
            
        Returns:
            Dictionary with summary statistics
        """
        try:
            df = pd.read_excel(sessions_file)
            
            report = {
                "total_sessions": len(df),
                "status_distribution": df['status'].value_counts().to_dict(),
                "problem_type_distribution": df['inferred_problem_type'].value_counts().to_dict() if 'inferred_problem_type' in df.columns else {},
                "successful_rate": len(df[df['status'] == 'completed']) / len(df) * 100 if len(df) > 0 else 0,
                "average_progress": float(df['progress'].mean()) if 'progress' in df.columns else 0,
                "date_range": {
                    "earliest": df['created_at'].min().isoformat() if len(df) > 0 else None,
                    "latest": df['created_at'].max().isoformat() if len(df) > 0 else None
                },
                "top_models": [],
                "common_errors": []
            }
            
            # Top models
            if 'best_model_name' in df.columns:
                top_models = df['best_model_name'].value_counts().head(5)
                report["top_models"] = [
                    {"model": model, "count": int(count)}
                    for model, count in top_models.items()
                ]
            
            # Common errors
            if 'error_message' in df.columns:
                errors_df = df[df['error_message'].notna()]
                if len(errors_df) > 0:
                    # Group similar errors
                    error_counts = errors_df['error_message'].str[:50].value_counts().head(5)
                    report["common_errors"] = [
                        {"error_preview": error, "count": int(count)}
                        for error, count in error_counts.items()
                    ]
            
            return report
            
        except Exception as e:
            logger.error(f"Failed to generate summary report: {e}")
            raise


class SessionCleaner:
    """Utility class for cleaning up session data and artifacts"""
    
    @staticmethod
    def cleanup_artifacts(session_record: str = 'SessionRecord') -> Dict[str, bool]:
        """
        Clean up all artifacts for a session
        
        Args:
            session_record: SessionRecord instance
            
        Returns:
            Dictionary with cleanup status for each artifact
        """
        cleanup_status = {}
        
        # List of artifact paths
        artifacts = {
            'file': session_record.file_path,
            'code': session_record.generated_code_path,
            'results': session_record.results_path,
            'model': session_record.model_path,
            'summary': session_record.ai_summary_path,
            'workflow': session_record.workflow_info_path
        }
        
        for artifact_type, path in artifacts.items():
            if path and Path(path).exists():
                try:
                    Path(path).unlink()
                    cleanup_status[artifact_type] = True
                    logger.info(f"Deleted {artifact_type} artifact: {path}")
                except Exception as e:
                    cleanup_status[artifact_type] = False
                    logger.error(f"Failed to delete {artifact_type} artifact {path}: {e}")
            else:
                cleanup_status[artifact_type] = None  # Not found
        
        return cleanup_status
    
    @staticmethod
    def cleanup_old_artifacts(
        sessions_file: Path,
        days: int = 30,
        dry_run: bool = True
    ) -> Dict[str, Any]:
        """
        Clean up artifacts for sessions older than specified days
        
        Args:
            sessions_file: Path to sessions Excel file
            days: Age threshold in days
            dry_run: If True, only report what would be deleted
            
        Returns:
            Dictionary with cleanup results
        """
        try:
            df = pd.read_excel(sessions_file)
            cutoff_date = datetime.now() - pd.Timedelta(days=days)
            
            # Find old sessions
            old_sessions = df[pd.to_datetime(df['created_at']) < cutoff_date]
            
            results = {
                'total_sessions': len(old_sessions),
                'dry_run': dry_run,
                'deleted_artifacts': {},
                'errors': []
            }
            
            for _, session in old_sessions.iterrows():
                session_id = session['session_id']
                
                # Paths to clean
                paths = [
                    session.get('file_path'),
                    session.get('generated_code_path'),
                    session.get('results_path'),
                    session.get('model_path'),
                    session.get('ai_summary_path'),
                    session.get('workflow_info_path')
                ]
                
                deleted_count = 0
                for path in paths:
                    if path and pd.notna(path) and Path(path).exists():
                        if not dry_run:
                            try:
                                Path(path).unlink()
                                deleted_count += 1
                            except Exception as e:
                                results['errors'].append(f"Failed to delete {path}: {e}")
                        else:
                            deleted_count += 1
                
                results['deleted_artifacts'][session_id] = deleted_count
            
            logger.info(f"Cleanup {'would delete' if dry_run else 'deleted'} artifacts for {len(old_sessions)} sessions")
            return results
            
        except Exception as e:
            logger.error(f"Failed to cleanup artifacts: {e}")
            raise


class SessionAnalyzer:
    """Utility class for analyzing session data"""
    
    @staticmethod
    def analyze_performance(sessions_file: Path) -> Dict[str, Any]:
        """
        Analyze performance metrics across sessions
        
        Args:
            sessions_file: Path to sessions Excel file
            
        Returns:
            Dictionary with performance analysis
        """
        try:
            df = pd.read_excel(sessions_file)
            
            # Filter completed sessions
            completed = df[df['status'] == 'completed']
            
            analysis = {
                "total_completed": len(completed),
                "problem_type_performance": {},
                "model_performance": {},
                "average_metrics": {}
            }
            
            # Performance by problem type
            for problem_type in completed['inferred_problem_type'].unique():
                if pd.notna(problem_type):
                    type_df = completed[completed['inferred_problem_type'] == problem_type]
                    analysis["problem_type_performance"][problem_type] = {
                        "count": len(type_df),
                        "success_rate": 100.0  # All are completed
                    }
            
            # Model performance
            if 'best_model_name' in completed.columns:
                model_counts = completed['best_model_name'].value_counts()
                analysis["model_performance"] = {
                    model: {
                        "times_selected": int(count),
                        "percentage": float(count / len(completed) * 100)
                    }
                    for model, count in model_counts.items()
                    if pd.notna(model)
                }
            
            # Average metrics
            if 'metrics' in completed.columns:
                all_metrics = []
                for metrics_str in completed['metrics'].dropna():
                    try:
                        metrics = json.loads(metrics_str) if isinstance(metrics_str, str) else metrics_str
                        all_metrics.append(metrics)
                    except:
                        pass
                
                if all_metrics:
                    # Aggregate metrics
                    analysis["average_metrics"] = {
                        "total_sessions_with_metrics": len(all_metrics)
                    }
            
            return analysis
            
        except Exception as e:
            logger.error(f"Failed to analyze performance: {e}")
            raise
    
    @staticmethod
    def find_similar_sessions(
        sessions_file: Path,
        reference_session_id: str,
        similarity_threshold: float = 0.7
    ) -> List[Dict[str, Any]]:
        """
        Find sessions similar to a reference session
        
        Args:
            sessions_file: Path to sessions Excel file
            reference_session_id: Session ID to compare against
            similarity_threshold: Minimum similarity score (0-1)
            
        Returns:
            List of similar sessions with similarity scores
        """
        try:
            df = pd.read_excel(sessions_file)
            
            # Find reference session
            ref_session = df[df['session_id'] == reference_session_id]
            if ref_session.empty:
                return []
            
            ref = ref_session.iloc[0]
            similar_sessions = []
            
            for _, session in df.iterrows():
                if session['session_id'] == reference_session_id:
                    continue
                
                # Calculate similarity
                similarity_score = 0.0
                factors = 0
                
                # Problem type match
                if session['inferred_problem_type'] == ref['inferred_problem_type']:
                    similarity_score += 0.3
                factors += 0.3
                
                # Best model match
                if session.get('best_model_name') == ref.get('best_model_name'):
                    similarity_score += 0.3
                factors += 0.3
                
                # Status match
                if session['status'] == ref['status']:
                    similarity_score += 0.2
                factors += 0.2
                
                # Tune model match
                if session.get('tune_model') == ref.get('tune_model'):
                    similarity_score += 0.2
                factors += 0.2
                
                # Normalize score
                if factors > 0:
                    similarity_score = similarity_score / factors
                
                if similarity_score >= similarity_threshold:
                    similar_sessions.append({
                        'session_id': session['session_id'],
                        'similarity_score': similarity_score,
                        'filename': session['filename'],
                        'status': session['status'],
                        'created_at': session['created_at']
                    })
            
            # Sort by similarity score
            similar_sessions.sort(key=lambda x: x['similarity_score'], reverse=True)
            
            return similar_sessions
            
        except Exception as e:
            logger.error(f"Failed to find similar sessions: {e}")
            raise