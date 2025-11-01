"""
Thread-Safe LLM Manager with Connection Pooling and Error Handling
"""

from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os
import threading
import logging
from typing import Optional, Dict, Any
from langchain_core.prompts import ChatPromptTemplate
from functools import lru_cache
import time

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)


class ThreadSafeLLMManager:
    """
    Thread-safe manager for LLM interactions with connection pooling
    and error handling for parallel execution
    """
    
    # Class-level cache for LLM instances (one per thread)
    _thread_local = threading.local()
    _lock = threading.RLock()
    
    # Retry configuration
    MAX_RETRIES = 3
    RETRY_DELAY = 2  # seconds
    
    def __init__(
        self,
        model: str = "openai/gpt-oss-120b",
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        timeout: int = 120
    ):
        """
        Initialize LLM Manager
        
        Args:
            model: Model name to use
            temperature: Temperature for generation (0.0 = deterministic)
            max_tokens: Maximum tokens to generate
            timeout: Request timeout in seconds
        """
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.api_key = os.getenv("GROQ_API_KEY")
        
        if not self.api_key:
            raise ValueError("GROQ_API_KEY not found in environment variables")
        
        # Validate model name
        self._validate_model()
        
        logger.info(f"LLMManager initialized with model: {model}")
    
    def _validate_model(self):
        """Validate that the model name is supported"""
        supported_models = [
            "openai/gpt-oss-120b",
            "llama-3.1-70b-versatile",
            "llama-3.1-8b-instant",
            "mixtral-8x7b-32768",
            "gemma-7b-it",
            "llama3-70b-8192",
            "llama3-8b-8192"
        ]
        
        if self.model not in supported_models:
            logger.warning(f"Model {self.model} may not be supported. Supported models: {supported_models}")
    
    def _get_llm_instance(self) -> ChatGroq:
        """
        Get thread-local LLM instance for connection pooling
        Each thread gets its own LLM instance to avoid conflicts
        """
        if not hasattr(self._thread_local, 'llm'):
            with self._lock:
                # Double-check after acquiring lock
                if not hasattr(self._thread_local, 'llm'):
                    thread_name = threading.current_thread().name
                    logger.debug(f"Creating new LLM instance for thread: {thread_name}")
                    
                    self._thread_local.llm = ChatGroq(
                        model=self.model,
                        temperature=self.temperature,
                        max_tokens=self.max_tokens,
                        timeout=self.timeout,
                        api_key=self.api_key
                    )
                    
                    logger.info(f"LLM instance created for thread: {thread_name}")
        
        return self._thread_local.llm
    
    def invoke(
        self,
        prompt: ChatPromptTemplate,
        retry: bool = True,
        **kwargs
    ) -> str:
        """
        Invoke LLM with prompt and keyword arguments (thread-safe)
        
        Args:
            prompt: ChatPromptTemplate to use
            retry: Whether to retry on failure
            **kwargs: Variables to format the prompt
            
        Returns:
            str: Generated response content
            
        Raises:
            Exception: If all retries fail
        """
        thread_name = threading.current_thread().name
        attempt = 0
        last_error = None
        
        while attempt < (self.MAX_RETRIES if retry else 1):
            try:
                # Format messages
                messages = prompt.format_messages(**kwargs)
                
                # Get thread-local LLM instance
                llm = self._get_llm_instance()
                
                # Log request
                logger.debug(f"[{thread_name}] Invoking LLM (attempt {attempt + 1}/{self.MAX_RETRIES})")
                start_time = time.time()
                
                # Invoke LLM
                response = llm.invoke(messages)
                
                # Log success
                duration = time.time() - start_time
                logger.info(
                    f"[{thread_name}] LLM response received in {duration:.2f}s "
                    f"(length: {len(response.content)} chars)"
                )
                
                return response.content
                
            except Exception as e:
                attempt += 1
                last_error = e
                
                error_msg = f"[{thread_name}] LLM invocation failed (attempt {attempt}/{self.MAX_RETRIES}): {str(e)}"
                
                if attempt < self.MAX_RETRIES and retry:
                    logger.warning(f"{error_msg}. Retrying in {self.RETRY_DELAY}s...")
                    time.sleep(self.RETRY_DELAY)
                else:
                    logger.error(error_msg)
        
        # All retries failed
        raise Exception(f"LLM invocation failed after {self.MAX_RETRIES} attempts: {str(last_error)}")
    
    async def ainvoke(
        self,
        prompt: ChatPromptTemplate,
        retry: bool = True,
        **kwargs
    ) -> str:
        """
        Async invoke LLM with prompt (for async contexts)
        
        Args:
            prompt: ChatPromptTemplate to use
            retry: Whether to retry on failure
            **kwargs: Variables to format the prompt
            
        Returns:
            str: Generated response content
        """
        thread_name = threading.current_thread().name
        attempt = 0
        last_error = None
        
        while attempt < (self.MAX_RETRIES if retry else 1):
            try:
                # Format messages
                messages = prompt.format_messages(**kwargs)
                
                # Get thread-local LLM instance
                llm = self._get_llm_instance()
                
                # Log request
                logger.debug(f"[{thread_name}] Async invoking LLM (attempt {attempt + 1}/{self.MAX_RETRIES})")
                start_time = time.time()
                
                # Async invoke LLM
                response = await llm.ainvoke(messages)
                
                # Log success
                duration = time.time() - start_time
                logger.info(
                    f"[{thread_name}] Async LLM response received in {duration:.2f}s "
                    f"(length: {len(response.content)} chars)"
                )
                
                return response.content
                
            except Exception as e:
                attempt += 1
                last_error = e
                
                error_msg = f"[{thread_name}] Async LLM invocation failed (attempt {attempt}/{self.MAX_RETRIES}): {str(e)}"
                
                if attempt < self.MAX_RETRIES and retry:
                    logger.warning(f"{error_msg}. Retrying in {self.RETRY_DELAY}s...")
                    import asyncio
                    await asyncio.sleep(self.RETRY_DELAY)
                else:
                    logger.error(error_msg)
        
        # All retries failed
        raise Exception(f"Async LLM invocation failed after {self.MAX_RETRIES} attempts: {str(last_error)}")
    
    def batch_invoke(
        self,
        prompts: list[tuple[ChatPromptTemplate, Dict[str, Any]]],
        parallel: bool = True
    ) -> list[str]:
        """
        Invoke multiple prompts in batch
        
        Args:
            prompts: List of (prompt, kwargs) tuples
            parallel: Whether to invoke in parallel (not implemented yet)
            
        Returns:
            List of response strings
        """
        thread_name = threading.current_thread().name
        logger.info(f"[{thread_name}] Batch invoking {len(prompts)} prompts")
        
        results = []
        for i, (prompt, kwargs) in enumerate(prompts):
            try:
                logger.debug(f"[{thread_name}] Processing batch item {i+1}/{len(prompts)}")
                response = self.invoke(prompt, **kwargs)
                results.append(response)
            except Exception as e:
                logger.error(f"[{thread_name}] Batch item {i+1} failed: {str(e)}")
                results.append(f"ERROR: {str(e)}")
        
        logger.info(f"[{thread_name}] Batch completed: {len(results)}/{len(prompts)} successful")
        return results
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model configuration"""
        return {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "timeout": self.timeout,
            "max_retries": self.MAX_RETRIES,
            "retry_delay": self.RETRY_DELAY
        }
    
    def cleanup_thread_local(self):
        """
        Cleanup thread-local resources
        Call this when a thread is finishing
        """
        if hasattr(self._thread_local, 'llm'):
            thread_name = threading.current_thread().name
            logger.debug(f"Cleaning up LLM instance for thread: {thread_name}")
            delattr(self._thread_local, 'llm')
    
    @classmethod
    def cleanup_all_instances(cls):
        """Cleanup all thread-local instances (call on shutdown)"""
        logger.info("Cleaning up all LLM instances")
        if hasattr(cls._thread_local, 'llm'):
            delattr(cls._thread_local, 'llm')


# Backward compatibility alias
LLMManager = ThreadSafeLLMManager


# Example usage and testing
if __name__ == "__main__":
    import concurrent.futures
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(threadName)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Test thread-safe LLM manager
    def test_llm(thread_id: int):
        """Test function for parallel execution"""
        manager = ThreadSafeLLMManager()
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful assistant."),
            ("user", "Say hello from thread {thread_id}")
        ])
        
        try:
            response = manager.invoke(prompt, thread_id=thread_id)
            print(f"Thread {thread_id} response: {response[:100]}...")
            return response
        except Exception as e:
            print(f"Thread {thread_id} error: {str(e)}")
            return None
        finally:
            manager.cleanup_thread_local()
    
    # Run parallel test
    print("Testing parallel LLM invocations...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        futures = [executor.submit(test_llm, i) for i in range(3)]
        results = [f.result() for f in concurrent.futures.as_completed(futures)]
    
    print(f"\nCompleted {len([r for r in results if r])} out of {len(results)} requests")
        