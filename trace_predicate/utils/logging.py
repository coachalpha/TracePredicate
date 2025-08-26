"""
Logging utilities for TracePredicate.

This module provides centralized logging configuration and utilities
for the TracePredicate system.
"""

import logging
import logging.handlers
import sys
from pathlib import Path
from typing import Optional

from config.settings import get_settings


def setup_logging(
    log_level: Optional[str] = None,
    log_file: Optional[Path] = None,
    console_output: bool = True
) -> None:
    """
    Set up logging configuration for the TracePredicate system.
    
    Args:
        log_level: Override the default log level
        log_file: Override the default log file path
        console_output: Whether to output to console
    """
    settings = get_settings()
    
    # Determine log level
    if log_level is None:
        log_level = settings.logging.log_level
    
    # Determine log file
    if log_file is None and settings.logging.log_file:
        log_file = Path(settings.logging.log_file)
    
    # Create formatter
    formatter = logging.Formatter(settings.logging.log_format)
    
    # Get root logger
    logger = logging.getLogger()
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Remove existing handlers
    logger.handlers.clear()
    
    # Add console handler if requested
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, log_level.upper()))
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # Add file handler if specified
    if log_file:
        # Ensure log directory exists
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=settings.logging.log_max_bytes,
            backupCount=settings.logging.log_backup_count
        )
        file_handler.setLevel(getattr(logging, log_level.upper()))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    # Set specific logger levels
    logging.getLogger('sqlalchemy').setLevel(
        getattr(logging, settings.logging.sqlalchemy_log_level.upper())
    )
    logging.getLogger('requests').setLevel(
        getattr(logging, settings.logging.requests_log_level.upper())
    )
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    
    logger.info(f"Logging initialized - Level: {log_level}, File: {log_file}")


def get_logger(name: str) -> logging.Logger:
    """Get a logger for a specific module."""
    return logging.getLogger(name)


class LoggerMixin:
    """Mixin class to add logging functionality to other classes."""
    
    @property
    def logger(self) -> logging.Logger:
        """Get a logger for this class."""
        return logging.getLogger(f"{self.__class__.__module__}.{self.__class__.__name__}")


def log_function_call(logger: logging.Logger, level: int = logging.DEBUG):
    """
    Decorator to log function calls.
    
    Args:
        logger: Logger to use
        level: Log level for the messages
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            logger.log(level, f"Calling {func.__name__} with args={args[:3]}, kwargs={list(kwargs.keys())}")
            try:
                result = func(*args, **kwargs)
                logger.log(level, f"Completed {func.__name__}")
                return result
            except Exception as e:
                logger.error(f"Error in {func.__name__}: {e}")
                raise
        return wrapper
    return decorator


def main():
    """Test logging setup."""
    setup_logging(log_level="DEBUG", console_output=True)
    
    logger = get_logger(__name__)
    
    logger.debug("Debug message")
    logger.info("Info message")
    logger.warning("Warning message")
    logger.error("Error message")
    
    print("Logging test completed")


if __name__ == "__main__":
    main()