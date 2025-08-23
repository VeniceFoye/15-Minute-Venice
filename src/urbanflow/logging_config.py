"""
Centralized logging configuration for the urbanflow package.

This module sets up a logger for the urbanflow package that can be
imported and used throughout the codebase to replace print statements
with proper logging calls.
"""

import logging
import sys
from typing import Optional


def setup_logger(
    name: str = "urbanflow",
    level: int = logging.INFO,
    format_string: Optional[str] = None
) -> logging.Logger:
    """
    Set up and configure a logger for the urbanflow package.
    
    Parameters
    ----------
    name : str, optional
        Name of the logger. Defaults to "urbanflow".
    level : int, optional
        Logging level. Defaults to logging.INFO.
    format_string : str, optional
        Custom format string for log messages. If None, uses a default format.
        
    Returns
    -------
    logging.Logger
        Configured logger instance.
    """
    logger = logging.getLogger(name)
    
    # Avoid adding multiple handlers if logger is already configured
    if logger.handlers:
        return logger
    
    # Set log level
    logger.setLevel(level)
    
    # Create console handler
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)
    
    # Create formatter
    if format_string is None:
        format_string = "%(name)s - %(levelname)s - %(message)s"
    
    formatter = logging.Formatter(format_string)
    handler.setFormatter(formatter)
    
    # Add handler to logger
    logger.addHandler(handler)
    
    return logger


# Create the default package logger
logger = setup_logger()