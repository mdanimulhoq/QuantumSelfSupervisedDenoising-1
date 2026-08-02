"""
Tests for logging utilities.
"""
import tempfile
import logging
from pathlib import Path
import pytest
from src.utils.logging import setup_logger

def test_setup_logger_returns_logger():
    """setup_logger should return a logging.Logger instance."""
    logger = setup_logger("test_logger")
    assert isinstance(logger, logging.Logger)

def test_setup_logger_creates_file():
    """setup_logger should create a log file when log_dir is provided."""
    with tempfile.TemporaryDirectory() as tmpdir:
        log_dir = Path(tmpdir)
        logger = setup_logger("test_logger", log_dir=log_dir)
        
        # Check that a log file was created
        log_files = list(log_dir.glob("*.txt"))
        assert len(log_files) > 0
        
        # Check content
        with open(log_files[0]) as f:
            content = f.read()
            assert len(content) > 0

def test_setup_logger_levels():
    """setup_logger should respect the level parameter."""
    logger = setup_logger("test_logger", level=logging.WARNING)
    assert logger.level == logging.WARNING
