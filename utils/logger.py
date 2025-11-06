"""
Output Logger for CLI to File.

Captures all console output (stdout and stderr) and saves to a log file
next to the output video and JSON files.
"""

import sys
from datetime import datetime
from pathlib import Path
from typing import Optional


class GoalkeeperLogger:
    """
    Dedicated logger for goalkeeper debugging that writes to a separate file.

    Usage:
        gk_logger = GoalkeeperLogger("output_videos/sample_1_goalkeeper_debug.log")
        gk_logger.log("[GK DEBUG] Frame 0: Detected 1 goalkeeper")
    """

    _instance = None

    def __init__(self, log_file_path: str):
        """
        Initialize goalkeeper logger.

        Args:
            log_file_path: Path where goalkeeper log file will be saved
        """
        self.log_file_path = Path(log_file_path)
        self.log_file = None

        # Create directory if needed
        self.log_file_path.parent.mkdir(parents=True, exist_ok=True)

        # Open log file
        self.log_file = open(self.log_file_path, 'w', encoding='utf-8', buffering=1)

        # Write header
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        header = f"""
{'='*80}
Goalkeeper Detection & Tracking Debug Log
Started: {timestamp}
Log File: {self.log_file_path}
{'='*80}

"""
        self.log_file.write(header)
        self.log_file.flush()

    @classmethod
    def get_instance(cls):
        """Get singleton instance of goalkeeper logger."""
        return cls._instance

    @classmethod
    def initialize(cls, log_file_path: str):
        """Initialize the singleton goalkeeper logger."""
        if cls._instance is None:
            cls._instance = cls(log_file_path)
        return cls._instance

    def log(self, message: str):
        """
        Write a message to the goalkeeper log file only (not to console).

        Args:
            message: Message to log
        """
        if self.log_file:
            self.log_file.write(message + '\n')
            self.log_file.flush()

    def close(self):
        """Close the goalkeeper log file."""
        if self.log_file:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            footer = f"""
{'='*80}
Goalkeeper debugging completed: {timestamp}
{'='*80}
"""
            self.log_file.write(footer)
            self.log_file.flush()
            self.log_file.close()
            self.log_file = None
            print(f"✓ Goalkeeper debug log saved to: {self.log_file_path}")

    def __del__(self):
        """Cleanup when object is destroyed."""
        self.close()


class DualLogger:
    """
    Captures console output and writes to both console and file.

    Usage:
        logger = DualLogger("output_videos/sample_1_analysis.log")
        logger.start()
        print("This goes to both console and file")
        logger.stop()
    """

    def __init__(self, log_file_path: str):
        """
        Initialize dual logger.

        Args:
            log_file_path: Path where log file will be saved
        """
        self.log_file_path = Path(log_file_path)
        self.log_file = None
        self.original_stdout = None
        self.original_stderr = None

        # Create directory if needed
        self.log_file_path.parent.mkdir(parents=True, exist_ok=True)

    def start(self):
        """Start capturing output to file."""
        # Save original streams
        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr

        # Open log file
        self.log_file = open(self.log_file_path, 'w', encoding='utf-8', buffering=1)  # Line buffered

        # Write header
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        header = f"""
{'='*80}
Soccer Analysis Pipeline - Execution Log
Started: {timestamp}
Log File: {self.log_file_path}
{'='*80}

"""
        self.log_file.write(header)
        self.log_file.flush()

        # Redirect stdout and stderr
        sys.stdout = _StreamTee(self.original_stdout, self.log_file)
        sys.stderr = _StreamTee(self.original_stderr, self.log_file)

        print(f"📝 Logging output to: {self.log_file_path}")

    def stop(self):
        """Stop capturing and restore original streams."""
        if self.log_file:
            # Write footer
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            footer = f"""
{'='*80}
Execution completed: {timestamp}
{'='*80}
"""
            self.log_file.write(footer)
            self.log_file.flush()

            # Restore original streams
            sys.stdout = self.original_stdout
            sys.stderr = self.original_stderr

            # Close log file
            self.log_file.close()
            self.log_file = None

            print(f"✓ Log saved to: {self.log_file_path}")

    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()


class _StreamTee:
    """
    Internal class to write to multiple streams simultaneously.
    Like Unix 'tee' command.
    """

    def __init__(self, *streams):
        """Initialize with multiple output streams."""
        self.streams = streams

    def write(self, data):
        """Write to all streams."""
        for stream in self.streams:
            stream.write(data)
            stream.flush()

    def flush(self):
        """Flush all streams."""
        for stream in self.streams:
            stream.flush()

    def isatty(self):
        """Check if any stream is a terminal."""
        return any(hasattr(s, 'isatty') and s.isatty() for s in self.streams)


def create_log_path(video_path: str, suffix: str = "_analysis") -> Path:
    """
    Create log file path next to output video.

    Args:
        video_path: Path to input or output video
        suffix: Suffix for output (same as video output suffix)

    Returns:
        Path object for log file

    Example:
        >>> create_log_path("input_videos/sample_1.mp4", "_complete_analysis")
        Path("output_videos/sample_1_complete_analysis.log")
    """
    video_path = Path(video_path)
    video_name = video_path.stem

    # Log goes to output_videos directory
    output_dir = Path("output_videos")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create log filename
    log_filename = f"{video_name}{suffix}.log"
    log_path = output_dir / log_filename

    return log_path


def create_goalkeeper_log_path(video_path: str, suffix: str = "_analysis") -> Path:
    """
    Create goalkeeper debug log file path next to output video.

    Args:
        video_path: Path to input or output video
        suffix: Suffix for output (same as video output suffix)

    Returns:
        Path object for goalkeeper log file

    Example:
        >>> create_goalkeeper_log_path("input_videos/sample_1.mp4", "_complete_analysis")
        Path("output_videos/sample_1_complete_analysis_goalkeeper_debug.log")
    """
    video_path = Path(video_path)
    video_name = video_path.stem

    # Log goes to output_videos directory
    output_dir = Path("output_videos")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create log filename with goalkeeper debug suffix
    log_filename = f"{video_name}{suffix}_goalkeeper_debug.log"
    log_path = output_dir / log_filename

    return log_path
