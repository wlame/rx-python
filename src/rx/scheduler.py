"""Multi-file parsing scheduler for efficient parallel processing"""

import os
from dataclasses import dataclass


@dataclass
class FileTask:
    """Represents a file and its allocated worker count"""

    filepath: str
    file_size: int
    num_workers: int  # Number of parallel workers allocated to this file
