import os
import shutil
import tempfile
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def temporary_directory():
    """Context manager for creating and cleaning up a temporary directory."""
    temp_dir_base = os.environ.get("TMPDIR", "/tmp")
    if not os.path.exists(temp_dir_base):
        temp_dir_base = os.getcwd()
    temp_dir = tempfile.mkdtemp(dir=temp_dir_base)
    try:
        yield Path(temp_dir)
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
