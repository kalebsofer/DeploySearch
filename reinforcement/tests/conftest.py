import pytest
import sys
from pathlib import Path

# Add the app directory to PYTHONPATH
root_dir = Path(__file__).parent.parent
sys.path.append(str(root_dir))
