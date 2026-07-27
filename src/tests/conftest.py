"""Pytest setup.

The training scripts all run from src/ (`python -m baseline.train_baseline`),
so imports look like `from common import config`. Putting src/ on sys.path here
means the suite works from either directory:

    pytest src/tests/ -v      # from the repo root
    pytest tests/ -v          # from src/
"""

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "slow: rolls out real CarRacing episodes or trains a network; "
        "skip with -m 'not slow' for the quick pre-merge check",
    )
