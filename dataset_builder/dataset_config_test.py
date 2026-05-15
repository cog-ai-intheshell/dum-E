"""Compatibility wrapper for basketball_sim.dataset.config_test."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from basketball_sim.dataset.config_test import *  # noqa: F401,F403
from basketball_sim.dataset.config_test import print_fields


if __name__ == "__main__":
    print_fields()
