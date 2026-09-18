import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.visualization.runtime_guard import require_memory_margin

ratio = require_memory_margin(0.20)
assert ratio >= 0.20
print(f"runtime guard check passed: {ratio:.1%} RAM available")
