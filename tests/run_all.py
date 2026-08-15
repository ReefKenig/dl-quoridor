"""
Run all tests.
Usage: pytest tests/ -v
       OR: python tests/run_all.py
"""

import subprocess
import sys
from pathlib import Path


TESTS = [
    "tests/test_mcts.py",
    "tests/test_mcts_perf.py",
    "tests/test_evaluator.py",
    "tests/test_tensor_spec.py",
    "tests/test_checkpoint.py",
    "tests/test_network.py",
    "tests/smoke_test.py",
]


def main():
    project_root = Path(__file__).parent.parent
    failed = []

    for test in TESTS:
        print(f"\n{'#' * 60}")
        print(f"# Running: {test}")
        print(f"{'#' * 60}")
        result = subprocess.run(
            [sys.executable, "-m", "pytest", test, "-v"],
            cwd=str(project_root),
        )
        if result.returncode != 0:
            failed.append(test)

    print(f"\n{'=' * 60}")
    if failed:
        print(f"FAILED: {', '.join(failed)}")
        sys.exit(1)
    else:
        print(f"ALL {len(TESTS)} TEST SUITES PASSED")
        sys.exit(0)


if __name__ == "__main__":
    main()
