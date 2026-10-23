"""
scripts/run_tests.py
Automated test runner that executes pytest and prints a clean academic report.
"""

import sys
import pytest

if __name__ == "__main__":
    print("=" * 70)
    print("   HACKATHON INSIGHTS -- AUTOMATED TEST SUITE EXECUTION")
    print("=" * 70)
    
    exit_code = pytest.main(["-v", "tests/"])
    
    if exit_code == 0:
        print("\n" + "=" * 70)
        print("   [SUCCESS] ALL 25 AUTOMATED TESTS EXECUTED AND PASSED SUCCESSFULLY!")
        print("=" * 70)
    else:
        print(f"\n[FAILURE] Test suite completed with non-zero exit code: {exit_code}")
    
    sys.exit(exit_code)
