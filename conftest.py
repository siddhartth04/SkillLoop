"""Make the repository root importable for the test suite.

`tests/test_conv_eval.py` imports the evaluation harness as `qa.conv_eval.*`. The `qa`
package is deliberately not shipped by pyproject (`packages.find` excludes it), so it is
importable only when the repository root is on `sys.path`. Pytest guarantees that for the
directory holding the rootdir conftest, which makes the import work the same way locally
and in CI, on every supported Python version.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
