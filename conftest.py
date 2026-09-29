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


# ---------------------------------------------------------------------------------------------------
# Windows: close SQLite connections before pytest deletes the temp directories the tests created.
#
# Tests build a SkillLoop in a tempfile.mkdtemp() and never close it, which is fine on Linux: unlinking
# a file another handle still has open is legal there, so cleanup succeeds and CI is green. Windows
# refuses (WinError 32), so the same suite reported dozens of teardown errors on a Windows machine even
# though every test body had passed. That made the project look broken to anyone developing on Windows.
#
# Closing the connection after each test fixes it without touching production code. Store.close() is
# already part of the API (Store is a context manager); the tests simply never called it.
# ---------------------------------------------------------------------------------------------------
import gc

import pytest


@pytest.fixture(autouse=True)
def _close_sqlite_connections():
    """Close every SkillLoop store opened during a test, so temp-dir cleanup can delete the database."""
    yield
    try:
        from skillloop.store import Store
    except Exception:                      # pragma: no cover - skillloop not importable yet
        return
    for obj in gc.get_objects():
        try:
            if isinstance(obj, Store):
                obj.close()
        except Exception:                  # a half-built store must not fail the test that just passed
            pass


def _cleanup(home, scope=None):
    """Close any SkillLoop/Store the test opened, then delete its temp directory.

    Windows will not unlink an open SQLite file (WinError 32), so a test that passed could still fail in
    cleanup. Closing first fixes that, and ignore_errors keeps a cleanup problem from being reported as a
    test failure on any platform.
    """
    import gc
    import shutil
    try:
        from skillloop.store import Store
        for obj in gc.get_objects():
            try:
                if isinstance(obj, Store):
                    obj.close()
            except Exception:
                pass
    except Exception:
        pass
    shutil.rmtree(home, ignore_errors=True)
