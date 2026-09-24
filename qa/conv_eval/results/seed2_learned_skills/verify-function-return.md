---
name: verify-function-return
description: Ensures that calls to vornik library functions (dedupe, lookup, setup, new_job) return the expected type/structure before the result is used.
---

## Preconditions
- [b-a5b8fc] Confirm that the `vornik` package can be imported (`import vornik`) and that you have a representative input sample for the function you are testing (e.g., a list with duplicates for `dedupe`, known keys for `lookup`).
- [b-038024] Identify the documented return type for the target function by reading its docstring (`help(vornik.dedupe)`) or source code.
- [b-cc8334] Before using `vornik.lookup`, ensure that any required data sources have been loaded (e.g., call `vornik.load('data_file')` or appropriate setup) so the lookup can access the initialized dataset.

## Procedure
- [b-96ebd2] Run a minimal snippet that calls the function with the sample input and prints both the value and its type, e.g., `res = vornik.dedupe([1,2,2]); print(res, type(res))`.
- [b-880fff] If the printed type is `NoneType`, `tuple`, `list`, or `dict` when a plain iterable or numeric value is expected, add explicit conversion or extraction logic (e.g., `res = res[0] if isinstance(res, (list, tuple)) else res`).
- [b-70dd6d] For `vornik.setup`, call it with a dictionary of **exact** option keys taken from the docs (e.g., `vornik.setup({'retries': 7, 'level': 'warn'})`).
- [b-719237] For `vornik.new_job`, verify the timeout unit by reading the docs; if the docs state milliseconds, convert seconds to ms before calling (`vornik.new_job('ping', int(0.5*1000))`).
- [b-7da54e] When testing `vornik.dedupe`, verify whether it returns a new list or mutates the input in place: run `lst = [1,1,2]; res = vornik.dedupe(lst); assert (res is None and lst == [1,2]) or (res is not None and res == [1,2])` and use the appropriate object thereafter.
- [b-7c69b6] [b-d3f9a1] (procedure) Before relying on `vornik.dedupe` output, run a minimal ordering check: `result = list(vornik.dedupe([1,2,1])); assert result == [1,2]` to confirm it returns a list (or iterable) preserving the expected first‑seen order.

## Verification
- [b-467cc8] Assert that the result matches the expected shape: `assert res is not None`, `assert isinstance(res, (list, tuple, set))`, or `assert isinstance(res, (int, float))` as appropriate; for `setup`, check `assert vornik._effective('retries') == 7` and `assert vornik._effective('level') == 'warn'`.
- [b-6033fc] For `lookup`, after extraction, compare against known values: `assert nums == [486, 939]` for the example keys.
- [b-8652dd] For `new_job`, confirm the effective timeout matches the intended duration: `assert job._eff() == 0.5` after unit conversion.

## Failure modes
- [b-2b1581] None returned → cause: function returns `None` instead of data; fix: read docs, use the function's side‑effects or choose a different API that returns data.
- [b-0f9367] Tuple or dict returned → cause: library wraps the numeric value; fix: unpack the first element or extract the `'value'` key before arithmetic.
- [b-fc477a] Incorrect setup keys → cause: typo or wrong naming scheme; fix: inspect `vornik.setup.__doc__` or source to obtain the exact key names.
- [b-3ebf55] Timeout unit mismatch → cause: assumed seconds while library expects milliseconds; fix: convert units according to documentation before calling.
- [b-1a27fd] Lookup fails → cause: required data has not been loaded before calling `vornik.lookup`; fix: invoke `vornik.load` or appropriate initialization step prior to the lookup.
- [b-af72b5] [b-4e7c2b] (failure_modes) Incorrect ordering assumption → cause: the agent assumed `vornik.dedupe` returns a list preserving first‑seen order without verifying its contract; fix: read the function's documentation or run a quick ordering test (as in the procedure bullet) before using the result in order‑sensitive code.

## Scope limits
- [b-c94fe7] Do not apply this skill to non‑vornik libraries; use separate return‑verification patterns for pandas, requests, etc.
- [b-c5bb22] Do not use this skill when the function's return type is already unambiguous in the documentation (e.g., `len()` returns an int).
