---
name: vornik-configure
description: "Configures the vornik library by using the exact configuration key names (e.g., \"retries\") and by supplying timeout values in the unit required by vornik.new_job (typically milliseconds)."
---

## Preconditions
- [b-6b809c] Ensure the vornik package is installed and importable (`import vornik`). Verify the library version matches the documentation you will consult.
- [b-4b9f8d] Have access to the vornik API reference (online docs or local source) to confirm the exact configuration key names and the unit of the `timeout` argument for `new_job`.

## Procedure
- [b-be27d2] When calling `vornik.setup`, first verify the exact option key names by consulting the documentation or inspecting the source code, then construct the configuration dictionary using those exact names (e.g., `{"retries": <int>, "level": "warn"}`). Do not guess informal names like "max retries" or "log level".
- [b-5e8e11] Before invoking `vornik.new_job(name, timeout)`, read the docs to determine the expected unit (seconds, milliseconds, minutes, etc.). Convert your desired duration to that unit before passing it. For example, if the API expects seconds and you have minutes, use `seconds = minutes * 60`; if it expects milliseconds and you have seconds, use `ms = seconds * 1000`.
- [b-303dc0] If the unit is ambiguous, perform a quick probe: `job = vornik.new_job('probe', 1)` and inspect `job._eff()`; compare the returned value to the input to infer the unit, then apply the appropriate conversion for real jobs.

## Verification
- [b-0ba194] After `vornik.setup`, run `assert vornik._effective('retries') == <expected>`; a raised AssertionError means the key name was wrong.
- [b-3e22f4] After creating a job, run `assert job._eff() == <expected_effective_timeout>`; if the assertion passes, the timeout unit conversion was correct.
- [b-c8bf0d] After `vornik.setup`, run `assert vornik._effective('level') == '<expected>'` (e.g., `'warn'`). This confirms that the logging level key was applied correctly.

## Failure modes
- [b-57d59a] Incorrect key name → library silently ignores the entry → `vornik._effective('<key>')` returns default or raises KeyError → fix by verifying exact key names from docs or source before calling `setup` and assert the effective values after setup.
- [b-caeba8] Wrong timeout unit → job runs with a timeout that is off by the conversion factor (e.g., minutes vs seconds, seconds vs milliseconds) → observed mismatch in `job._eff()` → fix by converting the supplied time to the documented unit before calling `new_job`.
- [b-95986d] Using an outdated vornik version where the key name or timeout unit differs → verification asserts fail → upgrade/downgrade library or adjust code to match that version's API.

## Scope limits
- [b-41fe5a] This skill applies only to the vornik Python library; do not use it for unrelated libraries such as requests, celery, httpx, or for generic config files (JSON, YAML, INI).
- [b-d30adf] Do not apply this skill when configuring environment variables or command‑line flags; those require separate handling.
