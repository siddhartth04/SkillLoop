---
name: verify-timeout-units
description: "Ensures the timeout argument passed to Vornik's `new_job` is in the correct unit (seconds vs milliseconds) before creating a job."
---

## Preconditions
- [b-82d852] Vornik library is installed and importable; the `new_job` function accepts a `timeout` parameter; the intended timeout value (e.g., 14 seconds) is known; the returned job object provides an `_eff()` method to read the effective timeout.
- [b-a472cb] (preconditions) Access to Vornik documentation is required; first consult the docs to determine the unit expected by `new_job` for the `timeout` parameter. If the documentation is unavailable or ambiguous, fall back to a quick probe (create a dummy job) to discover the unit.

## Procedure
- [b-88d2f5] 1. Import Vornik and create a minimal probe job: `probe = vornik.new_job('sync', 1)`.
2. Inspect the effective timeout: `probe_eff = probe._eff()`.
3. Compare `probe_eff` to the supplied value (1). If `probe_eff == 1`, the unit is seconds; if `probe_eff == 1000`, the unit is milliseconds; adjust accordingly.
4. Convert the intended timeout to the discovered unit (e.g., `timeout_ms = intended_seconds * 1000` for ms).
5. Call `new_job` with the converted value: `job = vornik.new_job('sync', timeout_converted)`.
- [b-ce2d58] If documentation explicitly states the unit, skip the probe step and convert the intended timeout to that unit directly.

## Verification
- [b-edcefc] (verification) After creation, assert that the job's effective timeout matches the intended duration, regardless of the unit used in the call. For example, if the library expects milliseconds, call `job = vornik.new_job('sync', intended_seconds * 1000)` and then assert `job._eff() == intended_seconds`. Conversely, if the library expects seconds, call `job = vornik.new_job('sync', intended_seconds)` and assert `job._eff() == intended_seconds`. This ensures the effective timeout is validated in the original time unit.
- [b-56ac51] Check the process exit code of the verification script; a zero exit code confirms the assertion passed.
- [b-531af5] (verification) Example for a half‑second timeout: if the library uses milliseconds, execute `result = vornik.new_job('ping', 500)` and assert `result._eff() == 0.5`; if it uses seconds, execute `result = vornik.new_job('ping', 0.5)` and assert `result._eff() == 0.5`.

## Failure modes
- [b-d55434] symptom: effective timeout is 1000x smaller than expected → cause: supplied value was interpreted as seconds while library expects milliseconds → fix: multiply the intended seconds by 1000 before passing to `new_job`.
- [b-cd735b] symptom: `AttributeError: 'Job' object has no attribute '_eff'` → cause: using a Vornik version where the method name changed → fix: consult the version's docs for the correct attribute (e.g., `timeout` property) and adjust verification accordingly.
- [b-138156] (failure_modes) symptom: effective timeout appears as a fraction (e.g., 0.5) when a whole number was supplied → cause: the timeout argument was provided in seconds while the library expects milliseconds, or vice‑versa → fix: consult the Vornik documentation to confirm the required unit, then convert the desired duration to that unit before calling `new_job`.

## Scope limits
- [b-151b1c] Do not apply this skill when using Vornik functions other than `new_job` (e.g., `schedule_job`, `retry_job`).
- [b-b29ff2] Do not use for libraries other than Vornik; other job queues have their own timeout semantics.
- [b-6fc3f9] If the job type does not accept a `timeout` argument, this skill is irrelevant.
