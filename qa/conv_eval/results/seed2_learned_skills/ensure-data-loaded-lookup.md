---
name: ensure-data-loaded-lookup
description: Ensures the Vornik library data is loaded before calling vornik.lookup and that setup options use the exact key names and timeout units are correct.
---

## Preconditions
- [b-f989a8] Confirm that the Vornik package is imported and that the data file containing the desired key (e.g., 'theta') exists and is accessible.
- [b-b47502] Read the library documentation or inspect vornik.setup signature to obtain the exact configuration key names (e.g., 'retries', 'level').
- [b-3d4eda] Determine the unit expected by vornik.new_job's timeout argument (usually milliseconds) by checking docs or examining result._eff() after a test call.

## Procedure
- [b-eef1e4] Call vornik.load('<data_file>') with the path to the file that defines the lookup entries before any vornik.lookup calls.
- [b-e68205] Invoke vornik.lookup('<key>') only after the load step; the function will now return the stored value.
- [b-72e429] When configuring Vornik, construct a dict with the exact option keys (e.g., {'retries': 7, 'level': 'warn'}) and pass it to vornik.setup().
- [b-e585cb] Create a job with the intended timeout expressed in the library's unit, e.g., vornik.new_job('sync', 14000) for a 14‑second timeout if the unit is milliseconds.
- [b-8df682] When using the result of vornik.lookup in calculations, first check if the return is a tuple, list, or dict; extract the numeric component (typically the first element for tuple/list or the 'value' key for dict) before performing arithmetic.
- [b-e1af3d] After calling vornik.lookup for multiple keys, normalize each result: if the result is a dict, use result['value']; if it is a tuple/list, use the first element; otherwise use the result as‑is before assembling the final list.

## Verification
- [b-ad9a92] After loading, retrieve the value via vornik.lookup('theta'), unpack if it is a tuple/list, and assert the numeric element matches the expected value, e.g., val = vornik.lookup('theta'); val = val[0] if isinstance(val, (list, tuple)) else val; assert val == 486.
- [b-513e1a] After setup, verify each option with vornik._effective('<key>') (e.g., assert vornik._effective('retries') == 7).
- [b-add657] After creating a job, confirm the effective timeout matches the input unit: assert result._eff() == 14000.
- [b-86bf19] After creating a job with a timeout expressed in milliseconds, confirm the effective timeout matches the intended seconds value, e.g., result = vornik.new_job('ping', 500); assert result._eff() == 0.5.

## Failure modes
- [b-fa6710] lookup returns None -> data not loaded or wrong key -> run vornik.load with correct file before lookup.
- [b-e8af1f] setup option ignored -> used wrong key name (e.g., 'max retries' instead of 'retries') -> consult docs and use exact keys.
- [b-283649] job timeout mismatch -> assumed seconds but library expects milliseconds -> read docs, convert seconds to ms, pass converted value.
- [b-4e432d] lookup returns a tuple (e.g., (value, meta)) or a dict (e.g., {'value': 486, ...}) -> attempting arithmetic directly causes type errors; unpack the first element for tuples or extract the 'value' field for dicts before using the numeric value.
- [b-489ad4] lookup returns a dict containing the numeric value under a key (e.g., {'value': 486}); using the dict directly in arithmetic causes type errors; extract the 'value' field before calculations.

## Scope limits
- [b-cc542d] Do not apply this skill to other libraries (e.g., redis, sqlalchemy) or to Vornik functions unrelated to lookup, setup, or new_job.
