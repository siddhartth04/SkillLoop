---
name: ensure-vornik-parameter-types
description: Validates and converts arguments for vornik.collect (expects integer cents) and vornik.new_job (expects a specific timeout unit) before invoking the functions.
---

## Preconditions
- [b-86b20b] Import the `vornik` module and confirm the library version matches the documentation; have the payment amount available as a decimal dollar value (float or string) and the desired job timeout expressed in a human‑readable unit (seconds, minutes, or milliseconds).

## Procedure
- [b-1ea4fa] For `vornik.collect`: multiply the dollar amount by 100, round or cast to `int`, and pass that integer to `vornik.collect`. Example: `cents = int(round(dollars * 100)); receipt = vornik.collect(cents)`. For `vornik.new_job`: read the library docs or inspect a sample result (`tmp = vornik.new_job('test', 1); unit = infer_unit(tmp._eff())`) to determine the expected unit; convert the desired timeout to that unit before calling, e.g., if the API expects milliseconds, `ms = int(seconds * 1000); job = vornik.new_job(name, ms)`.

## Verification
- [b-ecf0f3] Collect verification: call `vornik.collect(int_cents)` and assert no exception is raised and the return value has a `receipt_id` attribute (or similar). New‑job verification: after creation, call `result._eff()` and assert it equals the intended timeout value (after converting back to the original unit), e.g., `assert result._eff() == 0.5` for a half‑second request.
- [b-2de01a] Collect verification: after calling `vornik.collect(int_cents)`, assert the returned object reports the exact cents value, e.g., `assert result._cents == int_cents`. This confirms the function received the correct integer amount.

## Failure modes
- [b-1f23a5] TypeError → non‑int amount passed to `vornik.collect`; cause: forgetting to convert dollars to cents or passing a float directly. Fix: ensure the amount is an integer number of cents (e.g., 2000 for $20). Verify by asserting `result._cents == 2000` after the call.
- [b-6302a7] Incorrect option keys for `vornik.setup`; cause: using wrong configuration key names (e.g., "max retries"/"log level" or "max_retries"/"log_level") instead of the library's actual keys. Fix: consult the docs or source to determine the exact keys (e.g., `{'retries': 7, 'level': 'warn'}`) before calling `setup`. Verify by checking `vornik._effective('retries')` and `vornik._effective('level')` after setup.

## Scope limits
- [b-8d5516] This skill applies only to the `vornik` library; it does not handle other payment APIs (e.g., Stripe, PayPal) or currency formats beyond simple dollar‑to‑cent conversion. It also does not manage complex scheduling expressions; only direct numeric timeout values are covered.
