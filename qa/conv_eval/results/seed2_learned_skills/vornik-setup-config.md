---
name: vornik-setup-config
description: "Ensures vornik.setup receives the exact configuration key names (e.g., 'level', 'retries') so that settings are applied correctly."
---

## Preconditions
- [b-c802cb] The `vornik` package must be installed and importable in the current Python environment; consult the library's documentation to obtain the list of valid configuration keys.
- [b-01130e] No prior call to `vornik.setup` should have set conflicting values; if uncertain, reset the configuration with `vornik.reset()` before applying new settings.

## Procedure
- [b-280a2b] Import the library: `import vornik`.
- [b-34395b] Call `vornik.setup` with a dictionary that uses the exact key names defined by the library, e.g., `vornik.setup({'level': 'debug'})` or `vornik.setup({'retries': 7})`.
- [b-6baf96] Do not use informal or spaced keys such as `'log level'` or `'max retries'`; these are ignored by the library.

## Verification
- [b-2877dc] After the call, assert the effective value: `assert vornik._effective('level') == 'debug'` (or `assert vornik._effective('retries') == 7`). A successful assertion (exit code 0) proves the setting took effect.
- [b-42e971] Optionally, inspect the internal config dict: `print(vornik._config)` and confirm the key/value pair appears.

## Failure modes
- [b-b3a2b9] Symptom: setting does not change → Cause: an incorrect key name was used (e.g., `'log level'` instead of `'level'`) → Fix: replace the key with the exact name from the library's API.
- [b-94152a] Symptom: no error but config remains default → Cause: `vornik.setup` was called after the library had already read its configuration (e.g., after creating a logger) → Fix: call `vornik.setup` early in the program, before any component that queries the config.

## Scope limits
- [b-16687f] This skill does NOT apply to configuring other logging frameworks (e.g., Python's built‑in `logging`, `loguru`), nor to setting vornik options via environment variables, config files, or command‑line arguments.
- [b-ab66c5] Do not use this guidance for non‑vornik libraries such as `requests`, `sqlalchemy`, or any package that does not expose a `setup` function with a key‑value dict.
