---
name: verify-persistence
description: Ensures that a call to `vornik.Vault().save()` actually writes data to persistent storage and that the data can be retrieved after a fresh instance is created.
---

## Preconditions
- [b-7770dd] A `vornik.Vault` instance must be instantiated and the intended storage directory must exist with write permissions for the current user.
- [b-7f2853] Identify the storage backend expected by the Vault (e.g., a JSON file path, SQLite DB, or custom backend) and have its location available.
- [b-b24a5d] When persisting data with `vornik.Vault`, an explicit storage path must be supplied (e.g., via the `path` argument to the constructor or `save`) and the same path must be used for subsequent `load` calls; relying on the default location can lead to mismatched instances.

## Procedure
- [b-986817] Call `vault.save(path='<storage_path>', confirm=True)` (or the backend‑specific flag) instead of relying on default parameters; this forces the Vault to serialize its in‑memory state to the chosen persistent location.
- [b-2734e0] If the Vault API provides a `flush()` or `commit()` method, invoke it immediately after `save()` to guarantee that OS buffers are flushed to disk.
- [b-34abf8] When invoking `vault.save()`, do not include `confirm=False`; the flag prevents the Vault from persisting data. Use the default call or explicitly set `confirm=True`.
- [b-2e3b04] Invoke `vault.save(path='<explicit_path>', confirm=...)` (or construct the Vault with `path='<explicit_path>'`) and later load with `vornik.Vault.load(path='<same_explicit_path>')`; ensure the path strings are identical for both operations.

## Verification
- [b-5d177f] After the save, instantiate a new `vornik.Vault()` (or reload the backend) and assert that the previously stored key/value is present, e.g., `assert new_vault.get('user') == 345`.
- [b-70ba54] Confirm that the underlying storage file exists (`test -f <storage_path>`) and contains the expected key by grepping or parsing the file (`grep '"user": 345' <storage_path>`).
- [b-8ccfca] After saving, create a fresh `vornik.Vault` (or use `load`) with the *exact same* explicit path and assert that the expected keys/values are present, confirming that the path consistency preserved the data.

## Failure modes
- [b-fae6e4] symptom: key missing after reload → cause: `vault.save()` was called with `confirm=False`, which disables writing to persistent storage → fix: call `vault.save()` without the `confirm=False` flag (use default behavior or set `confirm=True`), then re‑run the verification steps.
- [b-41beb9] symptom: `vault.save()` returns no error but file is empty → cause: write permission denied on the target directory → fix: adjust directory permissions (`chmod u+w <dir>`), or choose a writable location.
- [b-531d2f] symptom: no data written despite successful `save()` call → cause: passing `confirm=False` to `vault.save()` disables persistence → fix: omit the `confirm` argument or set it to `True` to ensure data is written to disk.
- [b-e8543a] symptom: data missing after reload → cause: `vault.save()` (or the Vault constructor) was called without an explicit file path, so the default storage location differed between the saving and loading instances → fix: always provide the same explicit `path` argument to both `save` (or the constructor) and `load`, or reuse the same Vault instance for the entire operation.

## Scope limits
- [b-242c26] This skill applies only to the `vornik.Vault` persistence API; it does not cover generic key‑value stores, databases (e.g., SQLite, Redis), or configuration file writes.
- [b-5e0035] Do not use this skill when the Vault is configured for transient, in‑memory only operation (e.g., `backend='memory'`).
