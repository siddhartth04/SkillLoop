"""Seeded generator for the conventions eval.

A small Python library whose API follows conventions the model cannot know from pretraining: it does not exist
anywhere, and its names, data and which conventions are active all come from a random seed. The public API
reference is neutral and truthful but incomplete, like most real internal docs. Nine conventions exist in a fixed
pool of generic API-design quirks; each seed activates K of them and randomizes every name.

Each active convention gets 3 TRAIN tasks and 3 TEST tasks: same trap, different surface. Inactive conventions
contribute 1 NO-TRAP test task each (the function behaves normally), which measures harm from over-applied skills.

Every task carries a hidden checker, plus two reference programs used only to validate the harness:
  naive  - what a competent programmer would write from the docs alone
  quirk  - the correct program under the active conventions
"""
from __future__ import annotations

import random
import re
from dataclasses import dataclass, field

POOL = ("index", "range", "units", "keys", "commit", "cents", "tuple", "desc", "inplace")

NAMES = {
    "at": ["at", "nth", "pick", "item_at"],
    "span": ["span", "slice_of", "between", "window"],
    "make_job": ["make_job", "new_job", "schedule", "create_task"],
    "configure": ["configure", "set_options", "setup"],
    "get_config": ["get_config", "current_options", "read_config"],
    "Store": ["Store", "Vault", "Ledger"],
    "commit": ["commit", "save", "flush"],
    "charge": ["charge", "bill", "collect"],
    "lookup": ["lookup", "fetch", "find"],
    "rank": ["rank", "order", "arrange"],
    "dedupe": ["dedupe", "unique", "distinct"],
}
SYLL = ["vor", "nik", "tal", "bru", "zem", "qua", "lio", "rex", "dun", "pax", "mor", "sil"]
ORD = {2: "2nd", 3: "3rd", 4: "4th", 5: "5th", 6: "6th"}


@dataclass
class Task:
    id: str
    quirk: str
    split: str            # train | test
    trap: bool            # False for no-trap tasks (convention inactive)
    prompt: str
    check: str            # python expression, True when correct; evaluated after the agent's code
    naive: str
    quirk_code: str

    @property
    def reference(self) -> str:
        return self.quirk_code if self.trap else self.naive


@dataclass
class Suite:
    seed: int
    pkg: str
    quirks: list[str]
    names: dict[str, str]
    library: str
    api_doc: str
    hints: dict[str, str]
    tasks: list[Task] = field(default_factory=list)

    def split(self, name: str) -> list[Task]:
        return [t for t in self.tasks if t.split == name]


def _lib(pkg: str, n: dict, quirks: list[str], db: dict) -> str:
    return f'''"""{pkg}: internal utilities."""
QUIRKS = {sorted(quirks)!r}
_DB = {db!r}
_DISK = {{}}
_CONFIG = ({{"MAX-RETRIES": 0, "LOG-LEVEL": "info"}} if "keys" in QUIRKS
           else {{"max_retries": 0, "log_level": "info"}})


def {n["at"]}(seq, pos):
    if "index" in QUIRKS:
        if pos < 1 or pos > len(seq):
            raise IndexError(f"position {{pos}} out of range")
        return seq[pos - 1]
    return seq[pos]


def {n["span"]}(seq, a, b):
    if "range" in QUIRKS:
        return list(seq[a + 1:b + 1])
    return list(seq[a:b])


class Job:
    def __init__(self, name, timeout):
        self.name = name
        self._raw = timeout

    def _eff(self):
        return self._raw / 1000 if "units" in QUIRKS else self._raw

    def __repr__(self):
        return f"Job({{self.name!r}})"


def {n["make_job"]}(name, timeout):
    return Job(name, timeout)


def {n["configure"]}(options):
    for k, v in dict(options).items():
        if k in _CONFIG:
            _CONFIG[k] = v


def {n["get_config"]}():
    return dict(_CONFIG)


def _effective(which):
    if which == "retries":
        return _CONFIG.get("MAX-RETRIES", _CONFIG.get("max_retries"))
    return _CONFIG.get("LOG-LEVEL", _CONFIG.get("log_level"))


class {n["Store"]}:
    def __init__(self):
        self._buf = {{}}

    def put(self, key, value):
        self._buf[key] = value

    def {n["commit"]}(self, confirm=False):
        if "commit" in QUIRKS and not confirm:
            return None
        _DISK.update(self._buf)
        self._buf = {{}}

    @staticmethod
    def load():
        return dict(_DISK)


class Receipt:
    def __init__(self, cents):
        self._cents = cents

    def __repr__(self):
        return "Receipt()"


def {n["charge"]}(amount):
    if "cents" in QUIRKS:
        if not isinstance(amount, int) or isinstance(amount, bool):
            raise TypeError("amount must be an int")
        return Receipt(amount)
    return Receipt(round(amount * 100))


def {n["lookup"]}(key):
    if "tuple" in QUIRKS:
        return (None, _DB[key]) if key in _DB else (KeyError(key), None)
    return _DB[key]


def {n["rank"]}(items):
    return sorted(items, reverse=("desc" in QUIRKS))


def {n["dedupe"]}(items):
    seen = set()
    out = [x for x in items if not (x in seen or seen.add(x))]
    if "inplace" in QUIRKS:
        items[:] = out
        return None
    return out
'''


def _doc(pkg: str, n: dict, db: dict) -> str:
    keys = ", ".join(repr(k) for k in db)
    return f"""API reference for `{pkg}` (import it with `import {pkg}`):

- {n["at"]}(seq, pos) -> the item of seq at position pos
- {n["span"]}(seq, a, b) -> list of the items of seq between a and b
- {n["make_job"]}(name, timeout) -> a Job with the given name and timeout
- {n["configure"]}(options: dict) -> None. Supported options: max retries (int), log level (str)
- {n["get_config"]}() -> dict of the current options
- {n["Store"]}() -> a key-value store with .put(key, value), .{n["commit"]}(confirm=False) and the static method {n["Store"]}.load() -> dict of persisted data
- {n["charge"]}(amount) -> a Receipt for charging the given amount of money
- {n["lookup"]}(key) -> the value stored under key. Known keys: {keys}
- {n["rank"]}(items) -> the items in sorted order
- {n["dedupe"]}(items) -> the items with duplicates removed, keeping first occurrences
"""


def _hints(n: dict) -> dict[str, str]:
    return {
        "index": f"{n['at']}() positions are 1-based: position 1 is the first item; position 0 raises IndexError.",
        "range": f"{n['span']}(seq, a, b) excludes index a and includes index b: it returns seq[a+1:b+1]. For python-style seq[x:y] call {n['span']}(seq, x-1, y-1).",
        "units": f"{n['make_job']}() timeouts are in milliseconds: 30 seconds is 30000.",
        "keys": "Option keys are UPPER-KEBAB-CASE: 'MAX-RETRIES' and 'LOG-LEVEL'. Any other key is silently ignored.",
        "commit": f"{n['Store']}.{n['commit']}() does nothing unless called as .{n['commit']}(confirm=True).",
        "cents": f"{n['charge']}() takes an int number of cents: $12.50 is 1250.",
        "tuple": f"{n['lookup']}() returns a tuple (error, value); the value is the second element.",
        "desc": f"{n['rank']}() sorts in DESCENDING order: the largest item comes first.",
        "inplace": f"{n['dedupe']}() modifies the list in place and returns None: call it, then use the list itself.",
    }


def _tasks(rng: random.Random, n: dict, db: dict, pkg: str) -> dict[str, list[tuple]]:
    """quirk -> list of 6 (prompt, check, naive, quirk_code); first 3 train, last 3 test."""
    def xs(k=7):
        return rng.sample(range(10, 99), k)

    def dup():
        base = rng.sample(range(1, 30), 5)
        return base + [base[1], base[3], base[0]]

    p = pkg
    out: dict[str, list[tuple]] = {}

    a, b, c, d, e, f = xs(), xs(), xs(), xs(), xs(), xs()
    i = rng.randint(2, 5)
    k = rng.randint(3, 5)
    at = n["at"]
    out["index"] = [
        (f"Use `{at}` to get the first element of `data = {a}`.", f"result == {a[0]}",
         f"result = {p}.{at}(data, 0)", f"result = {p}.{at}(data, 1)"),
        (f"Use `{at}` to get the element at zero-based index {i} of `data = {b}`.", f"result == {b[i]}",
         f"result = {p}.{at}(data, {i})", f"result = {p}.{at}(data, {i + 1})"),
        (f"Use `{at}` to build a list of the first three elements of `data = {c}`.", f"result == {c[:3]}",
         f"result = [{p}.{at}(data, j) for j in range(3)]", f"result = [{p}.{at}(data, j) for j in range(1, 4)]"),
        (f"Use `{at}` to get the {ORD[k]} element of `data = {d}` (the 1st element is the first one in the list).",
         f"result == {d[k - 1]}", f"result = {p}.{at}(data, {k - 1})", f"result = {p}.{at}(data, {k})"),
        (f"Use `{at}` to get the last element of `data = {e}`.", f"result == {e[-1]}",
         f"result = {p}.{at}(data, len(data) - 1)", f"result = {p}.{at}(data, len(data))"),
        (f"Use `{at}` in a loop to return all elements of `data = {f}` in reverse order, as a list.",
         f"result == {f[::-1]}", f"result = [{p}.{at}(data, j) for j in range(len(data) - 1, -1, -1)]",
         f"result = [{p}.{at}(data, j) for j in range(len(data), 0, -1)]"),
    ]

    a, b, c, d, e, f = xs(), xs(), xs(), xs(), xs(), xs()
    sp = n["span"]
    out["range"] = [
        (f"Use `{sp}` to get the elements at zero-based indices 1, 2 and 3 of `data = {a}`, as a list.",
         f"result == {a[1:4]}", f"result = {p}.{sp}(data, 1, 4)", f"result = {p}.{sp}(data, 0, 3)"),
        (f"Use `{sp}` to get the first two elements of `data = {b}`, as a list.", f"result == {b[:2]}",
         f"result = {p}.{sp}(data, 0, 2)", f"result = {p}.{sp}(data, -1, 1)"),
        (f"Use `{sp}` to get the elements from zero-based index 2 up to but not including index 5 of `data = {c}`.",
         f"result == {c[2:5]}", f"result = {p}.{sp}(data, 2, 5)", f"result = {p}.{sp}(data, 1, 4)"),
        (f"Use `{sp}` to get the elements at zero-based indices 3 and 4 of `data = {d}`, as a list.",
         f"result == {d[3:5]}", f"result = {p}.{sp}(data, 3, 5)", f"result = {p}.{sp}(data, 2, 4)"),
        (f"Use `{sp}` to get every element of `data = {e}` except the first, as a list.", f"result == {e[1:]}",
         f"result = {p}.{sp}(data, 1, len(data))", f"result = {p}.{sp}(data, 0, len(data) - 1)"),
        (f"Use `{sp}` to get the elements from zero-based index 2 through index 5 inclusive of `data = {f}`.",
         f"result == {f[2:6]}", f"result = {p}.{sp}(data, 2, 6)", f"result = {p}.{sp}(data, 1, 5)"),
    ]

    mj = n["make_job"]
    s1, s2 = rng.randint(10, 90), rng.randint(20, 60)
    out["units"] = [
        (f"Use `{mj}` to create a job named 'sync' with a timeout of {s1} seconds. Set `result` to the job.",
         f"result._eff() == {s1}", f"result = {p}.{mj}('sync', {s1})", f"result = {p}.{mj}('sync', {s1 * 1000})"),
        (f"Use `{mj}` to create a job named 'report' with a timeout of 2 minutes. Set `result` to the job.",
         "result._eff() == 120", f"result = {p}.{mj}('report', 120)", f"result = {p}.{mj}('report', 120000)"),
        (f"Use `{mj}` to create a job named 'ping' with a timeout of half a second. Set `result` to the job.",
         "result._eff() == 0.5", f"result = {p}.{mj}('ping', 0.5)", f"result = {p}.{mj}('ping', 500)"),
        (f"Use `{mj}` to create a job named 'backup' with a timeout of {s2} seconds. Set `result` to the job.",
         f"result._eff() == {s2}", f"result = {p}.{mj}('backup', {s2})", f"result = {p}.{mj}('backup', {s2 * 1000})"),
        (f"Use `{mj}` to create a job named 'import' with a timeout of 1.5 minutes. Set `result` to the job.",
         "result._eff() == 90", f"result = {p}.{mj}('import', 90)", f"result = {p}.{mj}('import', 90000)"),
        (f"Use `{mj}` to create jobs named 'a', 'b' and 'c' with timeouts of 5, 10 and 20 seconds. "
         f"Set `result` to the list of jobs.", "[j._eff() for j in result] == [5, 10, 20]",
         f"result = [{p}.{mj}(x, t) for x, t in [('a', 5), ('b', 10), ('c', 20)]]",
         f"result = [{p}.{mj}(x, t * 1000) for x, t in [('a', 5), ('b', 10), ('c', 20)]]"),
    ]

    cf, gc = n["configure"], n["get_config"]
    r1, r2, r3, r4 = (rng.randint(2, 9) for _ in range(4))
    out["keys"] = [
        (f"Use `{cf}` to set max retries to {r1}.", f"{p}._effective('retries') == {r1}",
         f"{p}.{cf}({{'max_retries': {r1}}})", f"{p}.{cf}({{'MAX-RETRIES': {r1}}})"),
        (f"Use `{cf}` to set the log level to 'debug'.", f"{p}._effective('level') == 'debug'",
         f"{p}.{cf}({{'log_level': 'debug'}})", f"{p}.{cf}({{'LOG-LEVEL': 'debug'}})"),
        (f"Use `{cf}` to set max retries to {r2} and the log level to 'warn'.",
         f"{p}._effective('retries') == {r2} and {p}._effective('level') == 'warn'",
         f"{p}.{cf}({{'max_retries': {r2}, 'log_level': 'warn'}})",
         f"{p}.{cf}({{'MAX-RETRIES': {r2}, 'LOG-LEVEL': 'warn'}})"),
        (f"Use `{cf}` to set max retries to {r3}.", f"{p}._effective('retries') == {r3}",
         f"{p}.{cf}({{'max_retries': {r3}}})", f"{p}.{cf}({{'MAX-RETRIES': {r3}}})"),
        (f"Use `{cf}` to set the log level to 'error'.", f"{p}._effective('level') == 'error'",
         f"{p}.{cf}({{'log_level': 'error'}})", f"{p}.{cf}({{'LOG-LEVEL': 'error'}})"),
        (f"Use `{cf}` to set the log level to 'trace' and max retries to {r4}.",
         f"{p}._effective('retries') == {r4} and {p}._effective('level') == 'trace'",
         f"{p}.{cf}({{'log_level': 'trace', 'max_retries': {r4}}})",
         f"{p}.{cf}({{'LOG-LEVEL': 'trace', 'MAX-RETRIES': {r4}}})"),
    ]

    st, cm = n["Store"], n["commit"]
    v = [rng.randint(1, 500) for _ in range(6)]

    def store(pairs):
        body = "; ".join(f"s.put({k!r}, {val})" for k, val in pairs)
        chk = " and ".join(f"{p}.{st}.load().get({k!r}) == {val}" for k, val in pairs)
        return (f"s = {p}.{st}(); {body}; s.{cm}()", f"s = {p}.{st}(); {body}; s.{cm}(confirm=True)", chk)

    specs = [[("user", v[0])], [("a", v[1]), ("b", v[2])], [("limit", v[3])],
             [("count", v[4])], [("x", v[5]), ("y", v[0])], [("owner", v[1]), ("total", v[2]), ("size", v[3])]]
    phr = ["Using `{st}`, save the key {ks} so that it persists.",
           "Using `{st}`, save the keys {ks} so that they persist.",
           "Using `{st}`, persist {ks}.", "Using `{st}`, store {ks} permanently.",
           "Using `{st}`, write {ks} so they are persisted.", "Using `{st}`, persist all of {ks}."]
    out["commit"] = []
    for sp_, ph in zip(specs, phr):
        naive, quirk, chk = store(sp_)
        ks = ", ".join(f"'{k}'={val}" for k, val in sp_)
        out["commit"].append((ph.format(st=st, ks=ks), chk, naive, quirk))

    ch = n["charge"]
    out["cents"] = [
        (f"Use `{ch}` to charge $12.50. Set `result` to the receipt.", "result._cents == 1250",
         f"result = {p}.{ch}(12.50)", f"result = {p}.{ch}(1250)"),
        (f"Use `{ch}` to charge $3.99. Set `result` to the receipt.", "result._cents == 399",
         f"result = {p}.{ch}(3.99)", f"result = {p}.{ch}(399)"),
        (f"Use `{ch}` to charge $20. Set `result` to the receipt.", "result._cents == 2000",
         f"result = {p}.{ch}(20)", f"result = {p}.{ch}(2000)"),
        (f"Use `{ch}` to charge $7. Set `result` to the receipt.", "result._cents == 700",
         f"result = {p}.{ch}(7)", f"result = {p}.{ch}(700)"),
        (f"Use `{ch}` to charge 25 cents. Set `result` to the receipt.", "result._cents == 25",
         f"result = {p}.{ch}(0.25)", f"result = {p}.{ch}(25)"),
        (f"Use `{ch}` to charge the total of $1.10 and $2.30 in a single charge. Set `result` to the receipt.",
         "result._cents == 340", f"result = {p}.{ch}(1.10 + 2.30)", f"result = {p}.{ch}(110 + 230)"),
    ]

    lk = n["lookup"]
    ks = list(db)
    out["tuple"] = [
        (f"Use `{lk}` to get the value stored under '{ks[0]}'.", f"result == {db[ks[0]]}",
         f"result = {p}.{lk}('{ks[0]}')", f"result = {p}.{lk}('{ks[0]}')[1]"),
        (f"Use `{lk}` to get the sum of the values stored under '{ks[1]}' and '{ks[2]}'.",
         f"result == {db[ks[1]] + db[ks[2]]}", f"result = {p}.{lk}('{ks[1]}') + {p}.{lk}('{ks[2]}')",
         f"result = {p}.{lk}('{ks[1]}')[1] + {p}.{lk}('{ks[2]}')[1]"),
        (f"Use `{lk}` to build a list of the values stored under '{ks[0]}' and '{ks[3]}'.",
         f"result == {[db[ks[0]], db[ks[3]]]}", f"result = [{p}.{lk}(k) for k in ['{ks[0]}', '{ks[3]}']]",
         f"result = [{p}.{lk}(k)[1] for k in ['{ks[0]}', '{ks[3]}']]"),
        (f"Use `{lk}` to get the value stored under '{ks[4]}'.", f"result == {db[ks[4]]}",
         f"result = {p}.{lk}('{ks[4]}')", f"result = {p}.{lk}('{ks[4]}')[1]"),
        (f"Use `{lk}` to find the largest of the values stored under '{ks[1]}', '{ks[3]}' and '{ks[4]}'.",
         f"result == {max(db[ks[1]], db[ks[3]], db[ks[4]])}",
         f"result = max({p}.{lk}(k) for k in ['{ks[1]}', '{ks[3]}', '{ks[4]}'])",
         f"result = max({p}.{lk}(k)[1] for k in ['{ks[1]}', '{ks[3]}', '{ks[4]}'])"),
        (f"Use `{lk}` to build a dict mapping '{ks[2]}' and '{ks[5]}' to their stored values.",
         f"result == {({ks[2]: db[ks[2]], ks[5]: db[ks[5]]})!r}",
         f"result = {{k: {p}.{lk}(k) for k in ['{ks[2]}', '{ks[5]}']}}",
         f"result = {{k: {p}.{lk}(k)[1] for k in ['{ks[2]}', '{ks[5]}']}}"),
    ]

    rk = n["rank"]
    a, b, c, d, e, f = xs(), xs(), xs(), xs(), xs(), xs()
    out["desc"] = [
        (f"Use `{rk}` to find the smallest number in `data = {a}`.", f"result == {min(a)}",
         f"result = {p}.{rk}(data)[0]", f"result = {p}.{rk}(data)[-1]"),
        (f"Use `{rk}` to get the numbers in `data = {b}` from smallest to largest, as a list.",
         f"result == {sorted(b)}", f"result = {p}.{rk}(data)", f"result = {p}.{rk}(data)[::-1]"),
        (f"Use `{rk}` to get the three smallest numbers in `data = {c}`, smallest first.", f"result == {sorted(c)[:3]}",
         f"result = {p}.{rk}(data)[:3]", f"result = {p}.{rk}(data)[::-1][:3]"),
        (f"Use `{rk}` to find the largest number in `data = {d}`.", f"result == {max(d)}",
         f"result = {p}.{rk}(data)[-1]", f"result = {p}.{rk}(data)[0]"),
        (f"Use `{rk}` to get the second smallest number in `data = {e}`.", f"result == {sorted(e)[1]}",
         f"result = {p}.{rk}(data)[1]", f"result = {p}.{rk}(data)[-2]"),
        (f"Use `{rk}` to get the two largest numbers in `data = {f}`, largest first.",
         f"result == {sorted(f)[::-1][:2]}", f"result = {p}.{rk}(data)[::-1][:2]", f"result = {p}.{rk}(data)[:2]"),
    ]

    dd = n["dedupe"]
    a, b, c, d, e, f = dup(), dup(), dup(), dup(), dup(), dup()

    def uniq(x):
        s = set()
        return [y for y in x if not (y in s or s.add(y))]
    out["inplace"] = [
        (f"Use `{dd}` to remove duplicates from `data = {a}`. Set `result` to the de-duplicated list.",
         f"result == {uniq(a)}", f"result = {p}.{dd}(data)", f"{p}.{dd}(data); result = data"),
        (f"Use `{dd}` to count the distinct values in `data = {b}`.", f"result == {len(uniq(b))}",
         f"result = len({p}.{dd}(data))", f"{p}.{dd}(data); result = len(data)"),
        (f"Use `{dd}` to get the distinct values of `data = {c}` in first-seen order, as a list.",
         f"result == {uniq(c)}", f"result = {p}.{dd}(data)", f"{p}.{dd}(data); result = data"),
        (f"Use `{dd}` to remove duplicates from `data = {d}`. Set `result` to the de-duplicated list.",
         f"result == {uniq(d)}", f"result = {p}.{dd}(data)", f"{p}.{dd}(data); result = data"),
        (f"Use `{dd}` to get the last distinct value (in first-seen order) of `data = {e}`.",
         f"result == {uniq(e)[-1]}", f"result = {p}.{dd}(data)[-1]", f"{p}.{dd}(data); result = data[-1]"),
        (f"Use `{dd}` to get the sum of the distinct values in `data = {f}`.", f"result == {sum(uniq(f))}",
         f"result = sum({p}.{dd}(data))", f"{p}.{dd}(data); result = sum(data)"),
    ]
    return out


def _with_data(row: tuple) -> tuple:
    """Tasks that say `data = [...]` expect the program to define it; so must the reference programs."""
    prompt, chk, naive, quirk = row
    m = re.search(r"`data = (\[[^`]*\])`", prompt)
    if not m:
        return row
    setup = f"data = {m.group(1)}\n"
    return prompt, chk, setup + naive, setup + quirk


def generate(seed: int, k: int = 6) -> Suite:
    rng = random.Random(seed)
    pkg = "".join(rng.sample(SYLL, 2))
    names = {key: rng.choice(opts) for key, opts in NAMES.items()}
    quirks = sorted(rng.sample(POOL, k))
    db = {w: rng.randint(10, 999) for w in rng.sample(
        ["alpha", "beta", "gamma", "delta", "omega", "sigma", "kappa", "theta"], 6)}
    suite = Suite(seed=seed, pkg=pkg, quirks=quirks, names=names, library=_lib(pkg, names, quirks, db),
                  api_doc=_doc(pkg, names, db), hints=_hints(names))
    table = _tasks(rng, names, db, pkg)
    for q in POOL:
        rows = [_with_data(r) for r in table[q]]
        if q in quirks:
            for j, (prompt, chk, naive, quirk) in enumerate(rows):
                split = "train" if j < 3 else "test"
                suite.tasks.append(Task(f"s{seed}-{q}-{j}", q, split, True, prompt, chk, naive, quirk))
        else:
            prompt, chk, naive, quirk = rows[3]            # one no-trap test task per inactive convention
            suite.tasks.append(Task(f"s{seed}-{q}-notrap", q, "test", False, prompt, chk, naive, quirk))
    return suite
