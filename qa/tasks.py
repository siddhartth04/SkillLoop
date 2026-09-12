"""Real tasks. Each has an INDEPENDENT checker command: exit 0 = the goal was actually achieved.

Phase A = first encounter (baseline, no skills in the library yet).
Phase B = a DIFFERENT task that needs the SAME lesson, worded differently and in a different domain surface,
          so retrieval has to work on more than keyword overlap.
"""

CSV_LATIN1 = "name,city\nJos\xe9,S\xe3o Paulo\nRen\xe9e,Z\xfcrich\nAnna,Berlin\n"

PHASE_A = [
    {
        "id": "a1_install",
        "prompt": "Install the python package 'tabulate' and then write a file out.txt containing the output of "
                  "python -c \"import tabulate; print(tabulate.__name__)\"",
        "checker": "test -f out.txt && grep -q tabulate out.txt",
    },
    {
        "id": "a2_json",
        "prompt": "In config.json, change the port from 8000 to 8080. The file must remain valid JSON.",
        "files": {"config.json": '{\n  "service": "api",\n  "port": 8000,\n  "timeout": 8000\n}\n'},
        "checker": "python -c \"import json;d=json.load(open('config.json'));assert d['port']==8080,d;assert d['timeout']==8000,d\"",
    },
    {
        "id": "a3_encoding",
        "prompt": "Read people.csv with python and write the number of rows (excluding the header) into count.txt",
        "files": {"people.csv": CSV_LATIN1},
        "checker": "test -f count.txt && grep -qx '3' count.txt",
    },
    {
        "id": "a4_test",
        "prompt": "The test in test_discount.py is failing. Fix the code so the test passes.",
        "files": {
            "pricing.py": "def discount(price):\n    # 15% off\n    return price * 0.90\n",
            "test_discount.py": "from pricing import discount\n\ndef test_discount():\n    assert discount(100) == 85.0\n",
        },
        "checker": "grep -q '== 85.0' test_discount.py && python -m pytest -q test_discount.py 2>&1 | tail -1 | grep -q '1 passed'",
    },
    {
        "id": "a5_script",
        "prompt": "Write a script report.py that prints the total of the numbers in data.txt, one per line. "
                  "Then save its output to total.txt",
        "files": {"data.txt": "10\n20\n5\n\n7\n"},
        "checker": "test -f total.txt && grep -qx '42' total.txt",
    },
]

PHASE_B = [
    {
        "id": "b1_install",
        "lesson_from": "a1_install",
        "prompt": "Set up the 'humanize' library in this environment, then create ok.txt holding the result of "
                  "python -c \"import humanize; print('ready')\"",
        "checker": "test -f ok.txt && grep -q ready ok.txt",
    },
    {
        "id": "b2_json",
        "lesson_from": "a2_json",
        "prompt": "Update settings.json so retries becomes 5. Keep the file parseable.",
        "files": {"settings.json": '{\n  "retries": 3,\n  "backoff": 3,\n  "name": "worker-3"\n}\n'},
        "checker": "python -c \"import json;d=json.load(open('settings.json'));assert d['retries']==5,d;assert d['backoff']==3,d;assert d['name']=='worker-3',d\"",
    },
    {
        "id": "b3_encoding",
        "lesson_from": "a3_encoding",
        "prompt": "Load cities.csv in python and write the city of the last row into last.txt",
        "files": {"cities.csv": CSV_LATIN1},
        "checker": "test -f last.txt && grep -q Berlin last.txt",
    },
    {
        "id": "b4_test",
        "lesson_from": "a4_test",
        "prompt": "test_tax.py is red. Make it green.",
        "files": {
            "tax.py": "def with_tax(amount):\n    # 20% VAT\n    return amount * 1.10\n",
            "test_tax.py": "from tax import with_tax\n\ndef test_with_tax():\n    assert with_tax(50) == 60.0\n",
        },
        "checker": "grep -q '== 60.0' test_tax.py && python -m pytest -q test_tax.py 2>&1 | tail -1 | grep -q '1 passed'",
    },
    {
        "id": "b5_script",
        "lesson_from": "a5_script",
        "prompt": "Create avg.py that prints the average of the values in scores.txt (one per line), then put its "
                  "output in avg.txt",
        "files": {"scores.txt": "4\n8\n\n6\n"},
        "checker": "test -f avg.txt && grep -qx '6.0' avg.txt",
    },
]


# ---- harder set: the baseline agent genuinely fails these ----
# Each has a trap: the obvious command half-works, and only verification catches it.

HARD_A = [
    {
        "id": "h1_sed_multi",
        "prompt": "In app.conf set the worker timeout to 60. Do not change anything else.",
        "files": {"app.conf": "timeout = 30\nretry_timeout = 30\nworkers = 30\nworker_timeout = 30\n"},
        "checker": "grep -qx 'worker_timeout = 60' app.conf && grep -qx 'timeout = 30' app.conf && "
                   "grep -qx 'retry_timeout = 30' app.conf && grep -qx 'workers = 30' app.conf",
    },
    {
        "id": "h2_silent_fail",
        "prompt": "Run the build script build.sh and put its exit status in status.txt. The build must actually succeed.",
        "files": {"build.sh": "#!/bin/bash\necho 'compiling...'\nmkdir -p out\ncp src/main.c out/ 2>/dev/null\necho 'build finished'\nexit 0\n"},
        "setup": "chmod +x build.sh",
        "checker": "test -f status.txt && grep -q 'fail\\|error\\|missing\\|1' status.txt",
    },
    {
        "id": "h3_partial_glob",
        "prompt": "Add the line '# reviewed' to the top of every .py file in this project.",
        "files": {"a.py": "print(1)\n", "lib/b.py": "print(2)\n", "lib/deep/c.py": "print(3)\n"},
        "checker": "for f in a.py lib/b.py lib/deep/c.py; do head -1 $f | grep -q '# reviewed' || exit 1; done",
    },
    {
        "id": "h4_stale_import",
        "prompt": "utils.py has a bug: total() returns the wrong value. Fix it so that total([1,2,3]) is 6, "
                  "and prove it works.",
        "files": {"utils.py": "def total(xs):\n    return len(xs)\n", "__pycache__/keep": ""},
        "checker": "python -c \"import utils;assert utils.total([1,2,3])==6;assert utils.total([])==0\"",
    },
]

HARD_B = [
    {
        "id": "hb1_sed_multi",
        "lesson_from": "h1_sed_multi",
        "prompt": "Change the cache size to 512 in server.ini, leaving the other settings alone.",
        "files": {"server.ini": "size = 128\nbuffer_size = 128\ncache_size = 128\nmax_size = 128\n"},
        "checker": "grep -qx 'cache_size = 512' server.ini && grep -qx 'size = 128' server.ini && "
                   "grep -qx 'buffer_size = 128' server.ini && grep -qx 'max_size = 128' server.ini",
    },
    {
        "id": "hb2_silent_fail",
        "lesson_from": "h2_silent_fail",
        "prompt": "Execute deploy.sh and record in result.txt whether the deployment really worked.",
        "files": {"deploy.sh": "#!/bin/bash\necho 'uploading...'\ncp artifact.tar /srv/ 2>/dev/null\necho 'done'\nexit 0\n"},
        "setup": "chmod +x deploy.sh",
        "checker": "test -f result.txt && grep -qi 'fail\\|error\\|missing\\|not\\|no' result.txt",
    },
    {
        "id": "hb3_partial_glob",
        "lesson_from": "h3_partial_glob",
        "prompt": "Append the comment '// checked' as the last line of each .js file here.",
        "files": {"x.js": "let a=1\n", "src/y.js": "let b=2\n", "src/nested/z.js": "let c=3\n"},
        "checker": "for f in x.js src/y.js src/nested/z.js; do tail -1 $f | grep -q '// checked' || exit 1; done",
    },
    {
        "id": "hb4_stale_import",
        "lesson_from": "h4_stale_import",
        "prompt": "calc.py's average() is wrong. Make average([2,4,6]) return 4.0 and show that it does.",
        "files": {"calc.py": "def average(xs):\n    return sum(xs)\n", "__pycache__/keep": ""},
        "checker": "python -c \"import calc;assert calc.average([2,4,6])==4.0\"",
    },
]
