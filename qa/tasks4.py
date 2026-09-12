"""E5: a SECOND domain — writing Python code, not shell/text manipulation.

Each task has a semantic trap: the obvious implementation passes a casual glance and fails a real check.
These are bugs a coding agent actually ships, not shell trivia. Every checker imports the agent's module and
asserts behaviour, so the agent cannot satisfy it by printing the right thing.
"""

DOMAIN2 = [
    {   # trap: mutable default argument shared across calls
        "id": "p1_mutable_default",
        "prompt": "Write cart.py with a function add_item(item, basket=None) that appends item to basket and "
                  "returns it. Calling add_item('a') twice must return a list with one item each time.",
        "checker": "python3 -c \"import cart;a=cart.add_item('a');b=cart.add_item('b');assert a==['a'],a;assert b==['b'],b\"",
    },
    {   # trap: float equality
        "id": "p2_float_equality",
        "prompt": "Write money.py with is_equal(a, b) that returns True when two prices are the same amount. "
                  "is_equal(0.1+0.2, 0.3) must be True.",
        "checker": "python3 -c \"import money;assert money.is_equal(0.1+0.2,0.3);assert not money.is_equal(0.1,0.2)\"",
    },
    {   # trap: naive datetime / timezone
        "id": "p3_timezone",
        "prompt": "Write when.py with to_utc_hour(iso_string) that takes an ISO timestamp with an offset like "
                  "'2026-01-01T09:00:00+02:00' and returns the hour in UTC as an int (7 for that example).",
        "checker": "python3 -c \"import when;assert when.to_utc_hour('2026-01-01T09:00:00+02:00')==7;assert when.to_utc_hour('2026-01-01T23:30:00-05:00')==4\"",
    },
    {   # trap: modifying a list while iterating it
        "id": "p4_mutate_while_iterating",
        "prompt": "Write clean.py with drop_negatives(nums) that removes every negative number from the list "
                  "in place and returns it.",
        "checker": "python3 -c \"import clean;x=[1,-1,-2,3,-4,5];r=clean.drop_negatives(x);assert r==[1,3,5],r;assert x==[1,3,5],x\"",
    },
    {   # trap: greedy regex
        "id": "p5_greedy_regex",
        "prompt": "Write tags.py with first_tag(text) that returns the first HTML tag including its angle "
                  "brackets. first_tag('<b>hi</b> <i>there</i>') must return '<b>'.",
        "checker": "python3 -c \"import tags;assert tags.first_tag('<b>hi</b> <i>there</i>')=='<b>';assert tags.first_tag('x <span class=\\\\\\\"a\\\\\\\">y')=='<span class=\\\\\\\"a\\\\\\\">'\"",
    },
    {   # trap: integer division / rounding of money
        "id": "p6_rounding",
        "prompt": "Write split.py with split_bill(total_cents, people) that divides a bill into whole cents so "
                  "the parts sum exactly to the total, largest parts first. split_bill(100, 3) must be [34, 33, 33].",
        "checker": "python3 -c \"import split;r=split.split_bill(100,3);assert r==[34,33,33],r;assert sum(split.split_bill(1001,7))==1001\"",
    },
]

# known-good reference solutions, used only by verify_checkers to prove each task is winnable
SOLUTIONS_D2 = {
    "p1_mutable_default": "cat > cart.py <<'EOF'\ndef add_item(item, basket=None):\n    if basket is None:\n        basket = []\n    basket.append(item)\n    return basket\nEOF",
    "p2_float_equality": "cat > money.py <<'EOF'\nimport math\ndef is_equal(a, b):\n    return math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-9)\nEOF",
    "p3_timezone": "cat > when.py <<'EOF'\nfrom datetime import datetime, timezone\ndef to_utc_hour(s):\n    return datetime.fromisoformat(s).astimezone(timezone.utc).hour\nEOF",
    "p4_mutate_while_iterating": "cat > clean.py <<'EOF'\ndef drop_negatives(nums):\n    nums[:] = [n for n in nums if n >= 0]\n    return nums\nEOF",
    "p5_greedy_regex": "cat > tags.py <<'EOF'\nimport re\ndef first_tag(text):\n    m = re.search(r'<[^>]*>', text)\n    return m.group(0) if m else ''\nEOF",
    "p6_rounding": "cat > split.py <<'EOF'\ndef split_bill(total_cents, people):\n    base, rem = divmod(total_cents, people)\n    return [base + 1] * rem + [base] * (people - rem)\nEOF",
}
