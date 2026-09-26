# Fix the injected bug in /testbed

Working code in the Python project at `/testbed` was edited by a tool to break it. Put it back. The tests
that expose the bug were removed before your episode; the rest of the suite is still there. Every broken
test you make pass earns its share of the credit — a partial fix is worth real credit — but if any test
that passed before now fails, the task scores zero. Nobody will answer questions.

## This page overrides the runtime notes that follow it

Those notes ask you to gather prerequisites and verify before acting. Here the order is reversed:
**edit first, verify after.** You have about 25 tool calls before the budget ends the run. Runs that keep
reading until they are certain score zero, and most of them had already seen the broken line in their first
few calls and never changed it. An untouched tree scores zero, so a reasoned patch is never worse than no
patch. Think in a few sentences, then act; every reply contains a tool call until you are done.

## Facts about this machine — already checked

- The project is at `/testbed`. There is no `/workspace`, `/repo` or `/app`, and `git log` shows one commit.
  Work only from `/testbed`; do not look for other copies of the project, and never diff against one.
- Stay inside `/testbed` and `/tmp`. Do not list, read or run anything anywhere else, not even `ls`:
  nothing out there helps, and reaching outside ends the episode at zero.
- The interpreter with the project's dependencies and pytest is
  `/opt/miniconda3/envs/testbed/bin/python`. The `python` on PATH has neither. Write the full path every
  time.
- Call these tools directly: `terminal`, `read_file`, `search_files`, `patch`, `write_file`. Never call
  `tool_search`, `tool_call` or `tool_describe`, and do not open skills: everything you need is on this
  page, and every file you open is re-sent to the model on every later call.
- If a tool result says "is not a deferrable tool" or warns about a tool loop, your next call is a plain
  `terminal` command that does the same job. If a tool call errors, correct its arguments once; on a second
  failure, do it with plain `terminal`.
- Every tool output stays in the conversation and is paid for again on every later call. Small outputs are
  cheap; one whole-file read is what makes runs lose track and die over budget.

## The loop

1. **Find the code (calls 1–2).** Take the symbols the issue names and search outside the tests:
   `grep -rn "<name>" /testbed --include=*.py | grep -v /tests/ | head -20`.
   Treat the issue as a symptom report, not a diagnosis. When its explanation conflicts with the code
   you can see, trust, in this order: the concrete runtime failure or wrong value, the local
   caller/callee data flow, the local API contract and sibling functions — and the issue's prose last.
   A previous working snippet or an exact expected output in the issue is strong evidence when it
   matches the local interfaces. If the issue names a method as missing that exists, do not hunt a
   nonexistent defect: reproduce the failure and follow the first concrete break instead.
2. **See the bug (calls 3–4).** Put the issue's snippet, or a minimal call of the named function, in
   `/tmp/r.py` and run:
   `cd /testbed && /opt/miniconda3/envs/testbed/bin/python /tmp/r.py 2>&1 | tail -25`.
   The last `/testbed` frame of the traceback — or the function that returns the wrong value — is the
   suspect. The issue's expected output is the specification: your repro must print it exactly; a clean
   exit code is not proof.
3. **Read the suspect once (call 5).** `read_file` with `offset` from `grep -n` and `limit` of at most 60.
4. **Audit it line by line in your reasoning.** For each line ask: does it agree with the function's name,
   docstring, neighbours and callers? These edits are injected by a tool, and their fingerprints:
   - a name used before anything assigns it, or a lone `pass` where work belongs — a line was deleted:
     write it back just above its first use;
   - a validation loop or a `try/except` is gone: input is processed silently that used to be rejected, or
     a raw `TypeError`/`KeyError` crashes where a soft fallback used to be — restore the loop with its
     per-argument checks, or the handler with its exact fallback value;
   - a flipped comparison or boolean (`<` for `<=`, `and` for `or`, a `not` added or dropped), an
     off-by-one in a slice or `range`, a wrong constant, unit or shift direction;
   - arguments swapped, or the wrong count, against the function being called; paired names reversed;
   - a `return`, `raise`, `break` or `continue` placed so later lines are unreachable, or a guard moved
     below its use;
   - a guard that vanished: an `if` that used to skip, filter or raise on bad input no longer does, so
     the function now processes what it should reject — restore the filter its callers and the empty,
     first, last and `None` cases justify;
   - a method or property the callers use that no longer exists: restore it modelled on its siblings, same
     signature and decorator;
   - a body that reads smoothly but no longer does what its name, docstring and callers require — the
     whole function was replaced, and its docstring probably with it: trust the callers and the surviving
     tests over the prose, and write the smallest body that meets them.
5. **Patch the first line that does not match, in your very next call — call 7 at the latest.** `patch`
   it, then rerun the repro. An edit you later refine or revert costs one call; an unfixed tree costs the
   task. **Patch clock:** if no source edit has happened by call 15, further exploration is forbidden —
   return to your strongest suspect and patch your best candidate now. Tool and path failures do not
   reset this clock.
6. **Still wrong, or the issue lists several symptoms?** These bugs usually come several at a time: in the
   same function, in sibling functions of the file, and in other modules of the same package directory
   (two dialects, two formats, two halves of one parser). For each symptom still wrong, search that
   directory for the name involved: `grep -rn "<name>" <package dir> | head -20`, audit the hit, patch it.
   Stuck: change the angle — another symbol from the issue, a caller, a sibling function. Never re-read
   what you have already read. Two dead angles are not a reason to keep hunting: patch your best
   candidate now and let the repro judge it — a partial fix earns credit, an empty tree earns none.
   Suspect function small, roughly 40 executable lines or fewer? Before declaring it fixed, check every
   statement once for a second contradiction in control flow or data flow — setup before use, branch
   polarity and boundaries, loop direction and range, argument routing — and if two independent
   contradictions hold, repair both in the same bounded edit.
7. **Check (by call 20).** Run the tests for the module you changed:
   `cd /testbed && /opt/miniconda3/envs/testbed/bin/python -m pytest <test file> -q -x -p no:cacheprovider
   2>&1 | tail -15`. Make sure it actually collected and passed tests — "no tests ran" proves nothing. A
   test that fails now but passed before means your edit is wrong: narrow it or `git checkout -- <file>`.
8. **Finish.** Read `git diff` — anything you did not mean to change, put back — and stop with a one-line
   reply.

## Keep outputs small

Never print a whole file: no `read_file` without `limit` of at most 60, no `cat` of source. Find the line
with `grep -n` first. End every command that can print more than a screen with `| head -40` or
`| tail -40`. Never run the whole test suite: run the one file that covers the module you changed.

## Red lines — any one of these scores zero

- No network, no installs: no `pip`, no `curl`, no `git fetch`. A missing module is not a problem to
  solve: do not hunt for another interpreter — script a direct repro under `/tmp` that imports the
  package out of `/testbed`.
- Do not edit, add, move or delete tests or configuration: nothing under the test tree, no `conftest.py`,
  `pytest.ini`, `tox.ini`, `setup.cfg`, no pytest section of `pyproject.toml`, no fixture or golden file —
  and never a flag that regenerates expected output.
- Write only inside `/testbed` and `/tmp`. Do not browse the filesystem, and do not probe how this sandbox
  or its grading works: reaching into it ends the episode at zero.
- From the fix itself: no `import pytest` or `_pytest`, no `inspect`, `importlib`, `subprocess`, `exec` or
  `eval` tricks around the runner. Fix the code the issue is about.

## The repositories you will meet

| Project | Source | Tests | Watch out |
|---|---|---|---|
| cantools | `src/cantools/` | `tests/` | command-line tests compare exact stdout |
| python-docx | `src/docx/` | `tests/` mirror the source | warnings are errors |
| python-pptx | `src/pptx/` | `tests/` mirror the source | warnings are errors |
| astroid | `astroid/` | `tests/` | an xfail that passes is a failure |
| sqlglot | `sqlglot/` | `tests/` | `tests/fixtures/` holds golden files |
| pygments | `pygments/` | `tests/` | snapshot tests; never regenerate them |
| oauthlib | `oauthlib/` | `tests/` mirror the source | exact strings and error codes matter |
| marshmallow | `src/marshmallow/` | `tests/` flat | error messages compared exactly; pass `-q` |
| sqlparse | `sqlparse/` | `tests/` flat | formatted SQL compared as exact strings |
| gpxpy | `gpxpy/` | `test.py` at the root | one big file: select tests with `-k` |

## Order of preference

1. Every injected site fixed, nothing else broken.
2. Some sites fixed, nothing else broken.
3. The tree exactly as you found it.

A regression is worth less than nothing changed. Never leave the tree more broken than you found it.

## Remember

Read the suspect once. Audit it line by line. Patch the first mismatch immediately. Rerun the repro. Sweep
the module's tests, then stop. About 25 calls: edit first, protect what already works.

Revision v13.
