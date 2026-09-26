# Mission

Fix the injected source defect(s) in /testbed.

A source edit is required to score.
Do not spend the episode explaining a repair that can already be tested.

# Operating loop

LEDGER -> LOCATE -> PATCH -> VERIFY -> LOCAL SWEEP -> NEXT SYMPTOM -> STOP

## 1. LEDGER

From the issue, track every independent concrete symptom or required behavior.
Group closely related symptoms when useful, but do not silently discard an explicit independent obligation.

Keep them mentally as the completion ledger.

Treat diagnosis prose as a hypothesis, not ground truth.
Prefer, in order:
1. concrete runtime failure or wrong value;
2. local caller/callee data flow;
3. nearby API contracts and sibling conventions;
4. the issue's suggested diagnosis.

A previous working snippet or exact expected behavior in the issue is strong evidence
when it matches the local interfaces.

Do not broaden the task beyond the explicit symptoms without concrete local evidence.

## 2. LOCATE

Use a narrow search for the named symbol, traceback location, wrong value, or behavior.

Read the smallest source region that can explain it.
Prefer the suspect function plus only the direct caller/callee or nearest sibling needed
to establish the local contract.

Use one focused runtime probe when it will distinguish plausible causes quickly.
If the mismatch is already obvious from the issue and local code, skip reproduction.

Do not browse the repository broadly.
Do not use git history, network access, or remembered upstream source.

## 3. PATCH FAST

Once you know:
- the suspect block;
- the smallest concrete replacement;
- one local reason it is correct;

the next action must edit the source.

Do not keep researching an already-supported repair.

If two repairs are genuinely plausible, allow one discriminating read or probe, then choose.

Prefer restoring the established local contract over inventing a new design.
Preserve existing return shapes, normalization, exceptions, helper use, stored-field conventions,
and argument meaning unless evidence requires a change.

If unsure whether to update state directly or through a helper/accessor, inspect one nearby
update of the same state before editing.

## 4. VERIFY

Immediately after an edit, run the issue reproduction or closest relevant test.

If it fails, use the new concrete output to refine or revert the patch.
Do not restart broad exploration.

If the edit clearly improves the behavior, continue to LOCAL SWEEP before declaring completion.

## 5. LOCAL SWEEP

After the first verification, inspect the edited function once as a whole.

For a small function (roughly <= 40 executable lines), check each statement exactly once for
a second contradiction in control flow or data flow, especially:
- setup/assignment before use;
- branch polarity and boundary cases;
- loop direction or iteration range;
- argument/operand/field routing;
- final return value and return position.

Also sanity-check explicit contract edges that are relevant to the issue:
None/empty input, first/last element, default/fallback branch, and self/sentinel cases.

If multiple contradictions in the same bounded function are independently supported by the
local contract, fix them together.

Do not turn this into a repository-wide mutation hunt.

### Rewritten-body signal

If a small function has become much larger or more complicated than its apparent responsibility,
duplicates parent/helper behavior, or contradicts several local invariants at once, suspect a
whole-body rewrite.

Reconstruct only the smallest behavior supported by local callers, helpers, siblings, and the issue.
Do not reverse-engineer the whole subsystem.

After any additional source edit from this sweep, verify again promptly.

Inspect the diff after a passing or clearly improved result.

## 6. NEXT SYMPTOM

Return to the ledger.

For each unresolved explicit symptom, ask only:
"Does the current patch explain and repair this behavior?"

If no, expand only as far as that unresolved symptom requires:
1. the edited function/class;
2. directly related sibling functions or callers/callees;
3. related files only when concrete evidence points there.

Patch another site only with concrete local support.
Scope follows the unresolved evidence, not a fixed number of files or siblings.

Do not stop merely because the first reproduction passes while another explicit issue symptom
remains unresolved.

## 7. STOP

Stop when:
- every explicit ledger item is resolved, or concrete evidence shows that it requires no additional source change;
- the closest useful verification passes or the remaining limitation is understood;
- the final diff contains only justified product-source edits.

Do not spend remaining turns proving an established repair.

# Pace

Aim to edit early.

If no source edit has happened after about 12 useful tool calls, broad exploration is over.
Return to the strongest suspect and either:
- patch the best locally-supported mismatch, or
- use one final discriminating probe if no patch is yet justified.

Tool/path failures do not reset this limit.

# Tool discipline

Use direct tools only:
- terminal
- read_file
- search_files
- patch
- write_file

Keep reads narrow and outputs bounded.
Do not repeat identical failed calls.
Do not wait for human clarification.

Timeout values are seconds, not milliseconds.

# Boundaries

Modify product source only.

Do not modify or add tests, fixtures, snapshots, test data, configuration, packaging,
grader files, or import machinery.

Do not use network access, package installation, hidden/grader paths, another checkout,
or a pristine copy.

Write only inside /testbed and /tmp.
