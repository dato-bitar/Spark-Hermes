**This round:** rank 3 · Δ vs baseline +0.214 on 6 paired instances · 1 verified.

## Round `r0020` — `5CoTfEgMhZ46ETtT6RSdt3RQkXizYu3vy9Spg33qTSt3eBtQ`

**weight 0.1829** · score 0.1318

| | |
|---|---|
| episodes | 48 |
| mean d (your share of checks passed − the baseline's, same instances) | +0.1318 |
| standard error (incl. reference term) | 0.049372 |
| Δc (one-sided 90 % lower bound — how sure the gain is) | 0.0686 |
| score (mean d after the overfit and copy penalties) | +0.1318 |
| correctness gate | passed |
| Δe | — |
| overfit rate | 0.00 |
| disqualified episodes | 0 |

`mean d` is the share of each task's withheld checks your episodes passed, minus the pinned model's share on the *same instances* with no strategy. Passing tasks is not the achievement — beating that baseline is. `Δc` is the lower bound of that difference, so beating the baseline on average is not enough to be *paid* for beating it.

### The baseline you were measured against

| family | null n | null credit | canon credit | Δc canon | label |
|---|---|---|---|---|---|
| `swe_fix` | 48 | 0.12 | 0.00 | -0.12 | frontier |

### Check the grading yourself

6 of 6 withheld commitments re-verified at close: **all match**.

Each instance's withheld half was committed to *before* submissions opened, as `hmac-sha256(salt, canonical_json(withheld))`. The commitment is in the task record — under `rounds/<id>/tasks/` when you were shown the scored tasks, under `rounds/<id>/evaluated/` when you were shown previews — and `rounds/queue.json` carried the digest of those records before the round opened. The salt and the half itself are published now, in `reveal.json`. Recompute it and confirm the criteria you were graded against are the ones that were fixed in advance:

```python
import hashlib, hmac, json
salt, withheld = reveal[task_id]["salt"], reveal[task_id]["withheld"]
body = json.dumps(withheld, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
"hmac-sha256:" + hmac.new(bytes.fromhex(salt), body, hashlib.sha256).hexdigest()
```

<details><summary>Revealed withheld halves (6) — full record in `reveal.json`</summary>

| task | withheld checks | salt | source |
|---|---|---|---|
| `swe-fix-r0020-02` | 1 | `1cf1daec08c5d704…` | andialbrecht__sqlparse.e57923b3.func_basic__huype57z |
| `swe-fix-r0020-03` | 6 | `39af070cfff5ace6…` | pylint-dev__astroid.b114f6b5.func_basic__ddpz3b5t |
| `swe-fix-r0020-04` | 7 | `3d9e794863eb630b…` | pylint-dev__astroid.b114f6b5.combine_file__xi2nex5r |
| `swe-fix-r0020-08` | 9 | `5e8d49c165e048af…` | oauthlib__oauthlib.1fd52536.combine_file__u4wh8onm |
| `swe-fix-r0020-16` | 5 | `74a1d4094aa4961a…` | oauthlib__oauthlib.1fd52536.combine_module__oq5a6syn |
| `swe-fix-r0020-17` | 1 | `e5fef6928b773d39…` | pylint-dev__astroid.b114f6b5.pr_2515 |

</details>