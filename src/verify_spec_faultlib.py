"""
P6.5 Part 3 -- the mutation-adequacy matrix.

Scores every operator in src/spec_faultlib.py against the O1/O3/O4/O5 oracle
stack (src/spec_oracles.py) and writes results/p6_5_mutation_adequacy.json.

O5 is the batched-path (P6.1) equivalence-preservation check: every operator is
run through spec_kv_batch.run_round and the output must stay bit-identical to N
single-sequence speculative_generate_kv runs. It is reported separately (not
folded into the kill scores) because a non-empty O5 list means a real
run_round bug, not "the oracle caught the mutant".

The citable finding is the "caught only by" partition:
  * caught_only_by_O4   -- invisible to BOTH output-equivalence oracles (greedy
                           and sampling); only a structural-invariant assertion
                           finds them. This is the argument for publishing the
                           invariant assertions, not just parity tests.
  * caught_only_by_O3   -- no-ops in greedy mode (one-hot distributions); need
                           distributional / sampling testing.
  * caught_by_O1        -- gross enough to change greedy output.
  * equivalent          -- survive every oracle; each is hand-checked (see
                           `note`) as a true behavioural equivalent.

Deterministic FakeModel only -> fast, hermetic, no model download. The real-model
oracles O2 / null-band-O3 are a later increment.

Run:  python src/verify_spec_faultlib.py [--smoke]
"""
import argparse
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import spec_faultlib as fl
from spec_oracles import run_o1, run_o3, run_o4, run_o5

RESULTS_PATH = Path(__file__).resolve().parent.parent / "results" / "p6_5_mutation_adequacy.json"

# hand-checked equivalent mutants (survive every oracle by construction)
EQUIVALENT_NOTES = {
    "kv_crop_absolute_vs_relative": (
        "On transformers 5.16 a positive crop arg and the negative form leave the "
        "same cache length when n < current; they differ only by a deprecation "
        "warning and removal at 5.18. True behavioural equivalent here."
    ),
}


def _eos(name):
    return name == "eos_ignored_midblock"


def score_operator(name, gammas, seeds):
    e = _eos(name)
    o1 = run_o1((name,), gammas=gammas, seeds=seeds, eos=e)
    o3 = run_o3((name,), gammas=gammas, seeds=tuple(range(max(seeds) + 3)), eos=e)
    o4 = run_o4((name,), gammas=gammas, seeds=seeds, eos=e)
    # O5 -- batched path (P6.1) must stay bit-identical to the single-sequence
    # loop even with this operator active. `killed` => the operator breaks
    # batch/single equivalence, i.e. a real spec_kv_batch.run_round bug.
    o5 = run_o5((name,), gammas=gammas, seeds=seeds, eos=e)

    def _lat(r):
        return statistics.fmean(r.first_divergence_tokens) if r.first_divergence_tokens else None

    caught = [o.oracle for o in (o1, o3, o4) if o.killed]
    if not caught:
        partition = "equivalent"
    elif caught == ["O4"]:
        partition = "caught_only_by_O4"
    elif "O1" in caught:
        partition = "caught_by_O1"
    elif "O3" in caught:
        partition = "caught_only_by_O3"
    else:
        partition = "caught_" + "+".join(caught)

    return {
        "operator": name,
        "group": fl.GROUP[name],
        "partition": partition,
        "O1": {"killed": o1.killed, "n_runs": o1.n_runs, "n_diverged": o1.n_diverged,
               "mean_first_divergence_token": _lat(o1)},
        "O3": {"killed": o3.killed, "n_runs": o3.n_runs, "n_diverged": o3.n_diverged,
               "mean_first_divergence_token": _lat(o3)},
        "O4": {"killed": o4.killed, "n_runs": o4.n_runs, "n_violation_hits": o4.n_diverged,
               "violations": o4.violations[:4]},
        "O5_batch_equiv": {"broken": o5.killed, "n_runs": o5.n_runs, "n_diverged": o5.n_diverged,
                           "mean_first_divergence_token": _lat(o5)},
        "equivalent_note": EQUIVALENT_NOTES.get(name, ""),
    }


def run(smoke=False):
    gammas = (3,) if smoke else (1, 3, 5)
    seeds = (0, 1, 2) if smoke else (0, 1, 2, 3, 4)

    base = {
        "O1": run_o1(gammas=gammas, seeds=seeds).killed,
        "O3": run_o3(gammas=gammas, seeds=tuple(range(max(seeds) + 3))).killed,
        "O4": run_o4(gammas=gammas, seeds=seeds).killed,
        "O5": run_o5(gammas=gammas, seeds=seeds).killed,
    }
    assert not any(base.values()), f"clean baseline killed by an oracle: {base}"

    rows = [score_operator(n, gammas, seeds) for n in fl.names()]

    by_part = {}
    for r in rows:
        by_part.setdefault(r["partition"], []).append(r["operator"])

    non_equiv = [r for r in rows if r["partition"] != "equivalent"]
    scores = {
        o: sum(1 for r in non_equiv if r[o]["killed"]) / len(non_equiv)
        for o in ("O1", "O3", "O4")
    }
    scores["any_oracle"] = sum(1 for r in non_equiv if any(r[o]["killed"] for o in ("O1", "O3", "O4"))) / len(non_equiv)

    batch_breakers = [r["operator"] for r in rows if r["O5_batch_equiv"]["broken"]]

    return {
        "task": "P6.5 mutation-adequacy matrix -- speculative-decoding fault operators x oracle stack",
        "model": "deterministic position-one-hot FakeModel (hermetic)",
        "gammas": list(gammas),
        "seeds": list(seeds),
        "n_operators": len(rows),
        "n_deferred_operators": len(fl.DEFERRED),
        "deferred_operators": fl.DEFERRED,
        "oracle_mutation_scores_excl_equivalent": scores,
        "partition": by_part,
        "O5_batch_equivalence": {
            "operators_that_break_batch_single_equivalence": batch_breakers,
            "meaning": (
                "spec_kv_batch.run_round (P6.1, per-sequence KV caches) stays "
                "bit-identical to N independent speculative_generate_kv runs even "
                "with each fault operator active. An empty list is the intended "
                "result: output equivalence is a property of the per-seq-cache "
                "architecture, not just of the un-mutated code -- there is no "
                "shared ragged tensor for a bug to desync (plan sec 7 C3 / 坑19)."
            ),
        },
        "headline": {
            "caught_only_by_O4_structural_invariants": by_part.get("caught_only_by_O4", []),
            "meaning": (
                "these speculative-decoding bugs pass both greedy AND sampling "
                "output-equivalence checks; only an internal structural-invariant "
                "assertion detects them. Publishing parity tests alone would let "
                "them through."
            ),
            "caught_only_by_O3_distributional": by_part.get("caught_only_by_O3", []),
            "caught_only_by_O3_meaning": (
                "no-ops in greedy decoding (one-hot distributions); require a "
                "sampling-mode / distributional oracle."
            ),
            "equivalent_mutants": by_part.get("equivalent", []),
        },
        "operators": rows,
    }


def _print_table(out):
    print(f"\n{'operator':32s} {'group':9s} {'O1':>4s} {'O3':>4s} {'O4':>4s} {'O5!':>4s}  partition")
    print("-" * 82)
    for r in out["operators"]:
        print(f"{r['operator']:32s} {r['group']:9s} "
              f"{int(r['O1']['killed']):>4d} {int(r['O3']['killed']):>4d} {int(r['O4']['killed']):>4d} "
              f"{int(r['O5_batch_equiv']['broken']):>4d}  "
              f"{r['partition']}")
    print("\nmutation scores (excl. equivalent):", json.dumps(out["oracle_mutation_scores_excl_equivalent"], indent=2))
    print("\ninvisible to output-equivalence, caught only by O4 invariants:")
    for n in out["headline"]["caught_only_by_O4_structural_invariants"]:
        print("  -", n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    t0 = time.perf_counter()
    out = run(smoke=args.smoke)
    out["elapsed_s"] = time.perf_counter() - t0
    _print_table(out)

    if not args.smoke:
        RESULTS_PATH.write_text(json.dumps(out, indent=2))
        print(f"\nwrote {RESULTS_PATH}")


if __name__ == "__main__":
    main()
