"""
P1.3 -- sampling-mode verifier for the P1.1 speculative decoder.

Three checks (notes/project_plan_v9.md sec 7 P1.3):

  1. STATISTICAL alpha -- the measured acceptance rate must match the theory
     alpha = E[min(p, q)]. For each draft proposal actually given an acceptance
     test we record both full distributions and m_i = sum_x min(p_DM(x), p_TM(x))
     (the exact per-position acceptance probability). Then:
        empirical  = mean(accepted_i)
        theoretical = mean(m_i)
     and we compare them against the Poisson-binomial standard error
     SE = sqrt(sum m_i (1 - m_i)) / n. |empirical - theoretical| within ~2*SE is
     "agrees" (sampling noise); anything larger is a real discrepancy.

  2. BONUS-TOKEN PROVENANCE (坑2) -- on full-accept rounds the bonus token must be
     drawn from p_TM, not p_DM. For each recorded bonus token x we have both
     log p_TM(x) and log p_DM(x); the statistic
        delta = mean(log p_TM(x) - log p_DM(x))
     is ~ +KL(p_TM || p_DM) >= 0 when x ~ p_TM (correct) and ~ -KL(p_DM || p_TM)
     <= 0 when x ~ p_DM (the bug). We run the decoder correctly and again with
     Injection(bonus_from_draft=True) and require delta_correct > delta_bug with
     delta_correct >= 0 >= delta_bug. This is the sampling-mode counterpart of
     P1.2's greedy fault-injection check -- in greedy mode 坑2 is often invisible
     (draft and target frequently share the bonus argmax), so the distributional
     test here is what actually pins it down.

  3. DOWNSTREAM PARITY -- one-sentence summaries of a small hermetic passage set
     (src/parity_data.py), scored with ROUGE-L against reference summaries.
     Speculative sampling vs target-only sampling, 3 seeds each. The plan asks for
     "HumanEval pass@1 / ROUGE, gap < 2 points"; ROUGE-L on a bundled set was
     chosen over HumanEval because it needs no dataset download and no
     code-execution sandbox, and a 1.5B target is too weak at code for pass@1 to
     be anything but noise at the sample sizes feasible on this machine. ROUGE-L
     is implemented inline (LCS F1) to keep the run dependency-free; the formula
     is the standard one used by `rouge-score`'s rougeL.

Run:  python src/verify_sampling.py
"""
import json
import math
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from model_loader import DRAFT_MODEL_NAME, TARGET_MODEL_NAME, load_model_and_tokenizer
from parity_data import PARITY_ITEMS
from prompts import PROMPTS
from rejection_sampling import Injection, speculative_generate, target_only_generate

GAMMA = 4
TEMPERATURE = 1.0
SEEDS = [0, 1, 2]
STAT_MAX_NEW_TOKENS = 48
PARITY_MAX_NEW_TOKENS = 60
RESULTS_PATH = Path(__file__).resolve().parent.parent / "results" / "p1_3_sampling_verifier.json"


# --------------------------------------------------------------------------- #
# ROUGE-L (LCS F1), inline
# --------------------------------------------------------------------------- #
def _tokenize(s):
    return re.findall(r"[a-z0-9]+", s.lower())


def _lcs_len(a, b):
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b, 1):
            cur.append(prev[j - 1] + 1 if x == y else max(prev[j], cur[-1]))
        prev = cur
    return prev[-1]


def rouge_l_f1(prediction, reference):
    p, r = _tokenize(prediction), _tokenize(reference)
    if not p or not r:
        return 0.0
    lcs = _lcs_len(p, r)
    if lcs == 0:
        return 0.0
    prec, rec = lcs / len(p), lcs / len(r)
    return 2 * prec * rec / (prec + rec)


# --------------------------------------------------------------------------- #
# Check 1: statistical alpha
# --------------------------------------------------------------------------- #
def statistical_alpha(draft_model, target_model, tokenizer):
    accepts, overlaps = [], []
    per_prompt = []
    for prompt in PROMPTS:
        p_acc, p_ov = [], []
        for seed in SEEDS:
            g = speculative_generate(
                prompt, draft_model, target_model, tokenizer,
                gamma=GAMMA, max_new_tokens=STAT_MAX_NEW_TOKENS,
                temperature=TEMPERATURE, seed=seed, record=True,
            )
            for pr in g.proposals:
                if pr["index"] == "bonus":
                    continue
                p_acc.append(1.0 if pr["accepted"] else 0.0)
                p_ov.append(pr["min_overlap"])
        accepts.extend(p_acc)
        overlaps.extend(p_ov)
        per_prompt.append({
            "prompt": prompt[:48],
            "n": len(p_acc),
            "empirical_alpha": sum(p_acc) / len(p_acc) if p_acc else 0.0,
            "theoretical_alpha": sum(p_ov) / len(p_ov) if p_ov else 0.0,
        })

    n = len(accepts)
    empirical = sum(accepts) / n
    theoretical = sum(overlaps) / n
    se_pb = math.sqrt(sum(m * (1 - m) for m in overlaps)) / n
    se_bern = math.sqrt(empirical * (1 - empirical) / n)
    diff = empirical - theoretical
    z = diff / se_pb if se_pb else float("inf")
    return {
        "n_proposals": n,
        "empirical_alpha": empirical,
        "theoretical_alpha": theoretical,
        "abs_diff": abs(diff),
        "se_poisson_binomial": se_pb,
        "se_bernoulli": se_bern,
        "z_score": z,
        "agrees_within_2se": abs(z) < 2.0,
        "per_prompt": per_prompt,
    }


# --------------------------------------------------------------------------- #
# Check 2: bonus-token provenance (坑2)
# --------------------------------------------------------------------------- #
def _collect_bonus_delta(draft_model, target_model, tokenizer, injection):
    deltas, lp_tm, lp_dm = [], [], []
    eps = 1e-12
    for prompt in PROMPTS:
        for seed in SEEDS:
            g = speculative_generate(
                prompt, draft_model, target_model, tokenizer,
                gamma=GAMMA, max_new_tokens=STAT_MAX_NEW_TOKENS,
                temperature=TEMPERATURE, seed=seed, record=True, injection=injection,
            )
            for pr in g.proposals:
                if pr["index"] != "bonus":
                    continue
                a = math.log(pr["p_tm"] + eps)
                b = math.log(pr["p_dm"] + eps)
                lp_tm.append(a)
                lp_dm.append(b)
                deltas.append(a - b)
    n = len(deltas)
    return {
        "n_bonus_tokens": n,
        "mean_log_p_tm": sum(lp_tm) / n if n else None,
        "mean_log_p_dm": sum(lp_dm) / n if n else None,
        "delta_mean": sum(deltas) / n if n else None,  # ~ +KL(tm||dm) if x~p_tm, ~ -KL(dm||tm) if x~p_dm
    }


def bonus_provenance(draft_model, target_model, tokenizer):
    correct = _collect_bonus_delta(draft_model, target_model, tokenizer, Injection())
    bug = _collect_bonus_delta(draft_model, target_model, tokenizer, Injection(bonus_from_draft=True))
    d_ok = correct["delta_mean"]
    d_bug = bug["delta_mean"]
    passed = (d_ok is not None and d_bug is not None and d_ok > d_bug and d_ok >= 0.0 >= d_bug)
    return {
        "correct_impl": correct,
        "bonus_from_draft_bug": bug,
        "separates_correct_from_bug": bool(passed),
        "note": "delta = mean(log p_TM(bonus) - log p_DM(bonus)); "
                ">=0 means bonus follows the target, <0 means it leaked from the draft",
    }


# --------------------------------------------------------------------------- #
# Check 3: downstream parity (ROUGE-L)
# --------------------------------------------------------------------------- #
def downstream_parity(draft_model, target_model, tokenizer):
    spec_scores, ref_scores = [], []
    per_item = []
    for item in PARITY_ITEMS:
        prompt = f"Summarize the following passage in one sentence:\n\n{item['text']}"
        s_item, r_item = [], []
        for seed in SEEDS:
            s = speculative_generate(
                prompt, draft_model, target_model, tokenizer,
                gamma=GAMMA, max_new_tokens=PARITY_MAX_NEW_TOKENS, temperature=TEMPERATURE, seed=seed,
            )
            r = target_only_generate(
                prompt, target_model, tokenizer,
                max_new_tokens=PARITY_MAX_NEW_TOKENS, temperature=TEMPERATURE, seed=seed,
            )
            s_item.append(rouge_l_f1(s.text, item["summary"]))
            r_item.append(rouge_l_f1(r.text, item["summary"]))
        spec_scores.extend(s_item)
        ref_scores.extend(r_item)
        per_item.append({
            "summary_ref": item["summary"][:60],
            "spec_rougeL_mean": sum(s_item) / len(s_item),
            "target_only_rougeL_mean": sum(r_item) / len(r_item),
        })
    spec_mean = 100 * sum(spec_scores) / len(spec_scores)
    ref_mean = 100 * sum(ref_scores) / len(ref_scores)
    gap = abs(spec_mean - ref_mean)
    return {
        "n_generations_per_method": len(spec_scores),
        "spec_rougeL_mean_points": spec_mean,
        "target_only_rougeL_mean_points": ref_mean,
        "gap_points": gap,
        "within_2_points": gap < 2.0,
        "per_item": per_item,
    }


def main():
    print(f"draft  = {DRAFT_MODEL_NAME}")
    print(f"target = {TARGET_MODEL_NAME}")
    print(f"gamma  = {GAMMA}, temperature = {TEMPERATURE}, seeds = {SEEDS}\n")

    draft_model, _ = load_model_and_tokenizer(DRAFT_MODEL_NAME)
    target_model, tokenizer = load_model_and_tokenizer(TARGET_MODEL_NAME)

    print("-- check 1: statistical alpha vs E[min(p,q)] --")
    stat = statistical_alpha(draft_model, target_model, tokenizer)
    print(f"   empirical={stat['empirical_alpha']:.4f}  theoretical={stat['theoretical_alpha']:.4f}  "
          f"|diff|={stat['abs_diff']:.4f}  z={stat['z_score']:.2f}  "
          f"-> {'AGREES' if stat['agrees_within_2se'] else 'DISCREPANCY'}")

    print("\n-- check 2: bonus-token provenance (坑2) --")
    prov = bonus_provenance(draft_model, target_model, tokenizer)
    print(f"   delta correct={prov['correct_impl']['delta_mean']:.4f}  "
          f"delta bug={prov['bonus_from_draft_bug']['delta_mean']:.4f}  "
          f"-> {'PASS' if prov['separates_correct_from_bug'] else 'FAIL'}")

    print("\n-- check 3: downstream ROUGE-L parity --")
    parity = downstream_parity(draft_model, target_model, tokenizer)
    print(f"   spec={parity['spec_rougeL_mean_points']:.2f}  "
          f"target_only={parity['target_only_rougeL_mean_points']:.2f}  "
          f"gap={parity['gap_points']:.2f} pts -> {'PASS' if parity['within_2_points'] else 'FAIL'}")

    result = {
        "draft_model": DRAFT_MODEL_NAME,
        "target_model": TARGET_MODEL_NAME,
        "gamma": GAMMA,
        "temperature": TEMPERATURE,
        "seeds": SEEDS,
        "statistical_alpha": stat,
        "bonus_provenance": prov,
        "downstream_parity": parity,
        "p1_3_pass": bool(
            stat["agrees_within_2se"]
            and prov["separates_correct_from_bug"]
            and parity["within_2_points"]
        ),
    }
    RESULTS_PATH.parent.mkdir(exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(result, indent=2))
    print("\n" + "=" * 60)
    print(f"P1.3 overall: {'PASS' if result['p1_3_pass'] else 'FAIL'}")
    print(f"written to {RESULTS_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    main()
