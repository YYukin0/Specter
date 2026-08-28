"""
支柱7 Bullet 3 -- downstream-task evaluation of the self-built AWQ model against
the fp16 baseline and the mlx-lm production int4 model, using the EleutherAI
lm-evaluation-harness over an OpenAI-compatible endpoint (mlx-lm server).

Question: does the +1.2 / +1.6 ppl hit from 4-bit AWQ (支柱2, P6.2) translate to
a GSM8K / IFEval accuracy hit, and by how much? The claim being tested is
"reasoning tasks degrade more than perplexity shows."

Tasks: gsm8k (5-shot CoT, generative) + ifeval (0-shot, generative). Loglikelihood
tasks (MMLU, HellaSwag) are NOT runnable over a chat/completions API -- no prompt
logprobs -- and are deliberately excluded (Bullet 3 pitfall 1).

All arms share ONE eval config and differ ONLY in weights (Bullet 3 pitfall 2):
  - each checkpoint carries the same Qwen2.5 chat template
  - --apply_chat_template, --fewshot_as_multiturn
  - greedy: temperature=0, do_sample=False
  - max_gen_toks=768 (enough for CoT; pitfall 4)
  - num_concurrent=1 (deterministic; no batch-invariance question in the eval)
  - no system prompt on any arm
  - identical --limit and --seed

Arms:
  fp16          HF cache snapshot of Qwen/Qwen2.5-1.5B-Instruct (served by mlx as fp16)
  self_awq      src/results/bullet3_self_awq_1.5b_hf  (self-built AWQ fake-quant,
                fp16 weights on the 4-bit grid -> downstream acc == fake-quant acc)
  mlx_awq_int4  src/results/p6_2_awq_1.5b  (mlx_lm.awq real int4, the P6.2 arm)

Requires the isolated harness venv .venv-lmeval (lm-eval[api] 0.4.x). This script
runs under the project .venv and shells out to .venv-lmeval/bin/lm-eval; the two
never share a process.

Run:  python src/verify_p6_6_downstream_eval.py [--limit N] [--tasks gsm8k,ifeval] [--smoke]
Writes results/p6_6_downstream_eval.json
"""
import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"
JSON_PATH = RESULTS_DIR / "p6_6_downstream_eval.json"

VENV_PY = ROOT / ".venv" / "bin" / "python"
LMEVAL = ROOT / ".venv-lmeval" / "bin" / "lm-eval"

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"

DEFAULT_LIMIT = 200
DEFAULT_TASKS = ["gsm8k", "ifeval"]
MAX_GEN_TOKS = 768
PORT = 8137


def _fp16_snapshot_path() -> Path:
    hub = Path.home() / ".cache" / "huggingface" / "hub"
    base = hub / f"models--{MODEL_ID.replace('/', '--')}" / "snapshots"
    snaps = sorted(base.glob("*"))
    # return the expected location even when absent; main() checks .exists() per arm
    return snaps[-1] if snaps else base / "MISSING"


def arms():
    return {
        "fp16": _fp16_snapshot_path(),
        "self_awq": ROOT / "src" / "results" / "bullet3_self_awq_1.5b_hf",
        "mlx_awq_int4": ROOT / "src" / "results" / "p6_2_awq_1.5b",
    }


# --------------------------------------------------------------------------- #
# mlx-lm server lifecycle
# --------------------------------------------------------------------------- #
def start_server(model_path: Path, log_path: Path):
    env = dict(os.environ)
    env["HF_HUB_OFFLINE"] = "1"
    env["TRANSFORMERS_OFFLINE"] = "1"
    lf = open(log_path, "w")
    proc = subprocess.Popen(
        [str(VENV_PY), "-m", "mlx_lm", "server", "--model", str(model_path),
         "--port", str(PORT), "--host", "127.0.0.1", "--log-level", "WARNING"],
        stdout=lf, stderr=subprocess.STDOUT, env=env,
    )
    return proc, lf


def wait_ready(proc, timeout=180) -> float:
    """Block until the server answers a real chat request. Returns load seconds."""
    t0 = time.time()
    payload = json.dumps({
        "model": "default_model",
        "messages": [{"role": "user", "content": "ping"}],
        "max_tokens": 1, "temperature": 0,
    }).encode()
    while time.time() - t0 < timeout:
        if proc.poll() is not None:
            raise RuntimeError(f"server exited early rc={proc.returncode}")
        try:
            req = urllib.request.Request(
                f"http://127.0.0.1:{PORT}/v1/chat/completions",
                data=payload, headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=10) as r:
                if r.status == 200:
                    json.loads(r.read())
                    return round(time.time() - t0, 1)
        except Exception:
            time.sleep(2)
    raise RuntimeError("server did not become ready in time")


def stop_server(proc, lf):
    try:
        proc.send_signal(signal.SIGINT)
        proc.wait(timeout=15)
    except Exception:
        proc.kill()
        proc.wait()
    finally:
        lf.close()


def server_request_count(log_path: Path) -> int:
    try:
        return log_path.read_text().count("POST /v1/chat/completions")
    except FileNotFoundError:
        return -1


# --------------------------------------------------------------------------- #
# lm-eval
# --------------------------------------------------------------------------- #
def run_lm_eval(task: str, out_dir: Path, limit: int, seed: int) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    model_args = (
        f"model=default_model,"
        f"base_url=http://127.0.0.1:{PORT}/v1/chat/completions,"
        f"num_concurrent=1,max_retries=2,tokenized_requests=False,"
        f"eos_string=<|im_end|>"
    )
    cmd = [
        str(LMEVAL), "--model", "local-chat-completions",
        "--model_args", model_args,
        "--tasks", task,
        "--apply_chat_template",
        "--fewshot_as_multiturn",
        "--gen_kwargs", f"temperature=0,do_sample=False,max_gen_toks={MAX_GEN_TOKS}",
        "--seed", str(seed),
        "--output_path", str(out_dir),
        "--log_samples",
    ]
    if limit:
        cmd += ["--limit", str(limit)]
    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True)
    wall = round(time.time() - t0, 1)
    if r.returncode != 0:
        print(r.stdout[-3000:])
        print(r.stderr[-3000:])
        raise RuntimeError(f"lm-eval failed for {task} rc={r.returncode}")
    res_files = sorted((out_dir / "default_model").glob("results_*.json"))
    data = json.loads(res_files[-1].read_text())
    return {"results": data["results"][task],
            "n_samples": data.get("n-samples", {}).get(task),
            "wall_seconds": wall}


METRIC_KEYS = {
    "gsm8k": ["exact_match,flexible-extract", "exact_match,strict-match"],
    "ifeval": ["prompt_level_strict_acc,none", "inst_level_strict_acc,none",
               "prompt_level_loose_acc,none", "inst_level_loose_acc,none"],
}


def ppl_same_harness(want_arms: list) -> dict:
    """wikitext-2 ppl for each arm on the *identical* P6.2 mlx harness (shared
    token array, non-overlapping 512-tok blocks) so the ppl->downstream comparison
    carries no cross-harness caveat. fp16 + mlx_awq_int4 are read from the P6.2
    result file when present; self_awq is computed here.
    """
    sys.path.insert(0, str(ROOT / "src"))
    from verify_p6_2_real_int4 import _wikitext_ids, _eval_ppl, PPL_SEQ_LEN, PPL_NUM_SAMPLES
    from mlx_lm.utils import load

    p62_path = RESULTS_DIR / "p6_2_awq_int4_real.json"
    p62 = json.loads(p62_path.read_text())["arms"] if p62_path.exists() else {}
    known = {
        "fp16": p62.get("fp16_mlx", {}).get("wikitext2_ppl", {}).get("perplexity"),
        "mlx_awq_int4": p62.get("mlx_awq_int4_g128", {}).get("wikitext2_ppl", {}).get("perplexity"),
    }
    all_arms = arms()
    out = {"harness": "P6.2 mlx non-overlapping 512-tok blocks, first 32 of "
                      "wikitext-2-raw-v1 test, shared token array"}
    base = known.get("fp16")
    for arm in want_arms:
        v = known.get(arm)
        if v is None:
            model, tok = load(str(all_arms[arm]))
            v = _eval_ppl(model, _wikitext_ids(tok, PPL_SEQ_LEN, PPL_NUM_SAMPLES))["perplexity"]
            del model
        out[arm] = {"wikitext2_ppl": round(v, 4),
                    "delta_vs_fp16": None if base is None else round(v - base, 4)}
    return out


def compute_deltas(out_arms: dict, tasks: list, want_arms: list) -> dict:
    """Per-arm, per-task {metric -> quantised - fp16} (stderr keys dropped)."""
    base = out_arms.get("fp16", {}).get("tasks", {})
    deltas = {}
    for arm in want_arms:
        if arm == "fp16" or arm not in out_arms:
            continue
        deltas[arm] = {}
        for task in tasks:
            if task not in base or task not in out_arms[arm]["tasks"]:
                continue
            bm = base[task].get("metrics")
            am = out_arms[arm]["tasks"][task].get("metrics")
            if not bm or not am:
                continue
            deltas[arm][task] = {
                k: round(am[k] - bm[k], 4)
                for k in bm if k in am and "stderr" not in k
            }
    return deltas


def pick_metrics(task: str, res: dict) -> dict:
    out = {}
    for k in METRIC_KEYS.get(task, []):
        if k in res:
            out[k] = res[k]
        stderr_k = k.split(",")[0] + "_stderr," + k.split(",")[1] if "," in k else None
        if stderr_k and stderr_k in res:
            out[stderr_k] = res[stderr_k]
    return out


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=DEFAULT_LIMIT,
                    help="examples per task per arm (0 = full set)")
    ap.add_argument("--tasks", default=",".join(DEFAULT_TASKS))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--arms", default="fp16,self_awq,mlx_awq_int4")
    ap.add_argument("--smoke", action="store_true",
                    help="limit=5, tasks=gsm8k, arms=fp16 -- plumbing check")
    ap.add_argument("--ppl-only", action="store_true",
                    help="just (re)compute same-harness wikitext-2 ppl and patch the result JSON")
    args = ap.parse_args()

    if args.ppl_only:
        want = [a.strip() for a in args.arms.split(",") if a.strip()]
        d = json.loads(JSON_PATH.read_text()) if JSON_PATH.exists() else {}
        d["perplexity_same_harness"] = ppl_same_harness(want)
        JSON_PATH.write_text(json.dumps(d, indent=2))
        print(json.dumps(d["perplexity_same_harness"], indent=2))
        return

    if args.smoke:
        args.limit, args.tasks, args.arms = 5, "gsm8k", "fp16"

    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    want_arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    all_arms = arms()
    for a in want_arms:
        if a not in all_arms:
            sys.exit(f"unknown arm {a}")
        if not all_arms[a].exists():
            sys.exit(f"arm {a}: missing model dir {all_arms[a]}")
    if not LMEVAL.exists():
        sys.exit(f"missing {LMEVAL} -- create the .venv-lmeval harness venv first")

    scratch = ROOT / "src" / "results" / "p6_6_lmeval_runs"
    if scratch.exists():
        shutil.rmtree(scratch)

    t_start = time.time()
    out = {
        "task": "支柱7 Bullet 3 -- downstream eval (GSM8K + IFEval) of self-built AWQ",
        "model": MODEL_ID,
        "harness": "EleutherAI lm-evaluation-harness (local-chat-completions) over mlx-lm server",
        "config": {
            "tasks": tasks, "limit": args.limit, "seed": args.seed,
            "decoding": "greedy (temperature=0, do_sample=False)",
            "max_gen_toks": MAX_GEN_TOKS, "num_concurrent": 1,
            "apply_chat_template": True, "fewshot_as_multiturn": True,
            "system_prompt": None,
            "note": "all arms identical except weights; MMLU/HellaSwag excluded "
                    "(loglikelihood tasks don't run over a chat API)",
        },
        "arms": {},
    }

    for arm in want_arms:
        mp = all_arms[arm]
        print(f"\n=== arm {arm} : {mp} ===", flush=True)
        log_path = scratch / f"{arm}_server.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        proc, lf = start_server(mp, log_path)
        arm_rec = {"model_path": str(mp), "tasks": {}}
        try:
            arm_rec["server_load_seconds"] = wait_ready(proc)
            print(f"  server ready in {arm_rec['server_load_seconds']}s", flush=True)
            for task in tasks:
                print(f"  running {task} (limit {args.limit}) ...", flush=True)
                od = scratch / arm / task
                try:
                    rr = run_lm_eval(task, od, args.limit, args.seed)
                except Exception as e:
                    print(f"    !! {task} failed: {e}", flush=True)
                    arm_rec["tasks"][task] = {"error": str(e)[:500]}
                    continue
                metrics = pick_metrics(task, rr["results"])
                arm_rec["tasks"][task] = {
                    "metrics": metrics,
                    "all_metrics": rr["results"],
                    "n_samples": rr["n_samples"],
                    "wall_seconds": rr["wall_seconds"],
                }
                hi = max((v for k, v in metrics.items() if "stderr" not in k), default=None)
                print(f"    -> {json.dumps(metrics)}  ({rr['wall_seconds']}s)", flush=True)
                if hi is not None and hi < 0.05:
                    print("    !! metric near zero -- check requests are reaching the server "
                          "(pitfall 3)", flush=True)
        finally:
            stop_server(proc, lf)
        arm_rec["server_post_requests_logged"] = server_request_count(log_path)
        out["arms"][arm] = arm_rec

    # deltas vs fp16
    if "fp16" in out["arms"]:
        out["deltas_vs_fp16"] = compute_deltas(out["arms"], tasks, want_arms)

    out["total_seconds"] = round(time.time() - t_start, 1)
    RESULTS_DIR.mkdir(exist_ok=True)
    JSON_PATH.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {JSON_PATH}  ({out['total_seconds']}s)")
    print(json.dumps(out.get("deltas_vs_fp16", {}), indent=2))


if __name__ == "__main__":
    main()
