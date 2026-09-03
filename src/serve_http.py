"""
Specter serving demo -- a stdlib HTTP wrapper around `SpecServer`.

The demo people actually look at is the *static* page `docs/site/index.html`: it
embeds several recorded real runs (`sample_runs.js`) and replays whichever one
the visitor picks, with no server. This module is (a) how those recordings are
produced (see SCENARIOS below) and (b) an optional live backend so you can
drive the same page against your own prompts.

No new dependencies (http.server + json + threading). Serves the lab page at `/`
and streams a live speculative-decoding run over Server-Sent Events at
`/generate`, with the same telemetry the terminal demo prints (src/demo/live.py):
streamed text, per-round gamma / accept length / rolling alpha / tok-s /
concurrency / circuit-breaker state.

    python -m src.serve_http                 # real Qwen2.5 0.5B/1.5B, loaded once
    python -m src.serve_http --fake          # deterministic FakeModel pair, no download
    python -m src.serve_http --capture docs/site/sample_run.json
                                             # every SCENARIOS entry -> sample_run.json
                                             # (default scenario) + sample_runs.js (all), exit

Endpoints:
    GET  /            docs/site/index.html
    GET  /health      {"ok", "backend", "ready"}
    GET  /sample      docs/site/sample_run.json  (for the no-backend replay)
    POST /generate    SSE stream; body:
                      {"prompt": str, "max_tokens": int<=160,
                       "spec": bool, "gammatune": bool, "breaker": bool,
                       "compare": bool, "demo_batch": bool}

Only one generation runs at a time (single Mac GPU); concurrent POSTs get 429.
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SITE = ROOT / "docs" / "site"
sys.path.insert(0, str(ROOT / "src"))

DEMO_PROMPTS = [
    "Explain in two sentences why the sky is blue.",
    "Write a Python function that returns the nth Fibonacci number.",
    "Give me three tips for staying focused while studying.",
    "Summarize the plot of Cinderella in one paragraph.",
]

# Named scenarios captured for the static replay page (docs/site/sample_runs.js).
# Order is the button order on the page. Each `body` is a /generate request dict.
SCENARIOS = {
    "batch": {
        "label": "Batch of 4",
        "caption": "Four mixed prompts under continuous batching — aggregate "
                   "throughput over a mixed workload lands at parity on this pair.",
        "body": {"demo_batch": True, "max_tokens": 128, "spec": True,
                 "breaker": True, "compare": True},
    },
    "codegen": {
        "label": "Code gen",
        "caption": "A single, highly predictable completion — long accepted "
                   "drafts give speculation its best case here.",
        "body": {"prompt": "Write a Python function that returns the nth "
                            "Fibonacci number, with a docstring and a doctest.",
                 "max_tokens": 128, "spec": True, "breaker": True, "compare": True},
    },
    "prose": {
        "label": "Open-ended prose",
        "caption": "An unpredictable creative continuation — short accepted "
                   "drafts, so the draft-model overhead is harder to earn back.",
        "body": {"prompt": "Continue this story in a surprising direction: the "
                            "lighthouse keeper had not seen another ship in "
                            "eleven years, until the morning the tide went out "
                            "and did not come back.",
                 "max_tokens": 128, "spec": True, "breaker": True, "compare": True},
    },
    "breaker": {
        "label": "Breaker trips",
        "caption": "Alpha floor pushed above this pair's normal acceptance rate "
                   "— watch the mode strip flip spec → degraded → "
                   "probe → spec as the breaker trips and re-probes.",
        "body": {"prompt": "Continue this story in a surprising direction: the "
                            "lighthouse keeper had not seen another ship in "
                            "eleven years, until the morning the tide went out "
                            "and did not come back.",
                 "max_tokens": 160, "spec": True, "breaker": True, "compare": True,
                 "alpha_floor": 0.6, "warmup_rounds": 2, "reprobe_every": 4},
    },
    "adaptive": {
        "label": "goodput controller",
        "caption": "Speculation length k is chosen every round from a calibrated "
                   "round-time model (P7 Track C), not a fixed gamma — watch "
                   "controller_k move as the batch fills.",
        "body": {"demo_batch": True, "max_tokens": 128, "spec": True,
                 "breaker": False, "compare": True, "controller": "goodput"},
    },
}
DEFAULT_SCENARIO = "batch"

_GEN_LOCK = threading.Lock()
_STATE = {"backend": "fake", "ready": False, "pair": None, "last": None}


# --------------------------------------------------------------------------- #
# model pair
# --------------------------------------------------------------------------- #
def load_pair(fake: bool):
    if fake:
        from spec_oracles import LengthOnlyCache, make_fake_pair
        draft, target, tok = make_fake_pair(phase_target=0.3)
        return draft, target, tok, LengthOnlyCache, "fake"
    from model_loader import (DRAFT_MODEL_NAME, TARGET_MODEL_NAME,
                              load_model_and_tokenizer)
    from spec_kv import _new_cache
    draft, _ = load_model_and_tokenizer(DRAFT_MODEL_NAME)
    target, tok = load_model_and_tokenizer(TARGET_MODEL_NAME)
    return draft, target, tok, _new_cache, "qwen"


# --------------------------------------------------------------------------- #
# one run -> a generator of SSE event dicts
# --------------------------------------------------------------------------- #
def run_stream(body: dict):
    """Yield (event, data) tuples for one /generate request."""
    from serving_loop import ServeConfig, SpecServer

    draft, target, tok, make_cache, backend = _STATE["pair"]
    max_tokens = int(min(max(8, body.get("max_tokens", 96)), 160))
    demo_batch = bool(body.get("demo_batch"))
    prompts = DEMO_PROMPTS if demo_batch else [str(body.get("prompt") or DEMO_PROMPTS[0])]

    def _cfg(spec: bool):
        return ServeConfig(
            gamma=int(body.get("gamma", 4)), temperature=0.0,
            max_new_tokens=max_tokens, max_active=4, make_cache=make_cache,
            spec_enabled=spec,
            gammatune_on=bool(body.get("gammatune")) and spec,
            breaker_on=bool(body.get("breaker", True)) and spec,
            alpha_floor=float(body.get("alpha_floor", 0.5)),
            warmup_rounds=int(body.get("warmup_rounds", 3)),
            reprobe_every=int(body.get("reprobe_every", 10)),
            controller=str(body.get("controller", "alpha_floor")),
            goodput_coeffs_path=str(ROOT / "results" / "p7_0_goodput_profile.json"),
            apply_chat_template=True,
        )

    def _one_pass(spec: bool, tag: str):
        server = SpecServer(draft, target, tok, _cfg(spec))
        for i, p in enumerate(prompts):
            server.submit(p, req_id=f"{tag}{i}", seed=1000 + i)
        sent = {}
        t0 = time.perf_counter()
        emitted = 0
        modes: dict = {}
        tps_series = []
        round_cap = 400
        while (server.active or server.pending) and server.round_index < round_cap:
            info = server.step()
            modes[info.mode] = modes.get(info.mode, 0) + 1
            emitted += info.emitted
            el = time.perf_counter() - t0
            tps = emitted / el if el else 0.0
            tps_series.append(round(tps, 2))
            texts = {}
            for s in server.active:
                txt = tok.decode(s.token_ids, skip_special_tokens=True) if s.token_ids else ""
                if txt != sent.get(s.req_id):
                    sent[s.req_id] = txt
                    texts[s.req_id] = txt
            for rid, r in server.results().items():
                if r.text != sent.get(rid):
                    sent[rid] = r.text
                    texts[rid] = r.text
            yield "round", {
                "pass": tag, "index": info.index, "mode": info.mode,
                "gamma": info.round_gamma, "rolling_alpha": round(info.rolling_alpha, 3),
                "realign_tax": round(info.realignment_overhead, 3),
                "n_active": info.n_active, "n_queued": info.n_queued,
                "emitted": info.emitted, "wall_ms": round(info.wall_s * 1e3, 1),
                "tok_per_s": round(tps, 2), "breaker": info.breaker_reason,
                "controller_k": info.controller_k,
                "texts": texts,
            }
        wall = time.perf_counter() - t0
        res = server.results()
        total = sum(len(r.token_ids) for r in res.values())
        accs = [a for r in res.values() for a in r.accept_lengths]
        summary = {
            "pass": tag, "backend": backend, "spec": spec,
            "prompts": len(prompts), "total_tokens": total,
            "wall_s": round(wall, 2),
            "agg_tok_per_s": round(total / wall, 2) if wall else 0.0,
            "n_rounds": sum(modes.values()), "mode_counts": modes,
            "mean_accept_len": round(sum(accs) / len(accs), 2) if accs else 0.0,
            "tps_series": tps_series,
            "final_texts": {rid: r.text for rid, r in res.items()},
        }
        return summary

    yield "start", {"backend": backend, "prompts": prompts, "max_tokens": max_tokens,
                    "spec": bool(body.get("spec", True)),
                    "compare": bool(body.get("compare"))}

    gen = _one_pass(bool(body.get("spec", True)), "A")
    main_summary = yield from gen
    yield "done", main_summary

    if body.get("compare"):
        base = yield from _one_pass(False, "B")
        speedup = (main_summary["agg_tok_per_s"] / base["agg_tok_per_s"]
                   if base["agg_tok_per_s"] else 0.0)
        yield "compare_done", {"speculative": main_summary, "baseline": base,
                               "speedup": round(speedup, 2)}


# --------------------------------------------------------------------------- #
# HTTP
# --------------------------------------------------------------------------- #
class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, format, *args):  # quiet; signature matches base
        del format, args

    def _send(self, code, body: bytes, ctype="application/json"):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/" or self.path.startswith("/index"):
            f = SITE / "index.html"
            if not f.exists():
                return self._send(404, b"index.html not built", "text/plain")
            return self._send(200, f.read_bytes(), "text/html; charset=utf-8")
        if self.path == "/health":
            return self._send(200, json.dumps(
                {"ok": True, "backend": _STATE["backend"], "ready": _STATE["ready"]}
            ).encode())
        if self.path == "/sample":
            f = SITE / "sample_run.json"
            if not f.exists():
                return self._send(404, b"{}", "application/json")
            return self._send(200, f.read_bytes(), "application/json")
        return self._send(404, b"not found", "text/plain")

    def do_POST(self):
        if self.path != "/generate":
            return self._send(404, b"not found", "text/plain")
        n = int(self.headers.get("Content-Length", 0))
        try:
            body = json.loads(self.rfile.read(n) or b"{}")
        except json.JSONDecodeError:
            return self._send(400, b'{"error":"bad json"}')

        if not _GEN_LOCK.acquire(blocking=False):
            return self._send(429, b'{"error":"busy - one run at a time"}')
        try:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close")   # no Content-Length: EOF ends the stream
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.close_connection = True
            captured = []
            for event, data in run_stream(body):
                captured.append({"event": event, "data": data})
                chunk = f"event: {event}\ndata: {json.dumps(data)}\n\n".encode()
                self.wfile.write(chunk)
                self.wfile.flush()
            _STATE["last"] = captured
        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            _GEN_LOCK.release()


# --------------------------------------------------------------------------- #
def _capture_events(body: dict):
    """Run one /generate body to completion, pruning tps_series as we go."""
    events = []
    for e, d in run_stream(body):
        if isinstance(d, dict):
            d = {k: v for k, v in d.items() if k != "tps_series"}
            for arm in ("speculative", "baseline"):
                if isinstance(d.get(arm), dict):
                    d[arm] = {k: v for k, v in d[arm].items() if k != "tps_series"}
        events.append({"event": e, "data": d})
    return events


def capture(path: Path):
    """Run every scenario in SCENARIOS and dump the SSE timelines.

    Writes:
      <path>                          JSON, DEFAULT_SCENARIO only -- for the
                                      server's /sample route and tooling
      <path.parent>/sample_runs.js    `window.SPECTER_RUNS = {key: {label,
                                      caption, captured, backend, events}}`
                                      -- every scenario, so the page replays
                                      with no server and no fetch (works from
                                      file:// and Pages)

    Drops the per-round `tps_series` (nothing reads it); keeps `final_texts`
    so the page can show the finished outputs before the replay starts.
    """
    runs = {}
    for key, scen in SCENARIOS.items():
        print(f"capturing '{key}' ({scen['label']}) ...")
        runs[key] = {
            "label": scen["label"], "caption": scen["caption"],
            "captured": time.strftime("%Y-%m-%d"), "backend": _STATE["backend"],
            "events": _capture_events(scen["body"]),
        }

    default = runs[DEFAULT_SCENARIO]
    blob = json.dumps({"captured": default["captured"], "backend": default["backend"],
                       "events": default["events"]}, separators=(",", ":"))
    path.write_text(blob)

    runs_js = path.parent / "sample_runs.js"
    runs_blob = json.dumps(runs, separators=(",", ":"))
    runs_js.write_text("window.SPECTER_RUNS = " + runs_blob + ";\n")
    n_events = sum(len(r["events"]) for r in runs.values())
    print(f"wrote {path} + {runs_js.name}  "
          f"({len(runs)} scenarios, {n_events} events, {len(runs_blob) // 1024} KB)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8137)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--fake", action="store_true", help="deterministic FakeModel pair")
    ap.add_argument("--capture", type=Path, help="every SCENARIOS entry -> replay JSON + JS, then exit")
    args = ap.parse_args()

    print(f"loading {'FakeModel' if args.fake else 'Qwen2.5 0.5B/1.5B'} pair ...")
    _STATE["pair"] = load_pair(args.fake)
    _STATE["backend"] = _STATE["pair"][4]
    _STATE["ready"] = True
    print("ready.")

    if args.capture:
        capture(args.capture)
        return

    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    url = f"http://{args.host}:{args.port}/"
    print(f"Specter lab -> {url}   (Ctrl-C to stop)")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nbye")
        srv.shutdown()


if __name__ == "__main__":
    main()
