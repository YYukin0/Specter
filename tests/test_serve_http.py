"""P6.8 -- the stdlib serving demo.

Hermetic: runs the real `ThreadingHTTPServer` on an ephemeral port with the
deterministic FakeModel pair (no download), and drives it over HTTP the way the
lab page does. Fake tok/s numbers are meaningless (FakeModel has no compute) --
these tests only assert the wire protocol and the control paths.
"""
import json
import sys
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import serve_http  # noqa: E402


@pytest.fixture(scope="module")
def server():
    serve_http._STATE["pair"] = serve_http.load_pair(fake=True)
    serve_http._STATE["backend"] = "fake"
    serve_http._STATE["ready"] = True
    srv = ThreadingHTTPServer(("127.0.0.1", 0), serve_http.Handler)
    port = srv.server_address[1]
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        srv.shutdown()
        t.join(timeout=5)


def _get(base, path):
    with urllib.request.urlopen(base + path, timeout=10) as r:
        return r.status, r.read()


def _sse(base, body, timeout=30):
    """POST /generate, parse the event stream into a list of (event, data)."""
    req = urllib.request.Request(
        base + "/generate", data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    out, buf = [], ""
    with urllib.request.urlopen(req, timeout=timeout) as r:
        assert r.headers.get("Content-Type") == "text/event-stream"
        for raw in r:
            buf += raw.decode()
            while "\n\n" in buf:
                block, buf = buf.split("\n\n", 1)
                ev = da = None
                for line in block.splitlines():
                    if line.startswith("event: "):
                        ev = line[7:]
                    elif line.startswith("data: "):
                        da = json.loads(line[6:])
                if ev is not None:
                    out.append((ev, da))
    return out


# --------------------------------------------------------------------------- #
def test_health_reports_fake_backend(server):
    status, body = _get(server, "/health")
    assert status == 200
    h = json.loads(body)
    assert h == {"ok": True, "backend": "fake", "ready": True}


def test_root_serves_the_lab_page(server):
    status, body = _get(server, "/")
    assert status == 200
    assert b"<title>Specter" in body
    assert b"/generate" in body  # the page knows the endpoint


def test_sample_route_is_present_or_404(server):
    # sample_run.json is committed, so this should be 200; tolerate 404 in a
    # checkout that hasn't captured one yet.
    status, body = _get(server, "/sample")
    assert status in (200, 404)
    if status == 200:
        doc = json.loads(body)
        assert isinstance(doc.get("events"), list) and doc["events"]


def test_generate_streams_start_rounds_done(server):
    evs = _sse(server, {"prompt": "Say hi.", "max_tokens": 24, "spec": True})
    kinds = [e for e, _ in evs]
    assert kinds[0] == "start"
    assert kinds[-1] == "done"
    assert "round" in kinds

    start = evs[0][1]
    assert start["backend"] == "fake"
    assert start["max_tokens"] == 24

    for _, d in [e for e in evs if e[0] == "round"]:
        assert d["index"] >= 0
        assert d["mode"] in {"spec", "degraded", "probe", "idle"}
        assert 0.0 <= d["rolling_alpha"] <= 1.0
        assert d["n_active"] >= 0 and d["n_queued"] >= 0
        assert isinstance(d["texts"], dict)

    done = evs[-1][1]
    assert done["total_tokens"] > 0
    assert done["n_rounds"] == sum(done["mode_counts"].values())
    assert done["mode_counts"], "no rounds ran"


def test_generate_compare_adds_baseline_and_speedup(server):
    evs = _sse(server, {"prompt": "Say hi.", "max_tokens": 24,
                        "spec": True, "compare": True})
    kinds = [e for e, _ in evs]
    assert kinds[-1] == "compare_done"
    assert kinds.count("done") == 1  # the speculative pass still reports its own

    cd = evs[-1][1]
    assert cd["speculative"]["spec"] is True
    assert cd["baseline"]["spec"] is False
    assert cd["speedup"] >= 0.0
    # baseline is plain target decoding -> no accepted-draft runs
    assert cd["baseline"]["mode_counts"].get("spec", 0) == 0


def test_max_tokens_is_clamped_at_both_ends(server):
    hi = _sse(server, {"prompt": "x", "max_tokens": 99999, "spec": True})
    assert hi[0][1]["max_tokens"] == 160
    lo = _sse(server, {"prompt": "x", "max_tokens": 1, "spec": True})
    assert lo[0][1]["max_tokens"] == 8


def test_demo_batch_runs_all_four_prompts(server):
    evs = _sse(server, {"demo_batch": True, "max_tokens": 16, "spec": True})
    assert evs[0][1]["prompts"] == serve_http.DEMO_PROMPTS
    assert evs[-1][1]["prompts"] == 4


def test_concurrent_generate_gets_429(server):
    # hold the generation lock the way an in-flight request would
    assert serve_http._GEN_LOCK.acquire(blocking=False)
    try:
        req = urllib.request.Request(
            server + "/generate", data=b'{"prompt":"x"}',
            headers={"Content-Type": "application/json"}, method="POST")
        with pytest.raises(urllib.error.HTTPError) as ei:
            urllib.request.urlopen(req, timeout=10)
        assert ei.value.code == 429
    finally:
        serve_http._GEN_LOCK.release()


def test_goodput_controller_scenario_drains_headless(server):
    # P7 Wrap: the "adaptive" scenario body (controller="goodput") must run to
    # completion on the fake pair and surface a non-(-1) controller_k per round.
    evs = _sse(server, {"demo_batch": True, "max_tokens": 32, "spec": True,
                        "breaker": False, "controller": "goodput"})
    kinds = [e for e, _ in evs]
    assert kinds[0] == "start" and kinds[-1] == "done"
    rounds = [d for e, d in evs if e == "round"]
    assert rounds, "no rounds ran"
    assert all("controller_k" in d for d in rounds)
    assert any(d["controller_k"] >= 0 for d in rounds)
    assert evs[-1][1]["total_tokens"] > 0


def test_unknown_routes_404(server):
    for path in ("/nope", "/generate"):  # /generate is POST-only
        with pytest.raises(urllib.error.HTTPError) as ei:
            _get(server, path)
        assert ei.value.code == 404


@pytest.mark.usefixtures("server")
def test_capture_writes_json_and_js_sidecar(tmp_path):
    # the static page loads the runs via <script src="sample_runs.js">, not
    # fetch, so capture must emit both files (the fixture populates
    # serve_http._STATE); <path> itself stays single-run for the /sample route
    out = tmp_path / "run.json"
    serve_http.capture(out)
    assert out.exists()
    runs_js = tmp_path / "sample_runs.js"
    assert runs_js.exists()

    payload = json.loads(out.read_text())
    assert [e["event"] for e in payload["events"][:1]] == ["start"]
    assert payload["events"][-1]["event"] == "compare_done"
    assert not any("tps_series" in e["data"] for e in payload["events"]
                   if isinstance(e["data"], dict))

    text = runs_js.read_text()
    assert text.startswith("window.SPECTER_RUNS = ")
    runs = json.loads(text[len("window.SPECTER_RUNS = "):].rstrip().rstrip(";"))
    assert set(runs) == set(serve_http.SCENARIOS)
    for key, scen in serve_http.SCENARIOS.items():
        r = runs[key]
        assert r["label"] == scen["label"]
        assert r["caption"] == scen["caption"]
        assert r["events"][0]["event"] == "start"
        assert r["events"][-1]["event"] == "compare_done"
        assert not any("tps_series" in e["data"] for e in r["events"]
                       if isinstance(e["data"], dict))


@pytest.mark.usefixtures("server")
def test_capture_scenarios_hit_max_tokens_cap_and_floor(tmp_path, monkeypatch):
    # smoke tests must exercise boundary values, not just shrunk params -- one
    # scenario body asks for more than the cap (160), one for less than the
    # floor (8), and capture() must still clamp + finish cleanly for both
    scenarios = {
        "cap": {"label": "Cap", "caption": "", "body":
                {"prompt": "x", "max_tokens": 99999, "spec": True, "compare": False}},
        "floor": {"label": "Floor", "caption": "", "body":
                  {"prompt": "x", "max_tokens": 1, "spec": True, "compare": False}},
    }
    monkeypatch.setattr(serve_http, "SCENARIOS", scenarios)
    monkeypatch.setattr(serve_http, "DEFAULT_SCENARIO", "cap")

    out = tmp_path / "run.json"
    serve_http.capture(out)

    text = (tmp_path / "sample_runs.js").read_text()
    runs = json.loads(text[len("window.SPECTER_RUNS = "):].rstrip().rstrip(";"))
    cap_start = next(e["data"] for e in runs["cap"]["events"] if e["event"] == "start")
    floor_start = next(e["data"] for e in runs["floor"]["events"] if e["event"] == "start")
    assert cap_start["max_tokens"] == 160
    assert floor_start["max_tokens"] == 8
