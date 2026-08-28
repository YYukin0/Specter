"""P6.3 -- the terminal demo runs headless on the FakeModel pair and its three
toggles have the effect they advertise."""
import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from demo.live import DEMO_PROMPTS, run_demo  # noqa: E402
from serving_loop import ServeConfig, SpecServer  # noqa: E402
from spec_oracles import LengthOnlyCache, make_fake_pair  # noqa: E402

FAKE = dict(fake=True, live=False, n_prompts=5, max_new_tokens=32, max_active=3)


def test_demo_runs_headless_and_drains():
    r = run_demo(spec=True, gammatune=False, breaker=True, **FAKE)
    assert len(r.texts) == 5
    assert r.total_tokens > 0
    assert r.n_rounds == sum(r.mode_counts.values())
    assert set(r.mode_counts) <= {"spec", "degraded", "probe", "idle"}


def test_no_spec_toggle_is_all_degraded():
    r = run_demo(spec=False, gammatune=False, breaker=False, **FAKE)
    assert set(r.mode_counts) <= {"degraded", "idle"}
    assert "spec" not in r.mode_counts
    # plain target decoding: one token per round per sequence, so more rounds
    spec = run_demo(spec=True, gammatune=False, breaker=True, **FAKE)
    assert r.n_rounds > spec.n_rounds
    assert r.mean_accept_len == 0.0


def test_gammatune_toggle_moves_gamma():
    d, t, tok = make_fake_pair(phase_target=0.3)
    cfg = ServeConfig(gamma=4, temperature=1.0, max_new_tokens=60, max_active=3,
                      make_cache=LengthOnlyCache, gammatune_on=True, breaker_on=True,
                      warmup_rounds=3, gamma_min=1, gamma_max=10)
    srv = SpecServer(d, t, tok, cfg)
    for i in range(6):
        srv.submit(f"prompt number {i}", req_id=f"r{i}", seed=i)
    srv.run_until_idle()
    gammas = {ri.round_gamma for ri in srv.round_log if ri.mode != "idle"}
    assert len(gammas) > 1, "gammatune_on but gamma never moved"

    # off -> gamma is pinned
    cfg2 = ServeConfig(gamma=4, make_cache=LengthOnlyCache, gammatune_on=False,
                       max_new_tokens=60, max_active=3)
    srv2 = SpecServer(d, t, tok, cfg2)
    for i in range(6):
        srv2.submit(f"prompt number {i}", req_id=f"r{i}", seed=i)
    srv2.run_until_idle()
    assert {ri.round_gamma for ri in srv2.round_log if ri.mode != "idle"} == {4}


def test_live_render_emits_ansi():
    buf = io.StringIO()
    run_demo(spec=True, gammatune=False, breaker=True, fake=True, live=True,
             n_prompts=3, max_new_tokens=24, max_active=2, out=buf, frame_min_s=0.0)
    s = buf.getvalue()
    assert "\x1b[H" in s and "\x1b[2J" in s          # home + clear
    assert "\x1b[?25l" in s and "\x1b[?25h" in s     # hide + show cursor
    assert "speculative-decoding serving loop" in s


def test_demo_prompt_set_is_stable():
    assert len(DEMO_PROMPTS) == 6
    assert all(isinstance(p, str) and p for p in DEMO_PROMPTS)
