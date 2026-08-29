// Placeholder for the Bullet 2 cloud benchmark (vLLM + EAGLE3 on a rented
// A100/RTX 4090). Not yet run -- see TASKS.md 支柱7 Bullet 2 and
// notes/cloud-bullet2-execution-plan_2026-08-29.md for the full plan.
//
// When the run is done, replace this whole object with the real one written
// by `python -m src.cloud_bench.orchestrate --to-demo-js` (see that script's
// --to-demo-js flag) and flip status to "ready". The page below only renders
// a chart when status === "ready"; until then it shows this note instead, so
// the site never displays fabricated numbers.
window.SPECTER_CLOUD_BENCH = {
  status: "pending",
  note: "Not yet run. ~$15 one-time vLLM + EAGLE3 speculative-decoding " +
        "benchmark on a rented A100 or RTX 4090, scanning concurrency to " +
        "find the break-even point against a Mac-local implementation.",
  target_model: "meta-llama/Llama-3.1-8B-Instruct",
  draft_model: "yuhuili/EAGLE3-LLaMA3.1-Instruct-8B",
  locked_config: {
    num_speculative_tokens: 3,
    temperature: 0,
    top_p: 1.0,
    output_len: 1024,
    dataset: "gsm8k (full, 1319)",
    concurrencies: [1, 4, 16, 32, 64],
  },
  arms: [], // ready-shape: [{name, points:[{concurrency, speedup, ttft_p99_ms, tpot_p99_ms}]}]
};
