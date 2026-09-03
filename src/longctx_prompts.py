"""
P7 Track E -- ~800-1000 token prompts for KV-cache-quant measurement.

KV quantization only matters once the cache is long: with a 64-position fp
residual window, a 40-token prompt never quantizes anything. These prompts pad a
real question with original neutral filler (same approach as
`goodput_profile._FILLER_UNIT`) so the target's KV cache is long enough that the
older, quantized region actually dominates attention.

The filler is deliberately dull, self-consistent, and topic-neutral; the real
task is the last line. Question set = SEGMENT_A[:4] + SEGMENT_B[:4] from
nonstationary_prompts. Tokenized length is asserted in
tests/test_kv_fakequant.py to sit in [700, 1100] for the Qwen2.5 tokenizer.
"""
from nonstationary_prompts import SEGMENT_A, SEGMENT_B

_FILLER = ("Consider the following notes on a fictional town's public transit "
           "system, compiled for planning purposes. The town runs four bus "
           "routes on a fixed weekday schedule, with reduced service on "
           "weekends and public holidays. Ridership is highest between seven "
           "and nine in the morning and again between five and seven in the "
           "evening. ") * 12

QUESTIONS = SEGMENT_A[:4] + SEGMENT_B[:4]

LONGCTX_PROMPTS = [f"{_FILLER}\n\nNow answer this, ignoring the notes above: {q}"
                   for q in QUESTIONS]
