"""
P5.1 -- non-stationary prompt streams for GammaTune robustness testing.

Reference: notes/project_plan_v9.md sec 7 P5.1 + sec 9.6 pitfall 9 + sec 13.

Pitfall 9: GammaTune's own Limitations section concedes that a history-based
controller degrades under adversarial / sharply shifting workloads, and that
gains are small when the draft/target pair is well matched (low variance). P5.1
builds a deliberately non-stationary stream and checks whether GammaTune still
helps or degrades as the paper warns.

Two segment "types":
  SEGMENT_A -- code / structured generation. Writing functions, completions,
    refactors, one-line shell, JSON output. Structured text is the high-alpha side
    (the draft model tracks the target well), so the accept length -- and thus the
    ideal gamma -- tends to run higher here.
  SEGMENT_B -- open-ended chat. Explanations, summaries, light creative writing.
    Free text is the lower-alpha side; ideal gamma tends to run lower.

Sequences (each a list of (segment_label, prompt)):
  A_to_B      -- one switch: does gamma follow alpha down after the task changes?
  A_to_B_to_A -- switch and switch back: can the controller recover / re-expand?
  ABAB        -- high-frequency alternation: the closest thing here to the
                 adversarial regime pitfall 9 describes.

Prompts are original, neutral-topic, and written in the style of the three code
prompts already in src/prompts.py. No copyrighted text.
"""

SEGMENT_A = [
    "Write a Python function `is_palindrome(s)` that ignores case and non-alphanumeric characters.",
    "Complete this function:\n\ndef running_total(nums):\n    # return a list where out[i] = sum(nums[:i+1])\n",
    "Refactor this into a single list comprehension:\n\nout = []\nfor x in range(20):\n    if x % 3 == 0:\n        out.append(x * x)\n",
    "Write a bash one-liner that prints the 5 largest files under the current directory.",
    "Write a Python function `merge_dicts(a, b)` that deep-merges two nested dicts, with b winning on conflicts.",
    "Output a JSON object describing a book with fields title, author, year, and a list of two tags.",
    "Write a regular expression that matches an ISO 8601 date like 2026-08-28, and a one-line Python snippet using it.",
    "Implement `def chunk(lst, n)` that splits a list into consecutive sublists of length n (last one may be shorter).",
]

SEGMENT_B = [
    "Explain in two or three sentences why the ocean appears blue.",
    "Summarize the water cycle for a ten-year-old in one short paragraph.",
    "What are three practical tips for keeping houseplants alive in a low-light apartment?",
    "Describe a quiet morning at a mountain lake, in four or five sentences.",
    "Explain the difference between weather and climate in plain language.",
    "Give a short, encouraging pep talk to someone starting a new job tomorrow.",
    "What makes sourdough bread different from regular sandwich bread? Answer briefly.",
    "Tell a two-sentence story about a lighthouse keeper who befriends a gull.",
]


def _labelled(seg_list, label):
    return [(label, p) for p in seg_list]


# Each sequence is a list of (label, prompt). The controller state is carried
# across every prompt in a sequence (gammatune_generate carry_state), so gamma
# never resets at a prompt boundary -- only the task type changes under it.
SEQUENCES = {
    "A_to_B": _labelled(SEGMENT_A, "A") + _labelled(SEGMENT_B, "B"),
    "A_to_B_to_A": (
        _labelled(SEGMENT_A, "A")
        + _labelled(SEGMENT_B, "B")
        + _labelled(SEGMENT_A, "A")
    ),
    "ABAB": [
        item
        for pair in zip(_labelled(SEGMENT_A, "A"), _labelled(SEGMENT_B, "B"))
        for item in pair
    ],
}


def switch_indices(sequence):
    """Prompt indices where the segment label differs from the previous prompt."""
    return [i for i in range(1, len(sequence)) if sequence[i][0] != sequence[i - 1][0]]
