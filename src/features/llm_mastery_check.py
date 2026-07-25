"""LLM mastery-verification feature: ask a causal LM to find every exchange
in the transcript specifically about ONE learning objective and verify
whether the student's final answer on each was actually correct -- not to
judge teaching style (llm_strategy_tags.py), and not to predict the quiz
outcome directly (llm_prompt.py's now-paused zero-shot route).

First-principles motivation (see the plan discussion this session): the
real submission confirmed the CV<->leaderboard gap traces to the model
being systematically overconfident on rare/unseen learning objectives --
four independent attempts at a better *statistical* fallback (LO target
encoding fallback value, explicit LO-count feature, true-zero-shot-vs-leaked
split, embedding-nearest-neighbor fallback) each moved the harsh-LO-holdout
stress test by only noise-level amounts. None of those mechanisms can ever
do better than "guess a smarter constant" for a genuinely novel LO, because
none of them read what actually happened in *this* session.

This feature is different in kind, not just a better proxy: whether a
student got a specific practice problem right during the session is a
direct, LO-frequency-agnostic signal -- verifying "is 51.5 x 100 = 5150
correct" doesn't require ever having seen this exact learning_objective_id
before, unlike every target-encoding-adjacent fix already tried. This is
also something TF-IDF/regex/embeddings structurally cannot do at all (no
bag-of-words or keyword-match method can verify arithmetic correctness),
so unlike llm_strategy_tags.py's tags -- which mostly have a same-construct
regex analogue already in session_stats.py -- there is no cheaper substitute
to compare this against.

response_id-level, NOT session-level (unlike llm_strategy_tags.py /
session_stats.py's convention): the whole point is to verify mastery of
ONE specific learning objective, and a session can touch several -- a
session-wide judgment would dilute exactly the signal this feature exists
to capture, the same way it caps every other session-level feature on the
~14% of sessions with mixed is_correct outcomes across response_ids. Cost
tradeoff, taken deliberately: ~35,072 calls (one per response_id) instead
of ~22,821 (one per session), since the prompt includes a specific
learning_objective and isn't deduplicable across a session's response_ids
the way llm_strategy_tags.py's session-only prompt is.

Backend (model loading + generate_fn) lives in src/features/llm_backend.py,
shared with llm_strategy_tags.py -- see that module's docstring for the
local-dev-vs-production-vLLM story. This module only owns the
prompt/JSON-parsing/fallback logic specific to the mastery-check feature --
build_prompt's transcript-formatting is delegated to
src.features.llm_config.build_shared_prefix, shared verbatim with
llm_strategy_tags.py so vLLM's prefix cache can treat both features' calls
for the same session as sharing a prefix (see that function's docstring,
and src/features/llm_annotate.py, which schedules the two features'
requests together to make that possible in practice).

Frozen pretrained weights, scored independently per response_id -- same
leak-proofing rationale as embeddings.py/llm_prompt.py/llm_strategy_tags.py:
nothing here is fit on this competition's labels.
"""

import json
import re
from pathlib import Path

import pandas as pd

MAX_NEW_TOKENS = 768

# One retry attempt, at a higher budget, for responses whose first pass
# produced no parseable JSON -- see score_responses_batch and
# llm_strategy_tags.py's identical mechanism (2026-07-24 finding there:
# thinking-mode generation can drift into unrelated text and never produce
# JSON at all, not just get cut mid-answer). No direct evidence of this
# specific failure on llm_mastery_check.py yet (n=8 local check: mean=562.5,
# max=774, all well under budget, no failures) -- added proactively since
# it's the same model/task family and the cost of having it is near-zero.
# 2800, not a smaller number: scripts/precompute_llm_mastery_check.py's CLI
# already overrides max_new_tokens to 1400 by default, so a retry budget
# must clear THAT, not the smaller module MAX_NEW_TOKENS above.
RETRY_MAX_NEW_TOKENS = 2800

# Bump whenever TASK_SUFFIX/build_prompt below changes meaning -- same
# rationale as llm_strategy_tags.py's PROMPT_VERSION.
# v2 (2026-07-25): restructured to put the transcript FIRST via
# src.features.llm_config.build_shared_prefix, matching
# llm_strategy_tags.py's identical v2 change -- see that module's
# PROMPT_VERSION comment and experiments.md's 2026-07-25 entry. Rubric/
# questions/JSON shape in TASK_SUFFIX below are unchanged from v1.
PROMPT_VERSION = "v2"

# Single source of truth for where the precomputed cache lives -- both
# scripts/precompute_llm_mastery_check.py (writes it) and src/train.py
# (reads it) import this rather than each hardcoding the path.
CACHE_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "processed" / "llm_mastery_check_cache.pkl"


def load_cache(path=CACHE_PATH) -> dict:
    """response_id -> tags dict. Missing file returns {} (every lookup then
    falls back to FALLBACK_TAGS via build_features's _apply_llm_mastery_check)
    rather than raising -- lets code that doesn't use this feature import
    and run cleanly on a machine where the precompute hasn't been run."""
    if not Path(path).exists():
        return {}
    return pd.read_pickle(path)

# Cap raised to 10, not 3 (2026-07-24 validation finding): unlike
# llm_strategy_tags.py's TAG_FIELDS, these are counts of real countable
# events, not an inherently-bounded ordinal rating -- a first small-scale
# run (thinking mode, Qwen3-4B) repeatedly and specifically cited 5 distinct,
# transcript-grounded exchanges (e.g. "2,500 equals something times 10...
# 30 times 100... 13 divided by 10... 1,250 divided by 10... 190 times 10...
# so that's 5 correct attempts") -- a cap of 3 was silently discarding real,
# content-grounded counts. 0 is still a real, sensible value when the LO
# genuinely wasn't practiced, which is also exactly the right fallback for a
# malformed/unparseable generation (claiming attempts happened without
# evidence would be a worse default than claiming none did).
COUNT_FIELDS = {"n_correct_attempts": (0, 10), "n_incorrect_attempts": (0, 10)}
BOOL_FIELDS = ("lo_addressed",)
FALLBACK_TAGS = {"n_correct_attempts": 0.0, "n_incorrect_attempts": 0.0, "lo_addressed": 0.0}

# Appended directly after src.features.llm_config.build_shared_prefix()'s
# output -- do NOT restate "Transcript:" or re-embed the transcript here,
# that lives in the shared prefix now. Content/wording otherwise unchanged
# from v1's PROMPT_TEMPLATE.
TASK_SUFFIX = """TASK: verify whether the student demonstrated CORRECT understanding of ONE specific learning objective during the tutoring session above -- you are not judging teaching style, and not predicting the outcome of any future quiz.

Learning objective: {learning_objective}

Find every exchange in the transcript above where the student attempts a question, problem, or check-for-understanding that is specifically about this learning objective (ignore exchanges about other topics, even if they appear in the same transcript). For each such exchange, verify for yourself whether the student's final answer (after any back-and-forth, hints, or correction) was mathematically/factually correct -- check the arithmetic or fact yourself rather than trusting the tutor's reaction alone.

Think briefly if useful, then reply with ONLY a single JSON object on the last line, matching exactly this shape:
{{"lo_addressed": <true/false>, "n_correct_attempts": <count of exchanges where the student's final answer was correct>, "n_incorrect_attempts": <count of exchanges where the student's final answer was wrong or never resolved>}}

If this learning objective was not clearly addressed in the transcript, reply {{"lo_addressed": false, "n_correct_attempts": 0, "n_incorrect_attempts": 0}}."""


def build_prompt(learning_objective: str, transcript_df) -> str:
    """transcript_df: role, content columns for one session, in original
    order -- see src.features.llm_config.build_shared_prefix, which keeps
    every row (including "background", unlike session_stats.py's
    tutor/student split) and preserves turn order, same rationale as
    llm_prompt.py's build_prompt: verifying a specific exchange needs the
    real conversational flow, not a bag of role-split text."""
    from src.features.llm_config import build_shared_prefix

    return build_shared_prefix(transcript_df) + TASK_SUFFIX.format(learning_objective=learning_objective)


_JSON_OBJECT_RE = re.compile(r"\{[^{}]*\}")


def _extract_tags(generated_text: str) -> dict | None:
    """Last well-formed {...} block that contains every expected key wins --
    same rationale as llm_strategy_tags.py's _extract_tags: the model may
    emit JSON-looking fragments inside its thinking trace before the real
    answer."""
    matches = _JSON_OBJECT_RE.findall(generated_text)
    for candidate in reversed(matches):
        try:
            obj = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if all(k in obj for k in COUNT_FIELDS) and all(k in obj for k in BOOL_FIELDS):
            return obj
    return None


def _coerce_tags(obj: dict) -> dict:
    out = {}
    for key, (lo, hi) in COUNT_FIELDS.items():
        try:
            out[key] = float(min(max(int(obj[key]), lo), hi))
        except (TypeError, ValueError):
            out[key] = FALLBACK_TAGS[key]
    for key in BOOL_FIELDS:
        out[key] = float(bool(obj.get(key, False)))
    return out


def score_response(generate_fn, prompt: str, max_new_tokens: int = MAX_NEW_TOKENS) -> dict:
    """One response_id -> one dict of numeric tag values, via a
    one-at-a-time `generate_fn(prompt, max_new_tokens) -> str`
    (make_hf_generate_fn's shape) -- local dev only. For production-scale
    scoring use score_responses_batch below instead (see
    llm_backend.py's module docstring for why looping this over a vLLM
    backend would forfeit almost all of its batching throughput -- this
    feature's response_id-level granularity makes that especially costly,
    since it needs ~1.5x llm_strategy_tags.py's call count).

    Falls back to FALLBACK_TAGS (never raises, never returns partial/NaN) if
    the model's output doesn't contain a parseable JSON object with all
    expected keys -- same graceful-degradation contract as
    llm_strategy_tags.py::score_session."""
    generated = generate_fn(prompt, max_new_tokens)
    obj = _extract_tags(generated)
    if obj is None:
        return dict(FALLBACK_TAGS)
    return _coerce_tags(obj)


def score_responses_batch(
    generate_batch_fn,
    prompts: list[str],
    max_new_tokens: int = MAX_NEW_TOKENS,
    retry_max_new_tokens: int | None = RETRY_MAX_NEW_TOKENS,
) -> list[dict]:
    """Many response_ids -> many dicts, via ONE call to a batched
    `generate_batch_fn(prompts, max_new_tokens) -> list[str]`
    (make_vllm_batch_generate_fn's shape) -- the production-scale path. Same
    per-item parsing/fallback logic as score_response, just applied across a
    whole batch's results at once.

    retry_max_new_tokens: see llm_strategy_tags.py::score_sessions_batch's
    identical parameter -- one retry at a higher budget for responses with
    no parseable JSON, defaulting to on for both this precompute-time path
    and submission_src/main.py's production call, for the same
    offline/production consistency reason. Pass None to disable."""
    generated_texts = generate_batch_fn(prompts, max_new_tokens)
    results: list[dict | None] = []
    failed_indices = []
    for i, generated in enumerate(generated_texts):
        obj = _extract_tags(generated)
        if obj is None:
            results.append(None)
            failed_indices.append(i)
        else:
            results.append(_coerce_tags(obj))

    if failed_indices and retry_max_new_tokens:
        retry_texts = generate_batch_fn([prompts[i] for i in failed_indices], retry_max_new_tokens)
        still_failed = 0
        for i, generated in zip(failed_indices, retry_texts):
            obj = _extract_tags(generated)
            if obj is None:
                results[i] = dict(FALLBACK_TAGS)
                still_failed += 1
            else:
                results[i] = _coerce_tags(obj)
        print(
            f"score_responses_batch: {len(failed_indices)}/{len(prompts)} needed retry, "
            f"{still_failed}/{len(failed_indices)} still failed after retry (fell back to FALLBACK_TAGS)"
        )
    else:
        for i in failed_indices:
            results[i] = dict(FALLBACK_TAGS)

    return results


def mastery_ratio(tags: dict) -> float:
    """Downstream aggregate: fraction of verified attempts that were
    correct, NaN when the LO was never addressed / no attempts were found
    (0/0) -- callers decide how to fill that NaN (e.g. a neutral value, or
    left for a model that handles missing values natively), not this
    function's job to guess one."""
    n_correct = tags["n_correct_attempts"]
    n_incorrect = tags["n_incorrect_attempts"]
    total = n_correct + n_incorrect
    if total == 0:
        return float("nan")
    return n_correct / total
