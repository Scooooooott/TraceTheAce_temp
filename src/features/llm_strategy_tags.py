"""LLM-derived tutoring-strategy tags: ask a causal LM to describe *observed
teaching behavior* in a session transcript (not to predict the outcome),
and turn its structured JSON answer into numeric feature columns.

Deliberately not a repeat of llm_prompt.py's Yes/No approach: that forced a
single-token result judgment and produced no signal (see experiments.md,
"Level 3 Phase 1" -- P(Yes) collapsed to a near-constant prior). Asking the
model to describe what the tutor/student did, with room to reason first
(thinking mode), is a different and easier task -- there is no dependence
on this file's usefulness being validated yet; that's what
scripts/validate_llm_strategy_tags.py is for.

Session-level, not response_id-level: a session's teaching behavior doesn't
change across the different learning objectives it may touch (same
convention as session_stats.py's compute_session_features), so this is one
LLM call per session, joined onto rows by session_id like every other
session-level feature. Known ceiling this shares with every other
session-level feature (see experiments.md): ~14% of sessions have mixed
is_correct outcomes across their response_ids, and a session-wide judgment
can't distinguish those rows. Accepted for now -- fixing it would require
localizing which utterances belong to which learning objective, which the
transcripts don't segment for us.

Frozen pretrained weights, scored independently per session -- same
leak-proofing rationale as embeddings.py and llm_prompt.py: nothing here is
fit on this competition's labels.

Backend (model loading + generate_fn) lives in src/features/llm_backend.py,
shared with llm_mastery_check.py -- see that module's docstring for the
local-dev-vs-production-vLLM story. This module only owns the
prompt/JSON-parsing/fallback logic specific to the strategy-tags feature --
build_prompt's transcript-formatting is delegated to
src.features.llm_config.build_shared_prefix, shared verbatim with
llm_mastery_check.py so vLLM's prefix cache can treat both features' calls
for the same session as sharing a prefix (see that function's docstring,
and src/features/llm_annotate.py, which is what actually schedules the two
features' requests together to make that possible in practice).
"""

import json
import re
from pathlib import Path

import pandas as pd

# Bumped from 768 (2026-07-24): real thinking=True output length on 8
# random sessions (Qwen3-8B) was mean=610/p95=1214/max=1535 tokens -- one of
# 8 ran to within 1 token of a 1536 cap, meaning the old 768 cap would have
# cut its thinking block before ever reaching the closing JSON for a
# non-trivial fraction of sessions (same failure mode already documented
# for llm_mastery_check.py, which is why that module's precompute script
# already overrides its own default to 1400). Raised further to 2048, not
# just 1536: re-running that exact session confirmed it produced NO
# parseable JSON at all even with a 1536-token budget -- pure drift into
# unrelated text (repetition_penalty-induced?), not an answer cut one
# token short. A higher cap costs nothing for the ~87% of sessions that
# stop naturally well under it; it only extends the compute spent on the
# rare drifting case, which is exactly where RETRY_MAX_NEW_TOKENS below
# also comes in. Still a small-n finding (n=8) -- re-check at scale during
# the actual rented-GPU run.
MAX_NEW_TOKENS = 2048

# One retry attempt, at a higher budget, for sessions whose first pass
# produced no parseable JSON -- see score_sessions_batch. Not a guarantee:
# FALLBACK_TAGS remains the final safety net if the retry also fails.
RETRY_MAX_NEW_TOKENS = 3072

# Bump whenever TASK_SUFFIX/build_prompt below changes meaning (not for
# formatting-only edits that don't change what's being asked). Baked into
# scripts/precompute_llm_annotations.py's per-run cache filename so a
# prompt change can never silently "resume" into a cache keyed for the old
# prompt's answers.
# v2 (2026-07-25): restructured to put the transcript FIRST via
# src.features.llm_config.build_shared_prefix, with this module's rubric
# moved into TASK_SUFFIX afterward -- was previously baked directly into a
# self-contained PROMPT_TEMPLATE with the transcript in the middle. Purely
# structural (the rubric/questions/JSON shape below are unchanged from v1);
# bumped anyway because it changes the literal prompt text a cache entry
# was keyed against. See experiments.md's 2026-07-25 entry: this exists so
# llm_mastery_check.py's calls for the same session can share this
# transcript as a literal vLLM prefix-cache hit, which requires this
# module's prompt to start with the identical text too.
PROMPT_VERSION = "v2"

# Single source of truth for where the precomputed cache lives -- both
# scripts/precompute_llm_strategy_tags.py (writes it) and src/train.py
# (reads it) import this rather than each hardcoding the path.
CACHE_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "processed" / "llm_strategy_tags_cache.pkl"


def load_cache(path=CACHE_PATH) -> dict:
    """session_id -> tags dict. Missing file returns {} (every lookup then
    falls back to FALLBACK_TAGS via build_features's _apply_llm_strategy_tags)
    rather than raising -- lets code that doesn't use this feature import
    and run cleanly on a machine where the precompute hasn't been run."""
    if not Path(path).exists():
        return {}
    return pd.read_pickle(path)

# Ordinal scores are 0-3 (0 = not observed, 3 = frequent/strong) so a failed
# or missing extraction can fall back to a defined neutral value without
# guessing a whole distribution -- same "fallback is semantically the
# genuinely-unknown value" rationale as lo_target_encoding.py's lo_log_count.
TAG_FIELDS = {
    "scaffolding_score": (0, 3),
    "answer_giveaway_score": (0, 3),
    "follow_up_depth_score": (0, 3),
    "student_confusion_score": (0, 3),
}
BOOL_FIELDS = ("self_correction_observed",)
FALLBACK_TAGS = {**{k: 1.0 for k in TAG_FIELDS}, "self_correction_observed": 0.0}

# Appended directly after src.features.llm_config.build_shared_prefix()'s
# output -- do NOT restate "Transcript:" or re-embed the transcript here,
# that lives in the shared prefix now. Content/wording otherwise unchanged
# from v1's PROMPT_TEMPLATE.
TASK_SUFFIX = """TASK: describe the TEACHING BEHAVIOR that occurred in the transcript above -- you are not predicting any quiz outcome.

Rate what you observed using these definitions (each score is 0-3, where 0 = not observed at all, 3 = frequent/strong):
- scaffolding_score: the tutor guides the student toward the answer with hints/questions rather than stating it outright.
- answer_giveaway_score: the tutor states the correct answer directly after the student struggles, instead of continuing to guide.
- follow_up_depth_score: the tutor follows up multiple times on the same point the student got wrong, rather than moving on after one attempt.
- student_confusion_score: the student's own words show confusion or uncertainty (e.g. "I don't get it", "I'm not sure").
- self_correction_observed: true if the student corrected their own mistake without the tutor pointing it out first, false otherwise.

Think briefly if useful, then reply with ONLY a single JSON object on the last line, matching exactly this shape:
{"scaffolding_score": <0-3>, "answer_giveaway_score": <0-3>, "follow_up_depth_score": <0-3>, "student_confusion_score": <0-3>, "self_correction_observed": <true/false>}"""


def build_prompt(transcript_df) -> str:
    from src.features.llm_config import build_shared_prefix

    return build_shared_prefix(transcript_df) + TASK_SUFFIX


_JSON_OBJECT_RE = re.compile(r"\{[^{}]*\}")


def _extract_tags(generated_text: str) -> dict | None:
    """Last well-formed {...} block that contains every expected key wins --
    the model may emit JSON-looking fragments inside its thinking trace
    before the real answer, so taking the *last* match rather than the first
    is deliberate."""
    matches = _JSON_OBJECT_RE.findall(generated_text)
    for candidate in reversed(matches):
        try:
            obj = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if all(k in obj for k in TAG_FIELDS) and all(k in obj for k in BOOL_FIELDS):
            return obj
    return None


def _coerce_tags(obj: dict) -> dict:
    out = {}
    for key, (lo, hi) in TAG_FIELDS.items():
        try:
            out[key] = float(min(max(int(obj[key]), lo), hi))
        except (TypeError, ValueError):
            out[key] = FALLBACK_TAGS[key]
    for key in BOOL_FIELDS:
        out[key] = float(bool(obj.get(key, False)))
    return out


def score_session(generate_fn, prompt: str, max_new_tokens: int = MAX_NEW_TOKENS) -> dict:
    """One session -> one dict of numeric tag values, via a one-at-a-time
    `generate_fn(prompt, max_new_tokens) -> str)` (make_hf_generate_fn's
    shape) -- local dev only. For production-scale scoring, use
    score_sessions_batch below instead: this function called in a loop over
    a vLLM backend would forfeit vLLM's batching throughput almost entirely
    (see llm_backend.py's module docstring).

    Falls back to FALLBACK_TAGS (never raises, never returns partial/NaN) if
    the model's output doesn't contain a parseable JSON object with all
    expected keys -- a production inference path must degrade gracefully,
    not crash a whole submission over one malformed generation."""
    generated = generate_fn(prompt, max_new_tokens)
    obj = _extract_tags(generated)
    if obj is None:
        return dict(FALLBACK_TAGS)
    return _coerce_tags(obj)


def score_sessions_batch(
    generate_batch_fn,
    prompts: list[str],
    max_new_tokens: int = MAX_NEW_TOKENS,
    retry_max_new_tokens: int | None = RETRY_MAX_NEW_TOKENS,
) -> list[dict]:
    """Many sessions -> many dicts, via ONE call to a batched
    `generate_batch_fn(prompts, max_new_tokens) -> list[str]`
    (make_vllm_batch_generate_fn's shape) -- the production-scale path.
    Same per-item parsing/fallback logic as score_session, just applied
    across a whole batch's results at once rather than one generate_fn call
    per session.

    retry_max_new_tokens: sessions with no parseable JSON on the first pass
    (2026-07-24 finding: this happens, and isn't always recoverable by a
    bigger max_new_tokens alone -- some sessions drift into unrelated text
    and never produce JSON) get ONE retry at this higher budget before
    falling back to FALLBACK_TAGS. Defaults to on (not None) for BOTH this
    precompute-time path and submission_src/main.py's production call --
    if only one side retried, that would reintroduce exactly the kind of
    offline/production inconsistency this module's callers have been
    trying to eliminate all session. Pass None to disable (single-pass,
    pre-2026-07-24 behavior)."""
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
            f"score_sessions_batch: {len(failed_indices)}/{len(prompts)} needed retry, "
            f"{still_failed}/{len(failed_indices)} still failed after retry (fell back to FALLBACK_TAGS)"
        )
    else:
        for i in failed_indices:
            results[i] = dict(FALLBACK_TAGS)

    return results
