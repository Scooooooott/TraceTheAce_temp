"""Frozen inference configuration shared by every LLM-feature consumer --
both scripts/precompute_llm_*.py (offline, rented-GPU labeling) and
submission_src/main.py (production vLLM inference) import these constants
rather than each hardcoding its own copy.

This exists specifically to prevent the offline-labeling run and the
production-inference run from silently drifting apart on sampling
behavior. experiments.md's 2026-07-24 validation entry found that
enable_thinking and repetition_penalty change output *quality/degeneracy*,
not just speed (e.g. a greedy-decoding repetition trap without
repetition_penalty) -- if the two sides used different values here, the
classifier would be trained on tags from one generation behavior and
scored against tags from another, on top of any train/production
*model*-mismatch risk.

Per-feature values that are NOT here on purpose: each feature module keeps
its own MAX_NEW_TOKENS (llm_strategy_tags.py: 768, llm_mastery_check.py:
768 by default but overridden to 1400 by its precompute script -- thinking
mode needs more room for that feature specifically, see experiments.md).
Only the sampling *method* is shared; output-length budget is legitimately
feature-specific.
"""

ENABLE_THINKING = True
REPETITION_PENALTY = 1.3

# Wall-clock safety net for submission_src/main.py's production LLM-scoring
# step ONLY -- see src.features.llm_annotate.score_combined_chunked, which
# is what actually uses this. NEVER applied to
# scripts/precompute_llm_annotations.py's offline run (not wall-clock
# constrained the way a graded submission is).
#
# 4.5h, not the full 6h container cap: the real test-set size (S_te) is an
# untested extrapolation (see gpu_rental/RUNBOOK.md) -- if it's bigger than
# expected, or real per-session throughput is worse than the rented-GPU
# smoke test found, this is what turns "container times out, whole
# submission scored zero, one of 3 weekly slots burned" into "some
# sessions fall back to FALLBACK_TAGS, submission still finishes and
# scores on the rest." 4.5h reserves 1.5h for everything that runs before
# and after this step (data load, session_stats/tiktoken pass, bge-m3
# embeddings if used, classical model predict, writing output) plus
# margin -- matches the "边缘" go/no-go threshold in gpu_rental/RUNBOOK.md.
LLM_TIME_BUDGET_SECONDS = 4.5 * 3600

# Sessions per score_combined_chunked() chunk -- same default as
# scripts/precompute_llm_annotations.py's --chunk-size for consistency,
# not because the two need to match.
LLM_CHUNK_SIZE = 300

# Input-length cap, shared by both backends' generate paths (see
# llm_backend.py's _generate_hf and make_vllm_batch_generate_fn's _truncate).
# 10240 rather than the old flat 8192: measured on 300 real sessions/responses
# (2026-07-24), prompt token length is p50=6374-6569/p95=8527-8702/
# max=9772-9834 across both features -- 8192 was this dev machine's 8GB-card
# convenience number, not a real constraint on the production A100 (32B-AWQ
# weights ~20GB, KV cache ~0.25MB/token, room for far more than 10240 even at
# meaningful batch sizes). 10240 covers the measured max with headroom while
# truncating only the genuine long tail, so fewer of the informative
# long-transcript sessions get cut.
PRODUCTION_MAX_PROMPT_TOKENS = 10240

# Local-smoke-test-only ceiling (HF backend, this dev machine's 8GB card) --
# deliberately smaller than PRODUCTION_MAX_PROMPT_TOKENS. This backend's job
# is validating the pipeline (data loading, JSON parsing, chunked
# checkpointing), not reproducing production-identical labels, so a lower
# local ceiling is an accepted, EXPLICIT inconsistency -- unlike the
# vLLM-path-had-no-truncation-at-all bug this replaces, this one is
# intentional and documented, not silent.
LOCAL_SMOKE_MAX_PROMPT_TOKENS = 8192

# vLLM's total context window: must exceed PRODUCTION_MAX_PROMPT_TOKENS by
# enough room for the longest realistic output. 14336 = 10240 +
# RETRY_MAX_NEW_TOKENS's largest value (llm_strategy_tags.py's 3072) +
# ~1000 buffer -- retries run at a HIGHER token budget than the primary
# pass (see both feature modules' RETRY_MAX_NEW_TOKENS), so this must cover
# the retry case, not just the primary MAX_NEW_TOKENS.
MAX_MODEL_LEN = 14336

# vLLM's automatic KV-block prefix cache. As of 2026-07-25 this is no
# longer a no-op for llm_mastery_check: both feature modules' build_prompt
# now start with build_shared_prefix() below (transcript BEFORE any
# per-task/per-response text), specifically so a session's llm_strategy_tags
# call and all its llm_mastery_check calls share a literal, position-0
# prefix and can hit this cache -- see src/features/llm_annotate.py, which
# is what actually submits them together in an order that gives the cache
# a chance to work (submission order/adjacency matters; see that module's
# docstring and experiments.md's 2026-07-25 entry on why a naive "just
# turn this on" wouldn't have been enough by itself).
ENABLE_PREFIX_CACHING = True

# The literal, byte-identical leading segment of EVERY LLM-feature prompt --
# llm_strategy_tags.build_prompt and llm_mastery_check.build_prompt both
# call build_shared_prefix() below rather than formatting the transcript
# themselves. This is the whole mechanism behind the prefix-cache benefit
# above: vLLM's cache only helps if two different tasks' prompts for the
# same session match token-for-token from position 0, which requires this
# text (and the transcript-formatting logic in build_shared_prefix) to
# never diverge between the two modules. Do NOT inline a transcript-first
# prompt anywhere else -- always call build_shared_prefix().
SHARED_SYSTEM_PROMPT_HEADER = (
    "You will be given a tutoring session transcript, followed by ONE "
    "specific analysis task. Read the entire transcript below carefully "
    "-- the task instructions, and what to reply with, come AFTER the "
    "transcript, not before.\n\nTranscript:\n"
)


def build_shared_prefix(transcript_df) -> str:
    """SHARED_SYSTEM_PROMPT_HEADER + the session's cleaned transcript,
    formatted identically to how both feature modules always formatted it
    (role-prefixed lines, in order) -- consolidated here purely so the two
    modules can never accidentally diverge on this formatting, since even a
    whitespace difference would break vLLM's exact-token-match prefix
    cache. Callers append their own task-specific suffix directly after
    this string's return value."""
    from src.features.session_stats import clean_content

    content = clean_content(transcript_df["content"])
    lines = [f"{role.capitalize()}: {c}" for role, c in zip(transcript_df["role"], content)]
    return SHARED_SYSTEM_PROMPT_HEADER + "\n".join(lines) + "\n\n"


# PROMPT_VERSION lives on each feature module (llm_strategy_tags.PROMPT_VERSION,
# llm_mastery_check.PROMPT_VERSION), not here -- it must bump independently
# per module whenever that module's PROMPT_TEMPLATE/build_prompt changes,
# and living next to the template it versions makes that harder to forget.
