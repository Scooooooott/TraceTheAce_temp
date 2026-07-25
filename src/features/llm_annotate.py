"""Combined scheduling for llm_strategy_tags.py + llm_mastery_check.py's
production-scale (vLLM batch) calls -- the ONE place this logic lives,
shared by scripts/precompute_llm_annotations.py and submission_src/main.py
rather than duplicated between them.

Why this exists at all (2026-07-25): both feature modules' build_prompt now
start with src.features.llm_config.build_shared_prefix()'s output (the full
transcript, byte-identical between the two modules). That alone does
NOTHING for vLLM's automatic prefix cache -- the cache only helps if a
session's llm_strategy_tags request and all its llm_mastery_check requests
are actually scheduled close together, so their shared prefix's KV blocks
are still resident when the later requests need them. Two independent,
sequential batched calls (the pre-2026-07-25 shape: all strategy_tags
prompts in one big generate() call, then separately all mastery_check
prompts in another) give the scheduler no reason to keep a session's
transcript blocks around between the two calls. score_combined below
builds ONE flat prompt list ordered session-by-session (a session's
strategy_tags request immediately followed by all its mastery_check
requests) and issues it as a SINGLE generate_batch_fn call, which is the
only way to give vLLM's scheduler the adjacency it needs.

Caveat, not yet measured (see experiments.md's 2026-07-25 entry / the
gpu_rental smoke-test checklist): if a single generate() call's internal
batch size is large, vLLM may schedule several of a session's requests into
the same parallel prefill wave, computing the shared prefix redundantly
within that wave rather than sequentially reusing a cached copy. The actual
cache hit rate must be read off vLLM's own stats during the rented-GPU
smoke test, not assumed from this ordering alone.

Only handles the vLLM-batch-scale path -- local HF dev/smoke-testing
still calls each feature's own score_session/score_response one at a time
(see llm_strategy_tags.py/llm_mastery_check.py), no combining needed or
useful at that tiny scale.
"""

import time

from src.features import llm_mastery_check, llm_strategy_tags

# Generous enough to cover whichever feature's primary/retry budget is
# larger -- costs nothing extra for requests that stop naturally well
# under it (autoregressive decoding halts per-sequence at EOS regardless of
# this ceiling; it only matters for the rare non-converging tail). This is
# what makes it safe to run both features through one shared budget instead
# of needing vLLM's per-request max_tokens list support.
COMBINED_MAX_NEW_TOKENS = max(llm_strategy_tags.MAX_NEW_TOKENS, llm_mastery_check.MAX_NEW_TOKENS, 1400)
COMBINED_RETRY_MAX_NEW_TOKENS = max(llm_strategy_tags.RETRY_MAX_NEW_TOKENS, llm_mastery_check.RETRY_MAX_NEW_TOKENS)


def score_combined(
    generate_batch_fn,
    strategy_prompts: dict[str, str],
    mastery_prompts: dict[str, str],
    mastery_session_of: dict[str, str],
    max_new_tokens: int = COMBINED_MAX_NEW_TOKENS,
    retry_max_new_tokens: int | None = COMBINED_RETRY_MAX_NEW_TOKENS,
    on_generated=None,
) -> tuple[dict[str, dict], dict[str, dict]]:
    """strategy_prompts: session_id -> prompt, for every session that needs
    an llm_strategy_tags result this call.
    mastery_prompts: response_id -> prompt, for every response that needs
    an llm_mastery_check result this call.
    mastery_session_of: response_id -> session_id, used only to group
    mastery_prompts by session for interleaved submission order -- every
    key in mastery_prompts must have an entry here.
    on_generated: optional (kind, generated_text) -> None callback, invoked
    once per PRIMARY-pass (not retry-pass) generated text, in the same
    "strategy"/"mastery" kind vocabulary as the results dicts. Purely an
    observability hook (e.g. scripts/precompute_llm_annotations.py uses it
    to track output-token-length distribution per task) -- has no effect on
    scoring/caching behavior, and defaults to None (no-op) so existing
    callers (submission_src/main.py) are unaffected.

    Returns (strategy_results, mastery_results), each a dict from the
    original id to its coerced tags dict (FALLBACK_TAGS for anything that
    never parsed, including after retry). Every session_id present in
    EITHER strategy_prompts or mastery_session_of's values ends up with its
    requests adjacent in the single flat list this builds -- see module
    docstring for why that adjacency is the entire point."""
    mastery_by_session: dict[str, list[str]] = {}
    for response_id, session_id in mastery_session_of.items():
        if response_id in mastery_prompts:
            mastery_by_session.setdefault(session_id, []).append(response_id)

    ordered_session_ids = list(dict.fromkeys(list(strategy_prompts.keys()) + list(mastery_by_session.keys())))

    # (kind, id) alongside a flat prompts list, same index -- kind in
    # {"strategy", "mastery"}, id = session_id (strategy) or response_id
    # (mastery). Kept as parallel lists rather than a list of tuples so the
    # hot path (building/slicing prompts for the generate_batch_fn call)
    # stays plain list operations.
    requests: list[tuple[str, str]] = []
    prompts: list[str] = []
    for session_id in ordered_session_ids:
        if session_id in strategy_prompts:
            requests.append(("strategy", session_id))
            prompts.append(strategy_prompts[session_id])
        for response_id in mastery_by_session.get(session_id, []):
            requests.append(("mastery", response_id))
            prompts.append(mastery_prompts[response_id])

    strategy_results: dict[str, dict] = {}
    mastery_results: dict[str, dict] = {}

    def _extract(kind: str, text: str):
        return llm_strategy_tags._extract_tags(text) if kind == "strategy" else llm_mastery_check._extract_tags(text)

    def _coerce(kind: str, obj: dict) -> dict:
        return llm_strategy_tags._coerce_tags(obj) if kind == "strategy" else llm_mastery_check._coerce_tags(obj)

    def _fallback(kind: str) -> dict:
        return dict(llm_strategy_tags.FALLBACK_TAGS if kind == "strategy" else llm_mastery_check.FALLBACK_TAGS)

    def _store(kind: str, id_: str, tags: dict) -> None:
        (strategy_results if kind == "strategy" else mastery_results)[id_] = tags

    if not prompts:
        return strategy_results, mastery_results

    generated_texts = generate_batch_fn(prompts, max_new_tokens)
    if on_generated is not None:
        for (kind, _id), text in zip(requests, generated_texts):
            on_generated(kind, text)
    failed_indices = []
    for i, ((kind, id_), generated) in enumerate(zip(requests, generated_texts)):
        obj = _extract(kind, generated)
        if obj is None:
            failed_indices.append(i)
        else:
            _store(kind, id_, _coerce(kind, obj))

    if failed_indices and retry_max_new_tokens:
        retry_texts = generate_batch_fn([prompts[i] for i in failed_indices], retry_max_new_tokens)
        still_failed = 0
        for i, generated in zip(failed_indices, retry_texts):
            kind, id_ = requests[i]
            obj = _extract(kind, generated)
            if obj is None:
                _store(kind, id_, _fallback(kind))
                still_failed += 1
            else:
                _store(kind, id_, _coerce(kind, obj))
        print(
            f"score_combined: {len(failed_indices)}/{len(prompts)} needed retry, "
            f"{still_failed}/{len(failed_indices)} still failed after retry (fell back to FALLBACK_TAGS)"
        )
    else:
        for i in failed_indices:
            kind, id_ = requests[i]
            _store(kind, id_, _fallback(kind))

    return strategy_results, mastery_results


def score_combined_chunked(
    generate_batch_fn,
    strategy_prompts: dict[str, str],
    mastery_prompts: dict[str, str],
    mastery_session_of: dict[str, str],
    chunk_size: int = 300,
    time_budget_seconds: float | None = None,
    max_new_tokens: int = COMBINED_MAX_NEW_TOKENS,
    retry_max_new_tokens: int | None = COMBINED_RETRY_MAX_NEW_TOKENS,
    on_generated=None,
) -> tuple[dict[str, dict], dict[str, dict]]:
    """Chunked wrapper around score_combined, with an optional wall-clock
    time-budget circuit breaker -- used by submission_src/main.py, NOT by
    scripts/precompute_llm_annotations.py (the offline labeling run isn't
    wall-clock-constrained the way a graded submission is; forcing an early
    bailout there would silently produce inferior training labels for no
    benefit).

    Why this exists: the competition container's real test-set size is
    unknown ahead of time -- this repo's S_te estimate (~6,838 sessions) is
    an extrapolation from the training set's response/session ratio, not a
    measurement (see gpu_rental/RUNBOOK.md) -- and a container that runs
    past its 6h limit fails the ENTIRE submission (zero credit, and burns
    one of the 3 weekly full-submission slots). A submission that falls
    back to FALLBACK_TAGS for whatever's left and still finishes on time is
    categorically better than one that times out with nothing written.

    Sessions are processed in chunks of chunk_size (same default as
    scripts/precompute_llm_annotations.py's --chunk-size, for consistency,
    not because they need to match). Before each chunk (after the first),
    remaining time is extrapolated from (elapsed / sessions_done done so
    far) -- if time_budget_seconds is set and this projection would exceed
    it, every remaining not-yet-processed session's requests are skipped
    entirely (no further generate_batch_fn calls at all) and filled with
    each feature's own FALLBACK_TAGS instead. The clock starts when this
    function is first called, NOT when submission_src/main.py itself
    started -- by the time this runs, the traditional feature/embedding
    steps are already done, so time_budget_seconds should be set to
    whatever's left of the 6h cap for the LLM step specifically (see
    src.features.llm_config.LLM_TIME_BUDGET_SECONDS), not the full 6h.

    Prints only counts (sessions done, sessions falling back) if the
    breaker triggers -- never any test-set content, consistent with the
    competition's "must not log test-set content" rule."""
    mastery_by_session: dict[str, list[str]] = {}
    for response_id, session_id in mastery_session_of.items():
        if response_id in mastery_prompts:
            mastery_by_session.setdefault(session_id, []).append(response_id)
    ordered_session_ids = list(dict.fromkeys(list(strategy_prompts.keys()) + list(mastery_by_session.keys())))

    strategy_results: dict[str, dict] = {}
    mastery_results: dict[str, dict] = {}
    run_start = time.time()
    n_sessions_done = 0
    triggered = False

    for start in range(0, len(ordered_session_ids), chunk_size):
        if time_budget_seconds is not None and n_sessions_done > 0:
            elapsed = time.time() - run_start
            rate = n_sessions_done / elapsed
            remaining_sessions = len(ordered_session_ids) - n_sessions_done
            projected_total = elapsed + (remaining_sessions / rate if rate > 0 else float("inf"))
            if projected_total > time_budget_seconds:
                triggered = True
                break

        chunk_ids = ordered_session_ids[start : start + chunk_size]
        chunk_strategy = {sid: strategy_prompts[sid] for sid in chunk_ids if sid in strategy_prompts}
        chunk_mastery = {rid: mastery_prompts[rid] for sid in chunk_ids for rid in mastery_by_session.get(sid, [])}
        chunk_mastery_session_of = {rid: mastery_session_of[rid] for rid in chunk_mastery}

        chunk_strategy_results, chunk_mastery_results = score_combined(
            generate_batch_fn,
            chunk_strategy,
            chunk_mastery,
            chunk_mastery_session_of,
            max_new_tokens=max_new_tokens,
            retry_max_new_tokens=retry_max_new_tokens,
            on_generated=on_generated,
        )
        strategy_results.update(chunk_strategy_results)
        mastery_results.update(chunk_mastery_results)
        n_sessions_done += len(chunk_ids)

    if triggered:
        remaining_session_ids = ordered_session_ids[n_sessions_done:]
        n_strategy_fallback = 0
        n_mastery_fallback = 0
        for session_id in remaining_session_ids:
            if session_id in strategy_prompts:
                strategy_results[session_id] = dict(llm_strategy_tags.FALLBACK_TAGS)
                n_strategy_fallback += 1
            for response_id in mastery_by_session.get(session_id, []):
                mastery_results[response_id] = dict(llm_mastery_check.FALLBACK_TAGS)
                n_mastery_fallback += 1
        print(
            f"score_combined_chunked: time budget ({time_budget_seconds:.0f}s) projected to be exceeded "
            f"after {n_sessions_done}/{len(ordered_session_ids)} sessions -- "
            f"{n_strategy_fallback} strategy + {n_mastery_fallback} mastery results fell back to FALLBACK_TAGS"
        )

    return strategy_results, mastery_results
