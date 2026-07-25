"""Precompute src/features/llm_strategy_tags.py and llm_mastery_check.py's
LLM-derived annotation caches for the training corpus, via the vLLM
production backend, submitting both features' requests together through
src.features.llm_annotate.score_combined so a session's requests share a
literal transcript prefix (see llm_annotate.py's module docstring for why
this must be one combined script rather than two independent ones -- two
separate process invocations can't give vLLM's scheduler the request
adjacency its prefix cache needs).

Replaces the retired scripts/precompute_llm_strategy_tags.py and
scripts/precompute_llm_mastery_check.py (2026-07-25): those submitted each
feature's requests in two entirely separate batched calls, incompatible
with any prefix-cache benefit, on top of needing separate rented-GPU
scheduling. See experiments.md's 2026-07-25 entry for the full rationale.

Session-atomic checkpointing, not per-request: a session counts as "done"
only when EVERY task in --tasks has a cached result for it (llm_strategy_tags:
one result for the session_id; llm_mastery_check: one result for EVERY
response_id under that session). On resume, a session with ANY missing
piece is redone in full (both tasks, even an already-cached half) rather
than patching in just the missing piece -- per-request-granularity resume
would need to track which half of a still-in-flight session finished, a
real source of subtle bugs for a cost (occasionally re-paying one session's
already-done half) that's negligible against the whole corpus.

**Cannot be run on this dev machine** -- vLLM doesn't install/run under
WSL2 (see llm_backend.py's module docstring). Meant to run wherever a real
vLLM environment exists (a rented GPU box) -- NOT inside a graded
competition submission (inference-only, 6-hour cap; see CLAUDE.md).

Usage:
  uv run python -m scripts.precompute_llm_annotations /path/to/Qwen3-8B-AWQ --model-id qwen3-8b-awq
  uv run python -m scripts.precompute_llm_annotations Qwen/Qwen3-8B --model-id qwen3-8b-hf-smoke --backend hf --limit 10
  uv run python -m scripts.precompute_llm_annotations /path/to/model --model-id foo --tasks strategy
"""

import argparse
import time
from pathlib import Path

import pandas as pd

from src.data import load_training_data, load_transcript
from src.features.llm_annotate import COMBINED_MAX_NEW_TOKENS, COMBINED_RETRY_MAX_NEW_TOKENS, score_combined
from src.features.llm_backend import (
    load_hf_model,
    load_vllm_model,
    make_hf_batch_generate_fn,
    make_vllm_batch_generate_fn,
)
from src.features.llm_mastery_check import CACHE_PATH as MASTERY_CACHE_PATH
from src.features.llm_mastery_check import PROMPT_VERSION as MASTERY_PROMPT_VERSION
from src.features.llm_mastery_check import build_prompt as build_mastery_prompt
from src.features.llm_strategy_tags import CACHE_PATH as STRATEGY_CACHE_PATH
from src.features.llm_strategy_tags import PROMPT_VERSION as STRATEGY_PROMPT_VERSION
from src.features.llm_strategy_tags import build_prompt as build_strategy_prompt


def _cache_path(base_path: Path, model_id: str, prompt_version: str) -> Path:
    return base_path.with_name(f"{base_path.stem}__{model_id}__prompt-{prompt_version}{base_path.suffix}")


def _load_cache(path: Path) -> dict:
    return pd.read_pickle(path) if path.exists() else {}


def _save_cache(cache: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.to_pickle(cache, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", help="local directory of the production model, or an HF hub id when --backend hf")
    parser.add_argument("--model-id", required=True, help="filesystem-safe label, baked into both cache filenames")
    parser.add_argument("--strategy-prompt-version", default=STRATEGY_PROMPT_VERSION)
    parser.add_argument("--mastery-prompt-version", default=MASTERY_PROMPT_VERSION)
    parser.add_argument("--tasks", default="strategy,mastery", help="comma-separated subset of {strategy,mastery}")
    parser.add_argument("--backend", choices=["vllm", "hf"], default="vllm", help="'hf' is a local-smoke-test-only path")
    parser.add_argument("--limit", type=int, default=None, help="only process the first N not-yet-done sessions -- for smoke tests")
    parser.add_argument("--no-thinking", action="store_true")
    parser.add_argument("--repetition-penalty", type=float, default=1.3)
    parser.add_argument("--max-new-tokens", type=int, default=COMBINED_MAX_NEW_TOKENS)
    parser.add_argument("--retry-max-new-tokens", type=int, default=COMBINED_RETRY_MAX_NEW_TOKENS)
    parser.add_argument("--chunk-size", type=int, default=300, help="sessions per batch call, cache checkpointed after each")
    args = parser.parse_args()

    tasks = set(args.tasks.split(","))
    assert tasks <= {"strategy", "mastery"}, f"unknown task(s) in {tasks!r}"
    do_strategy, do_mastery = "strategy" in tasks, "mastery" in tasks

    strategy_cache_path = _cache_path(STRATEGY_CACHE_PATH, args.model_id, args.strategy_prompt_version)
    mastery_cache_path = _cache_path(MASTERY_CACHE_PATH, args.model_id, args.mastery_prompt_version)
    strategy_cache = _load_cache(strategy_cache_path)
    mastery_cache = _load_cache(mastery_cache_path)
    print(f"strategy cache: {strategy_cache_path} ({len(strategy_cache)} cached)")
    print(f"mastery cache:  {mastery_cache_path} ({len(mastery_cache)} cached)")

    df = load_training_data()
    responses_by_session: dict[str, list[tuple[str, str]]] = {}
    for row in df.itertuples(index=False):
        responses_by_session.setdefault(row.session_id, []).append((row.response_id, row.learning_objective))
    session_ids = list(responses_by_session.keys())

    def session_done(sid: str) -> bool:
        if do_strategy and sid not in strategy_cache:
            return False
        if do_mastery and any(rid not in mastery_cache for rid, _ in responses_by_session[sid]):
            return False
        return True

    todo = [sid for sid in session_ids if not session_done(sid)]
    print(f"{len(session_ids)} total sessions, {len(todo)} remaining (tasks={sorted(tasks)})")
    if args.limit is not None:
        todo = todo[: args.limit]
        print(f"--limit {args.limit}: processing only {len(todo)} sessions this run")
    if not todo:
        print("nothing to do")
        return

    print(f"loading {args.model_path} via {args.backend} backend (thinking={not args.no_thinking})...")
    load_start = time.time()
    if args.backend == "vllm":
        llm = load_vllm_model(args.model_path)
        generate_batch_fn = make_vllm_batch_generate_fn(
            llm, enable_thinking=not args.no_thinking, repetition_penalty=args.repetition_penalty
        )
        count_tokens = llm.get_tokenizer().encode
    else:
        tokenizer, model = load_hf_model(args.model_path, load_in_4bit=True)
        generate_batch_fn = make_hf_batch_generate_fn(
            tokenizer, model, enable_thinking=not args.no_thinking, repetition_penalty=args.repetition_penalty
        )
        count_tokens = tokenizer.encode
    print(f"model loaded in {time.time() - load_start:.1f}s")

    # Output-token-length distribution per task -- NOT recoverable after the
    # fact from the cache pickles (score_combined discards raw generated
    # text once a chunk's JSON is parsed), so this is the only point this
    # can be measured. Cumulative across the whole run, printed after every
    # chunk (see _summarize_lengths call below) -- for a --limit smoke run
    # (single chunk) this is just that chunk's distribution.
    output_token_lengths: dict[str, list[int]] = {"strategy": [], "mastery": []}

    def _on_generated(kind: str, text: str) -> None:
        output_token_lengths[kind].append(len(count_tokens(text, add_special_tokens=False)))

    def _summarize_lengths(label: str, lengths: list[int]) -> str:
        if not lengths:
            return f"{label}: n=0"
        s = sorted(lengths)
        n = len(s)
        p95 = s[min(n - 1, int(n * 0.95))]
        return f"{label}: n={n} mean={sum(s) / n:.0f} p50={s[n // 2]} p95={p95} max={s[-1]}"

    run_start = time.time()
    n_done_this_run = 0
    for start in range(0, len(todo), args.chunk_size):
        chunk_session_ids = todo[start : start + args.chunk_size]
        transcripts = {sid: load_transcript(sid) for sid in chunk_session_ids}

        strategy_prompts = {sid: build_strategy_prompt(transcripts[sid]) for sid in chunk_session_ids} if do_strategy else {}
        mastery_prompts: dict[str, str] = {}
        mastery_session_of: dict[str, str] = {}
        if do_mastery:
            for sid in chunk_session_ids:
                for response_id, lo in responses_by_session[sid]:
                    mastery_prompts[response_id] = build_mastery_prompt(lo, transcripts[sid])
                    mastery_session_of[response_id] = sid

        chunk_start = time.time()
        strategy_results, mastery_results = score_combined(
            generate_batch_fn,
            strategy_prompts,
            mastery_prompts,
            mastery_session_of,
            max_new_tokens=args.max_new_tokens,
            retry_max_new_tokens=args.retry_max_new_tokens or None,
            on_generated=_on_generated,
        )
        chunk_elapsed = time.time() - chunk_start

        strategy_cache.update(strategy_results)
        mastery_cache.update(mastery_results)
        _save_cache(strategy_cache, strategy_cache_path)
        _save_cache(mastery_cache, mastery_cache_path)

        n_requests = len(strategy_prompts) + len(mastery_prompts)
        n_done_this_run += len(chunk_session_ids)
        rate = n_done_this_run / (time.time() - run_start)
        remaining = len(todo) - (start + len(chunk_session_ids))
        eta_hours = (remaining / rate) / 3600 if rate > 0 else float("inf")
        print(
            f"chunk {start}-{start + len(chunk_session_ids)} sessions ({n_requests} requests): "
            f"{chunk_elapsed:.1f}s ({chunk_elapsed / len(chunk_session_ids):.2f}s/session), "
            f"strategy={len(strategy_cache)} mastery={len(mastery_cache)}, ETA {eta_hours:.1f}h"
        )
        print(
            "  output tokens (cumulative): "
            f"{_summarize_lengths('strategy', output_token_lengths['strategy'])}, "
            f"{_summarize_lengths('mastery', output_token_lengths['mastery'])}"
        )

    print(f"done. wrote {strategy_cache_path if do_strategy else '(strategy skipped)'}")
    print(f"      wrote {mastery_cache_path if do_mastery else '(mastery skipped)'}")


if __name__ == "__main__":
    main()
