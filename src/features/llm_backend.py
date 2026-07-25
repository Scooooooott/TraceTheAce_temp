"""Shared inference backend for LLM-derived features (llm_strategy_tags.py,
llm_mastery_check.py). Extracted once a second consumer needed the exact
same generate_fn contract -- prompt design and JSON-parsing/fallback logic
are feature-specific and stay in each feature's own module; only the
"load a model, produce a callable that turns prompts into generated text"
piece is shared.

Two backends, with two DIFFERENT call shapes -- this is deliberate, not an
inconsistency:

- `make_hf_generate_fn`: local dev, via `transformers`
  `AutoModelForCausalLM.generate()`. One prompt in, one string out
  (`generate_fn(prompt, max_new_tokens) -> str`) -- fine for local dev,
  where every score_session/score_response call already happens one row at
  a time anyway.
- `make_vllm_batch_generate_fn`: the real submission backend (bigger,
  competition-legal model, e.g. Qwen3-8B-AWQ -- Apache 2.0, already cached
  locally, no competition-mandated model list beyond "license permits
  open-source commercial use" per CLAUDE.md). Deliberately BATCHED
  (`generate_batch_fn(prompts, max_new_tokens) -> list[str]`), not the same
  one-at-a-time shape as the HF backend: vLLM's entire throughput advantage
  comes from its continuous-batching scheduler processing many requests
  concurrently. Calling `llm.generate([single_prompt], ...)` in a loop --
  mimicking make_hf_generate_fn's shape -- would forfeit nearly all of that
  advantage and make production inference infeasible within the
  competition's 6-hour budget (see experiments.md's LLM validation entries:
  even the small local model needed 5-60s/call; the full training corpus is
  22,821-35,072 rows depending on feature granularity, and the real test set
  at submission time is presumably a comparable order of magnitude -- only
  batched throughput has any chance of fitting that in 6 hours). Every
  caller that needs production-scale throughput (scripts/precompute_*.py,
  submission_src/main.py) must collect ALL its prompts up front and make
  ONE call, not loop over score_session/score_response's single-item
  functions per row -- see llm_strategy_tags.py::score_sessions_batch /
  llm_mastery_check.py::score_responses_batch for the batch-shaped scoring
  wrappers that pair with this.

  **This backend has never been run, and cannot be run on this dev
  machine**: vLLM itself won't install/run under WSL2 (upstream bug,
  vllm-project/vllm#37883) -- unlike `transformers`, which works fine
  locally as a slower fallback, there is no local dev/test loop for this
  code at all. It can only be exercised inside the actual competition
  Docker container, a rented non-WSL2 GPU box, or wherever else vLLM
  actually runs -- written as carefully and minimally as possible against
  the documented vLLM API, but unverified until run somewhere real.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from src.features.llm_config import (
    ENABLE_PREFIX_CACHING,
    LOCAL_SMOKE_MAX_PROMPT_TOKENS,
    MAX_MODEL_LEN,
    PRODUCTION_MAX_PROMPT_TOKENS,
)


def load_hf_model(model_name: str, load_in_4bit: bool = False):
    """load_in_4bit: local-dev-only path. `autoawq` (needed for transformers
    to load a pre-quantized AWQ checkpoint outside of vLLM) is deprecated
    upstream and incompatible with the installed transformers version
    (missing `PytorchGELUTanh`) -- not fixable by retrying. bitsandbytes
    NF4 quantizes the plain fp16 checkpoint on load instead, no separate
    quantized checkpoint needed. This does NOT affect the production path:
    vLLM serves the AWQ checkpoint with its own internal kernels and never
    imports the `awq` package (confirmed absent from
    tutoring-outcomes-runtime's resolved uv.lock). The one real consequence
    is that local (bnb NF4) and production (AWQ) use different 4-bit
    quantization algorithms on the same base weights -- flagged as an
    unverified-but-plausibly-small risk, to be spot-checked once the AWQ
    path is testable via the container smoke test."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.truncation_side = "left"
    quantization_config = (
        BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16)
        if load_in_4bit
        else None
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name, device_map="cuda", quantization_config=quantization_config
    )
    model.eval()
    return tokenizer, model


def _generate_hf(
    tokenizer, model, prompt: str, max_new_tokens: int, enable_thinking: bool = True, repetition_penalty: float = 1.0
) -> str:
    """transformers backend: one chat-templated forward+decode pass, greedy
    (do_sample=False -- deterministic given frozen weights). Returns only
    the newly generated text (the prompt tokens are sliced off), same
    contract a vLLM generate_fn must honor.

    enable_thinking: Qwen3's chat template exposes this as a hybrid-mode
    toggle. Empirically (scripts/validate_llm_strategy_tags.py, 2026-07-23)
    Qwen3-4B's default (thinking-enabled) reasoning for a multi-field
    rubric frequently ran past even MAX_NEW_TOKENS=768 without ever
    reaching the closing JSON block (0-50% parse success at n=6, verbose
    repetitive "but wait, let me reconsider" loops despite the prompt's
    "think briefly if useful") -- not a signal-quality problem, a
    length-budget one. Left as a parameter (not hardcoded) since this is a
    Qwen-template-specific kwarg, not something every future model/backend
    will have.

    repetition_penalty: default 1.0 (off), matching every backend run so
    far. Exposed as a knob after a real, distinct failure mode surfaced in
    llm_mastery_check.py's validation (2026-07-24, thinking enabled): on a
    connectivity-noisy transcript, greedy decoding got stuck verbatim
    repeating a single tutor line ("could you mind increasing your
    volume...") five times running out the token budget with no JSON ever
    produced -- a classic greedy-decoding repetition trap, distinct from the
    thinking-mode length problem above. Not yet validated as an actual fix,
    just wired through so it can be tested rather than hand-waved."""
    messages = [{"role": "user", "content": prompt}]
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
        truncation=True,
        max_length=LOCAL_SMOKE_MAX_PROMPT_TOKENS,
        # Unused by chat templates that don't define this Jinja variable
        # (Jinja doesn't error on an unreferenced context var) -- safe to
        # always pass rather than special-case Qwen3's template specifically.
        enable_thinking=enable_thinking,
    ).to(model.device)
    with torch.no_grad():
        out = model.generate(
            inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            repetition_penalty=repetition_penalty,
        )
    return tokenizer.decode(out[0, inputs.shape[1] :], skip_special_tokens=True)


def make_hf_generate_fn(tokenizer, model, enable_thinking: bool = True, repetition_penalty: float = 1.0):
    """Binds a loaded transformers tokenizer/model into the backend-agnostic
    `generate_fn(prompt, max_new_tokens) -> str` shape every LLM feature
    module expects -- the swap point for a future vLLM equivalent (see
    module docstring)."""

    def generate_fn(prompt: str, max_new_tokens: int) -> str:
        return _generate_hf(
            tokenizer, model, prompt, max_new_tokens, enable_thinking=enable_thinking, repetition_penalty=repetition_penalty
        )

    return generate_fn


def make_hf_batch_generate_fn(tokenizer, model, enable_thinking: bool = True, repetition_penalty: float = 1.0):
    """Adapts make_hf_generate_fn's one-at-a-time contract into the batch
    shape (`generate_batch_fn(prompts, max_new_tokens) -> list[str]`) that
    scripts/precompute_llm_*.py and score_sessions_batch/score_responses_batch
    expect -- by looping, NOT real batching. This exists solely so those
    precompute scripts can be smoke-tested end to end on this dev machine
    (data loading, prompt assembly, JSON parsing, chunked cache
    checkpointing) via `--backend hf` on a handful of items, since vLLM
    itself cannot run here at all (see module docstring). Never use this
    for a real production-scale run -- at that scale it has none of vLLM's
    continuous-batching throughput and would take the same 5-60s/call the
    local HF backend has always taken (see experiments.md), just called
    in a loop instead of one at a time by the caller."""
    generate_fn = make_hf_generate_fn(
        tokenizer, model, enable_thinking=enable_thinking, repetition_penalty=repetition_penalty
    )

    def generate_batch_fn(prompts: list[str], max_new_tokens: int) -> list[str]:
        return [generate_fn(p, max_new_tokens) for p in prompts]

    return generate_batch_fn


def load_vllm_model(model_path: str, **llm_kwargs):
    """model_path: a local directory only (bundled in the submission zip --
    no network access at inference time, same contract as
    embeddings.py::load_model's MODEL_LOCAL_PATH). NEVER a HuggingFace hub
    id here in submission_src/main.py.

    `quantization="awq"` matches an AWQ-quantized checkpoint (e.g.
    Qwen/Qwen3-8B-AWQ, already cached locally in this dev environment from
    earlier exploration) -- vLLM serves this with its own internal AWQ
    kernels and never imports the `awq`/`autoawq` Python package (confirmed
    absent from tutoring-outcomes-runtime's resolved uv.lock), so the
    `autoawq`-incompatibility issue that blocks `load_hf_model`'s
    AWQ-via-transformers path locally does not apply here.

    llm_kwargs: pass-through for whatever the actual container run needs to
    tune (e.g. `gpu_memory_utilization`) -- deliberately not hardcoded here
    since most of these have never been tuned against real hardware from
    this dev machine (see module docstring: this function has never been
    called successfully anywhere). `max_model_len`/`enable_prefix_caching`
    ARE given real defaults below (from src.features.llm_config) since
    those two specifically must stay consistent between the offline
    labeling run and submission_src/main.py's production run, same
    single-source-of-truth rationale as ENABLE_THINKING/REPETITION_PENALTY
    -- override via llm_kwargs only for a deliberate one-off experiment,
    not as the normal path."""
    from vllm import LLM

    llm_kwargs.setdefault("max_model_len", MAX_MODEL_LEN)
    llm_kwargs.setdefault("enable_prefix_caching", ENABLE_PREFIX_CACHING)
    return LLM(model=model_path, quantization="awq", dtype="float16", **llm_kwargs)


def make_vllm_batch_generate_fn(llm, enable_thinking: bool = True, repetition_penalty: float = 1.0):
    """Batched production backend -- see module docstring for why this is
    a `(prompts: list[str], max_new_tokens: int) -> list[str]` shape rather
    than `make_hf_generate_fn`'s one-at-a-time contract, and why batching
    isn't optional here.

    temperature=0 (not do_sample=False -- vLLM's SamplingParams uses a
    temperature knob, and temperature=0 triggers vLLM's own greedy-decoding
    path internally) for the same determinism rationale as the HF backend:
    frozen weights, no reason to introduce sampling variance.
    """
    from vllm import SamplingParams

    tokenizer = llm.get_tokenizer()
    sampling_params_cache: dict[int, "SamplingParams"] = {}

    def _truncate(formatted: str) -> str:
        """Left-truncates to PRODUCTION_MAX_PROMPT_TOKENS -- found missing
        entirely from this path during local verification (2026-07-24):
        ~10% of real transcripts' prompts exceed this length, and without
        this, this (untested-on-real-hardware) production backend would
        silently see a different, longer input than whatever gets validated
        locally via _generate_hf (which truncates to the separate, smaller
        LOCAL_SMOKE_MAX_PROMPT_TOKENS -- see llm_config.py for why these are
        deliberately two different constants, not one). Manual
        encode/slice/decode rather than relying on the vLLM-obtained
        tokenizer object's own truncation_side state, which isn't ours to
        assume is 'left' by default."""
        ids = tokenizer.encode(formatted, add_special_tokens=False)
        if len(ids) <= PRODUCTION_MAX_PROMPT_TOKENS:
            return formatted
        return tokenizer.decode(ids[-PRODUCTION_MAX_PROMPT_TOKENS:])

    def generate_batch_fn(prompts: list[str], max_new_tokens: int) -> list[str]:
        formatted = [
            _truncate(
                tokenizer.apply_chat_template(
                    [{"role": "user", "content": p}],
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=enable_thinking,
                )
            )
            for p in prompts
        ]
        if max_new_tokens not in sampling_params_cache:
            sampling_params_cache[max_new_tokens] = SamplingParams(
                max_tokens=max_new_tokens, temperature=0, repetition_penalty=repetition_penalty
            )
        outputs = llm.generate(formatted, sampling_params_cache[max_new_tokens])
        return [o.outputs[0].text for o in outputs]

    return generate_batch_fn
