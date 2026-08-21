"""Frozen registry of tokenizers under benchmark.

To add or remove a model: edit REGISTRY below and add a matching row to a
pricing table under data/. Nothing elsewhere needs to change.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelSpec:
    id: str  # short id - used in results, cache keys, and pricing.json
    display_name: str
    kind: str  # "tiktoken" | "hf" | "gemini_api" | "anthropic_api"
    ref: str  # tiktoken encoding name, HF repo id, or Gemini API model id
    requires_auth: bool = False
    notes: str = ""
    # Pinned HF commit SHA. Tokenizer output must not drift when an upstream
    # repo is updated, so every "hf" kind pins one. None for tiktoken kinds
    # (pinned by the tiktoken version in requirements.txt) and API kinds
    # (tokenizer lives server-side and cannot be pinned).
    revision: str | None = None


REGISTRY = [
    ModelSpec(
        id="gpt-4o",
        display_name="OpenAI GPT-4o",
        kind="tiktoken",
        ref="o200k_base",
        notes="tiktoken encoding o200k_base",
    ),
    ModelSpec(
        id="gpt-5",
        display_name="OpenAI GPT-5",
        kind="tiktoken",
        ref="o200k_base",
        notes="Shares the o200k_base encoding with gpt-4o as of 2026-07-30",
    ),
    ModelSpec(
        id="llama-3.1-8b",
        display_name="Meta Llama 3.1 8B",
        kind="hf",
        ref="meta-llama/Llama-3.1-8B",
        revision="d04e592bb4f6aa9cfee91e2e20afa771667e1d4b",
        requires_auth=True,
        notes="Gated on Hugging Face - accept the license, then set HF_TOKEN",
    ),
    ModelSpec(
        id="gemma-2-9b",
        display_name="Google Gemma 2 9B",
        kind="hf",
        ref="google/gemma-2-9b",
        revision="33c193028431c2fde6c6e51f29e6f17b60cbfac6",
        requires_auth=True,
        notes="Gated on Hugging Face - accept the license, then set HF_TOKEN",
    ),
    ModelSpec(
        id="qwen2.5-7b",
        display_name="Qwen2.5 7B",
        kind="hf",
        ref="Qwen/Qwen2.5-7B",
        revision="d149729398750b98c0af14eb82c78cfe92750796",
        notes="Ungated",
    ),
    ModelSpec(
        id="mistral-nemo-tekken",
        display_name="Mistral Nemo (Tekken tokenizer)",
        kind="hf",
        ref="mistralai/Mistral-Nemo-Instruct-2407",
        revision="04d8a90549d23fc6bd7f642064003592df51e9b3",
        notes="Ungated; uses Mistral's newer Tekken tokenizer",
    ),
    ModelSpec(
        id="sarvam-1",
        display_name="Sarvam-1",
        kind="hf",
        ref="sarvamai/sarvam-1",
        revision="e9607337286ddf496d4a2562b194e489dcf3feea",
        notes="Ungated; India-focused open model tuned for Indic scripts",
    ),
    ModelSpec(
        id="sarvam-30b",
        display_name="Sarvam-30B",
        kind="hf",
        ref="sarvamai/sarvam-30b",
        revision="071ae95e933605ca1104a6b4524a36a98488efa4",
        notes="Ungated; only the tokenizer is loaded, not the 30B weights. NOT related to sarvam-1's vocabulary despite the shared vendor name - confirmed via direct token-ID comparison on FLORES+ sentences to be byte-identical to google/gemma-3-4b-it's tokenizer for English/Hindi/Marathi, and more efficient than Gemma specifically on Telugu/Kannada (2.06 vs 2.36 tokens/word Indic-avg). Same 262,144 vocab size as Gemma 3. Shares this same tokenizer with sarvam-105b (see below) - so 'Sarvam' tokenizer identity varies per model size and must be checked per-model, never assumed from the vendor name.",
    ),
    ModelSpec(
        id="sarvam-m",
        display_name="Sarvam-M",
        kind="hf",
        ref="sarvamai/sarvam-m",
        revision="01534a53c46f2788e392dbb3d994e0fa8f04d3fd",
        notes="Ungated (Apache-2.0); 24B hybrid-reasoning model built on mistralai/Mistral-Small-3.1-24B-Base-2503 - unlike sarvam-1/sarvam-30b, tests whether Sarvam kept Mistral's original tokenizer or extended it for Indic scripts.",
    ),
    ModelSpec(
        id="sarvam-105b",
        display_name="Sarvam-105B",
        kind="hf",
        ref="sarvamai/sarvam-105b",
        revision="fb187764dbad56ed8b26742a802909e858e9bb72",
        notes="Ungated (Apache-2.0); 105B-total/~10B-active custom MoE (DeepSeek-style Multi-head Latent Attention, model_type sarvam_mla). Confirmed via direct tokenizer comparison to share sarvam-30b's exact 262,144-token vocab - byte-identical token ids, not a distinct tokenizer despite the size difference.",
    ),
    ModelSpec(
        id="gemma-3-4b",
        display_name="Google Gemma 3 4B",
        kind="hf",
        ref="google/gemma-3-4b-pt",
        revision="cc012e0a6d0787b4adcc0fa2c4da74402494554d",
        requires_auth=True,
        notes="Gated on Hugging Face - accept the license, then set HF_TOKEN. Tokenizer is shared across all Gemma 3 sizes.",
    ),
    ModelSpec(
        id="qwen3-8b",
        display_name="Qwen3 8B",
        kind="hf",
        ref="Qwen/Qwen3-8B",
        revision="b968826d9c46dd6066d109eabc6255188de91218",
        notes="Ungated",
    ),
    ModelSpec(
        id="brahmic-131k",
        display_name="BrahmicTokenizer-131K",
        kind="hf",
        ref="theschoolofai/BrahmicTokenizer-131K",
        revision="93df154cbc9dbf038a222c010d9b43906a8a72c3",
        notes="Ungated (Apache 2.0); Indic-capable drop-in replacement for o200k_base",
    ),
    ModelSpec(
        id="cl100k-base",
        display_name="OpenAI GPT-3.5 / GPT-4 (cl100k_base)",
        kind="tiktoken",
        ref="cl100k_base",
        notes="Legacy tiktoken encoding shared by gpt-3.5-turbo and classic gpt-4",
    ),
    ModelSpec(
        id="gemini-3.6-flash",
        display_name="Google Gemini 3.6 Flash",
        kind="gemini_api",
        ref="gemini-3.6-flash",
        requires_auth=True,
        notes="Live Gemini API, current as of 2026-08 - tokenized via the free count_tokens endpoint, no local weights. Flash (not Pro) chosen for its much higher free-tier daily request quota. Needs GEMINI_API_KEY.",
    ),
    ModelSpec(
        id="gemini-2.5-flash",
        display_name="Google Gemini 2.5 Flash",
        kind="gemini_api",
        ref="gemini-2.5-flash",
        requires_auth=True,
        notes="Live Gemini API - tokenized via the free count_tokens endpoint, no local weights. Flash (not Pro) chosen for its much higher free-tier daily request quota. Needs GEMINI_API_KEY.",
    ),
    ModelSpec(
        id="claude-opus-4-8",
        display_name="Claude Opus 4.8",
        kind="anthropic_api",
        ref="claude-opus-4-8",
        requires_auth=True,
        notes="Live Anthropic API, current tokenizer as of 2026-08 - tokenized via the free messages.count_tokens endpoint, no local weights. Needs ANTHROPIC_API_KEY.",
    ),
    ModelSpec(
        id="claude-sonnet-4-5",
        display_name="Claude Sonnet 4.5",
        kind="anthropic_api",
        ref="claude-sonnet-4-5-20250929",
        requires_auth=True,
        notes="Live Anthropic API. Both claude-3-5-sonnet-20241022 (retired 2025-10-28) and claude-3-haiku-20240307 (documented as active but 404s live) were tried first and are no longer reachable - confirmed via GET /v1/models that this is the oldest model this account can still call. Predates Sonnet 5's new tokenizer (~30% more tokens on the same text per Anthropic's migration notes), so it's a same-family old-vs-new tokenizer comparison against claude-opus-4-8. Tokenized via the free messages.count_tokens endpoint. Needs ANTHROPIC_API_KEY.",
    ),
]

BY_ID = {m.id: m for m in REGISTRY}
