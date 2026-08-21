"""Loads tokenizers and counts tokens. Local tokenizers, plus Gemini's and
Anthropic's live APIs - but only ever via their free count_tokens endpoints,
never generation.

Applies two machine-specific fixes documented for this environment:
- corporate SSL-inspection proxy -> route TLS through the OS trust store
- hf-xet hangs on some downloads -> disabled in favor of the plain HTTP path
"""

import os
import sys
import time

import truststore

truststore.inject_into_ssl()
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

import tiktoken  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

from .registry import ModelSpec  # noqa: E402

_tiktoken_cache = {}
_hf_cache = {}
_gemini_client = None
_anthropic_client = None

# Free-tier Gemini rate limits are tight (as low as ~15 requests/min on some
# models) - space calls out so a multi-hundred-question run doesn't just
# spend its whole budget on 429s.
GEMINI_MIN_INTERVAL_SECONDS = 4.5
_gemini_last_call = 0.0

# No published free-tier rate limit for messages.count_tokens - pace it
# conservatively anyway since it's a live network call under a loop.
ANTHROPIC_MIN_INTERVAL_SECONDS = 1.0
_anthropic_last_call = 0.0

# A full-corpus live-API run takes hours unattended, so it has to ride out
# transient network loss (a wifi drop or laptop sleep surfaces as
# `getaddrinfo failed`, which killed a Gemini run at 3687/5060 rows once).
# 8 attempts with the delay capped at 60s gives ~3 minutes of patience
# instead of the ~30s that pure doubling from 2s would allow.
MAX_RETRIES = 8
MAX_RETRY_DELAY_SECONDS = 60.0


class TokenizerUnavailable(RuntimeError):
    """Raised when a tokenizer can't be loaded (missing auth, network, etc.)."""


def _get_tiktoken(encoding_name):
    if encoding_name not in _tiktoken_cache:
        _tiktoken_cache[encoding_name] = tiktoken.get_encoding(encoding_name)
    return _tiktoken_cache[encoding_name]


def _get_hf_tokenizer(spec: ModelSpec):
    cache_key = (spec.ref, spec.revision)
    if cache_key in _hf_cache:
        return _hf_cache[cache_key]

    token = os.environ.get("HF_TOKEN")
    if spec.requires_auth and not token:
        raise TokenizerUnavailable(
            f"{spec.display_name} ({spec.ref}) is gated on Hugging Face. "
            f"Accept the license at https://huggingface.co/{spec.ref}, then set "
            f"HF_TOKEN (e.g. `setx HF_TOKEN <token>` on Windows, then restart your shell)."
        )
    try:
        # fix_mistral_regex silences/corrects a known incorrect-tokenization
        # regex bug on Mistral's tokenizers; harmlessly ignored by others.
        tokenizer = AutoTokenizer.from_pretrained(
            spec.ref, revision=spec.revision, token=token, fix_mistral_regex=True
        )
    except Exception as exc:
        raise TokenizerUnavailable(
            f"Failed to load tokenizer for {spec.display_name} ({spec.ref}): {exc}"
        ) from exc

    _hf_cache[cache_key] = tokenizer
    return tokenizer


def _get_gemini_client():
    global _gemini_client
    if _gemini_client is not None:
        return _gemini_client

    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise TokenizerUnavailable(
            "Gemini API needs an API key. Get a free one at "
            "https://aistudio.google.com/apikey, then set GEMINI_API_KEY "
            "(e.g. `setx GEMINI_API_KEY <key>` on Windows, then restart your shell)."
        )
    try:
        from google import genai
    except ImportError as exc:
        raise TokenizerUnavailable(
            "google-genai package not installed. Run: pip install google-genai"
        ) from exc

    _gemini_client = genai.Client(api_key=api_key)
    return _gemini_client


def _count_gemini_tokens(spec: ModelSpec, text: str, max_retries: int = MAX_RETRIES) -> int:
    global _gemini_last_call
    client = _get_gemini_client()

    delay = 2.0
    for attempt in range(max_retries):
        wait = GEMINI_MIN_INTERVAL_SECONDS - (time.monotonic() - _gemini_last_call)
        if wait > 0:
            time.sleep(wait)
        try:
            result = client.models.count_tokens(model=spec.ref, contents=text)
            _gemini_last_call = time.monotonic()
            return result.total_tokens
        except Exception as exc:
            _gemini_last_call = time.monotonic()
            if attempt == max_retries - 1:
                raise TokenizerUnavailable(
                    f"Gemini count_tokens failed for {spec.display_name} ({spec.ref}) "
                    f"after {max_retries} attempts: {exc}"
                ) from exc
            print(
                f"[{spec.id}] retry {attempt + 1}/{max_retries - 1} in {delay:.0f}s: {exc}",
                file=sys.stderr,
            )
            time.sleep(delay)
            delay = min(delay * 2, MAX_RETRY_DELAY_SECONDS)


def _get_anthropic_client():
    global _anthropic_client
    if _anthropic_client is not None:
        return _anthropic_client

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise TokenizerUnavailable(
            "Anthropic API needs an API key. Get one at "
            "https://console.anthropic.com, then set ANTHROPIC_API_KEY "
            "(e.g. `setx ANTHROPIC_API_KEY <key>` on Windows, then restart your shell)."
        )
    try:
        from anthropic import Anthropic
    except ImportError as exc:
        raise TokenizerUnavailable(
            "anthropic package not installed. Run: pip install anthropic"
        ) from exc

    _anthropic_client = Anthropic(api_key=api_key)
    return _anthropic_client


def _count_anthropic_tokens(spec: ModelSpec, text: str, max_retries: int = MAX_RETRIES) -> int:
    global _anthropic_last_call
    client = _get_anthropic_client()

    delay = 2.0
    for attempt in range(max_retries):
        wait = ANTHROPIC_MIN_INTERVAL_SECONDS - (time.monotonic() - _anthropic_last_call)
        if wait > 0:
            time.sleep(wait)
        try:
            result = client.messages.count_tokens(
                model=spec.ref, messages=[{"role": "user", "content": text}]
            )
            _anthropic_last_call = time.monotonic()
            return result.input_tokens
        except Exception as exc:
            _anthropic_last_call = time.monotonic()
            if attempt == max_retries - 1:
                raise TokenizerUnavailable(
                    f"Anthropic count_tokens failed for {spec.display_name} ({spec.ref}) "
                    f"after {max_retries} attempts: {exc}"
                ) from exc
            print(
                f"[{spec.id}] retry {attempt + 1}/{max_retries - 1} in {delay:.0f}s: {exc}",
                file=sys.stderr,
            )
            time.sleep(delay)
            delay = min(delay * 2, MAX_RETRY_DELAY_SECONDS)


def count_tokens(spec: ModelSpec, text: str) -> int:
    """Count tokens for one sentence under one model's tokenizer.

    Special tokens (BOS/EOS/etc.) are excluded so counts reflect the content
    only, comparable across tokenizers that add different amounts of overhead.
    Gemini and Anthropic have no local tokenizer to load - they're counted via
    their live APIs' free count_tokens endpoints instead (never generation,
    so no spend). Those endpoints wrap text in a one-message request, so
    their counts include a small fixed per-request overhead unlike the local
    tokenizers above.
    """
    if spec.kind == "tiktoken":
        return len(_get_tiktoken(spec.ref).encode(text))
    if spec.kind == "hf":
        tokenizer = _get_hf_tokenizer(spec)
        return len(tokenizer.encode(text, add_special_tokens=False))
    if spec.kind == "gemini_api":
        return _count_gemini_tokens(spec, text)
    if spec.kind == "anthropic_api":
        return _count_anthropic_tokens(spec, text)
    raise ValueError(f"Unknown tokenizer kind: {spec.kind!r}")
