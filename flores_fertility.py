"""Tokenizer fertility on the FLORES+-derived corpus (English, Hindi, Telugu,
Kannada, Marathi) - reuses tokfertility.report wholesale rather than
reimplementing it like milu_fertility.py does, because FLORES+ is a genuine
parallel corpus (every sentence id is the same source sentence translated
into each language), so parity_vs_english and the cost columns are
meaningful here in a way they aren't for MILU's independent per-language
question banks.

Run flores_build_corpus.py first to produce flores_subset/corpus_flores_v1.json.

All 5 languages (en/hi/te/kn/mr) are reported, English included - it's the
baseline the Indic fertility numbers are read against, and it's tokenized
anyway to compute parity.

The 12 default models split into two very different cost profiles:
  --local (9 models) tokenize offline in minutes. Llama 3.1 8B, Gemma 2 9B,
  and Gemma 3 4B are gated on HF and need HF_TOKEN with the license accepted;
  the other six need no auth.
  The 3 live-API models (Claude Sonnet 4.5, Claude Opus 4.8, Gemini 3.6
  Flash) count via free rate-limited count_tokens endpoints - 5,060 calls
  each on this corpus, so ~1.4h per Claude model and ~6h for Gemini. Run
  them one at a time with --models; token counts flush to the cache after
  every call, so an interrupted run resumes where it stopped.

Usage:
    python flores_fertility.py --local                      # 9 local tokenizers, all 5 langs
    python flores_fertility.py --models sarvam-1 --langs hi,te
    python flores_fertility.py --models gemini-3.6-flash    # long live-API run
    python flores_fertility.py --list-models
"""

import argparse
import json
import sys
from pathlib import Path

from tokfertility import report
from tokfertility.cache import ResultCache
from tokfertility.pricing import default_pricing_path, load_pricing
from tokfertility.registry import BY_ID, REGISTRY

CORPUS_PATH_DEFAULT = "flores_subset/corpus_flores_v1.json"
DEFAULT_MODELS = [
    # Local tokenizers - fast, no network per sentence.
    "sarvam-1",
    "gemma-2-9b",
    "gemma-3-4b",
    "brahmic-131k",
    "mistral-nemo-tekken",
    "qwen3-8b",
    "llama-3.1-8b",
    "cl100k-base",
    "gpt-4o",
    # Live rate-limited count_tokens APIs - hours per model on a 1012-sentence
    # corpus. Run these separately via --models rather than in a single pass.
    "claude-sonnet-4-5",
    "claude-opus-4-8",
    "gemini-3.6-flash",
]
LOCAL_MODELS = [m for m in DEFAULT_MODELS if not m.startswith(("claude-", "gemini-"))]


def load_corpus(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", default=None, help=f"Comma-separated model ids (default: all 12 - {','.join(DEFAULT_MODELS)})")
    parser.add_argument("--local", action="store_true", help="Shorthand for the 9 local (non-live-API) models")
    parser.add_argument("--langs", default=None, help="Comma-separated of hi,te,kn,mr (default: all four)")
    parser.add_argument("--corpus", default=CORPUS_PATH_DEFAULT, help=f"Path to FLORES+ corpus JSON (default: {CORPUS_PATH_DEFAULT})")
    parser.add_argument("--pricing", default=None, help="Path to a pricing JSON (default: bundled pricing_2026-07-30.json)")
    parser.add_argument("--cache-dir", default="cache", help="Directory for the token-count cache (default: ./cache)")
    parser.add_argument("--out", default="flores_results", help="Directory to write results.csv/results.json (default: ./flores_results)")
    parser.add_argument("--list-models", action="store_true", help="List the model registry and exit")
    args = parser.parse_args(argv)

    if args.list_models:
        for spec in REGISTRY:
            auth = " [requires HF_TOKEN]" if spec.requires_auth else ""
            print(f"{spec.id:24s} {spec.display_name}{auth}")
        return 0

    corpus_path = Path(args.corpus)
    if not corpus_path.exists():
        print(f"{corpus_path} not found - run flores_build_corpus.py first.", file=sys.stderr)
        return 1
    corpus = load_corpus(corpus_path)
    price_table = load_pricing(args.pricing or default_pricing_path())

    if args.models:
        model_ids = [m.strip() for m in args.models.split(",")]
    else:
        model_ids = LOCAL_MODELS if args.local else DEFAULT_MODELS
    unknown = [m for m in model_ids if m not in BY_ID]
    if unknown:
        print(f"Unknown model id(s): {', '.join(unknown)}. Run --list-models to see valid ids.", file=sys.stderr)
        return 1
    model_specs = [BY_ID[m] for m in model_ids]

    # English included as a reportable language, not just the parity
    # denominator - its own fertility/chars-per-token is the baseline the
    # Indic numbers are read against. Costs nothing extra: build_rows already
    # tokenizes every English sentence for parity and hits the same cache key.
    all_langs = list(corpus["languages"])
    if args.langs:
        langs = [l.strip() for l in args.langs.split(",")]
        unknown_langs = [l for l in langs if l not in all_langs]
        if unknown_langs:
            print(f"Unknown language code(s): {', '.join(unknown_langs)}. Corpus has: {', '.join(all_langs)}.", file=sys.stderr)
            return 1
    else:
        langs = all_langs

    with ResultCache(args.cache_dir) as cache:
        rows, skipped = report.build_rows(corpus, model_specs, langs, price_table, cache)

    if skipped:
        for model_id, message in skipped.items():
            print(f"[skipped] {model_id}: {message}", file=sys.stderr)

    if not rows:
        print("No results produced - all requested models were skipped.", file=sys.stderr)
        return 1

    report.write_csv(rows, f"{args.out}/results.csv")
    report.write_json(rows, f"{args.out}/results.json")

    summary = report.summarize(rows)
    report.print_summary(summary)
    print(f"\nWrote {len(rows)} rows to {args.out}/results.csv and {args.out}/results.json")

    return 0


if __name__ == "__main__":
    sys.exit(main())
