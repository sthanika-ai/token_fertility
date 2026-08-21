"""Merges the per-model FLORES+ result dirs into one comparable table.

Two corrections are applied, without which the numbers aren't comparable
across backends:

1. Envelope overhead. Local tokenizers are called with
   add_special_tokens=False, so their counts are pure content. The Anthropic
   and Gemini backends have no local tokenizer - they count via
   messages.count_tokens / models.count_tokens, which wrap the text in a
   one-message request and therefore add a FIXED per-request overhead.
   Measured empirically (tokenize a sentence, then the same sentence repeated
   20x, and solve tokens = overhead + R*content): 8 tokens for Sonnet 4.5,
   7 for Opus 4.8. The overhead came out identical on English and Telugu
   probes, confirming it's script-independent rather than a tokenizer effect.
   Left uncorrected it inflates fertility AND compresses parity_vs_english
   (a constant added to both sides of a ratio pulls it toward 1) - which
   would flatter exactly the models that are worst on Indic scripts.

2. Corpus-level rather than per-sentence-mean parity. Averaging each
   sentence's own token ratio over-weights short sentences, where the ratio
   is noisiest. The corpus-level figure (total tokens in language / total
   tokens in English, over the same 1012 sentences) is what actually predicts
   a serving bill, so it's reported as parity_corpus alongside the
   per-sentence mean.

Cost columns (total_cost_usd, cost_usd_per_1m_chars, cost_penalty_vs_en) are
NOT a new measurement - flores_fertility.py already computed a per-sentence
est_input_cost_usd (via tokfertility.pricing, from a dated $/1M-token rate
card). This script doesn't just sum that column, though: it was computed
from each row's RAW token_count, before the envelope-overhead correction
this script applies everywhere else - for the three live-API models
(Gemini/Claude) that means it's charging for 2-8 tokens of fixed per-request
overhead that were never in the actual sentence. So cost is rebuilt here:
each row's true $/1M rate is recovered from est_input_cost_usd/token_count
(verified exactly recoverable - e.g. Gemini 3.6 Flash rows all reverse to
precisely $1.50/1M), then reapplied to the SAME envelope-corrected token
count fertility/parity already use, before summing to a corpus total. Only
6 of 13 tokenizers have an owner-published API price (GPT-4o, GPT-5, Gemini
3.6/2.5 Flash, Claude Sonnet 4.5/Opus 4.8); the other 7 are open-weight with
no single official serving price, so their cost columns are left null
rather than filled with a guessed self-hosting rate.

Usage:
    python flores_merge_report.py                        # table to stdout
    python flores_merge_report.py --out flores_summary    # + CSV/JSON
"""

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

# model_id -> fixed tokens the count_tokens envelope adds per request.
# 0 for every local tokenizer (add_special_tokens=False -> content only).
# Measure a new live-API model before adding it; don't guess.
ENVELOPE_OVERHEAD = {
    "claude-sonnet-4-5": 8,
    "claude-opus-4-8": 7,
    "gemini-3.6-flash": 2,
}

# Result dirs to merge, in the order their models should appear.
#
# flores_results_gated is deliberately NOT listed here even though the file
# still exists on disk: adding sarvam-30b required re-running the full local
# set (see project memory), and that rewrote flores_results/results.csv to
# also include gemma-2-9b/gemma-3-4b/llama-3.1-8b - the same rows
# flores_results_gated already had (verified byte-identical, 0 mismatched
# token_counts across all 5,060 sentences x 3 models). Listing both directories
# double-counts those three models' totals here - harmless to the ratios
# (fertility/parity/chars_per_token are unchanged by doubling both sides
# equally, and none of the three are priced), but it corrupts n_sentences
# and total_tokens/total_cost_usd, and would silently double any future cost
# column for these models. If flores_results_gated is ever regenerated
# independently again, re-verify byte-identity before dropping it again.
RESULT_DIRS = [
    "flores_results",
    "flores_results_sonnet45",
    "flores_results_opus48",
    "flores_results_gemini36",
]

LANG_ORDER = ["en", "hi", "mr", "te", "kn"]
LANG_LABEL = {"en": "English", "hi": "Hindi", "mr": "Marathi", "te": "Telugu", "kn": "Kannada"}


def load_rows(result_dirs):
    rows = []
    missing = []
    for d in result_dirs:
        path = Path(d) / "results.csv"
        if not path.exists():
            missing.append(d)
            continue
        with open(path, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                row["token_count"] = int(row["token_count"])
                row["word_count"] = int(row["word_count"])
                row["char_count"] = int(row["char_count"])
                # Open-weight models have no pricing row, so tokfertility.pricing
                # left this blank - csv.DictReader reads that back as "".
                row["est_input_cost_usd"] = (
                    float(row["est_input_cost_usd"]) if row.get("est_input_cost_usd") else None
                )
                rows.append(row)
    return rows, missing


def aggregate(rows):
    """Corpus-level totals per (model, lang), with envelope overhead removed."""
    totals = defaultdict(lambda: {"tokens": 0, "words": 0, "chars": 0, "n": 0, "cost_usd": 0.0, "priced": False})
    names = {}

    for row in rows:
        mid = row["model_id"]
        names[mid] = row["model_display_name"]
        overhead = ENVELOPE_OVERHEAD.get(mid, 0)
        # Guard: a sentence shorter than the envelope would go negative.
        tokens = max(row["token_count"] - overhead, 1)

        agg = totals[(mid, row["lang"])]
        agg["tokens"] += tokens
        agg["words"] += row["word_count"]
        agg["chars"] += row["char_count"]
        agg["n"] += 1
        # Pricing is per-model, not per-sentence, so this is all-or-nothing
        # across every row for a given model - either every sentence has a
        # cost or none do. Recover the model's actual $/1M rate from the
        # RAW (uncorrected) token_count rather than trust est_input_cost_usd
        # directly, then reapply it to the envelope-corrected `tokens` above -
        # otherwise Gemini/Claude get billed for overhead tokens that were
        # never part of the sentence.
        if row["est_input_cost_usd"] is not None and row["token_count"] > 0:
            rate_per_token = row["est_input_cost_usd"] / row["token_count"]
            agg["cost_usd"] += rate_per_token * tokens
            agg["priced"] = True

    return totals, names


def build_summary(totals, names):
    summary = []
    for (mid, lang), agg in totals.items():
        en = totals.get((mid, "en"))
        priced = agg["priced"]
        en_priced = en is not None and en["priced"]
        summary.append(
            {
                "model_id": mid,
                "model_display_name": names[mid],
                "lang": lang,
                "n_sentences": agg["n"],
                "envelope_overhead_removed": ENVELOPE_OVERHEAD.get(mid, 0),
                "total_tokens": agg["tokens"],
                "fertility_tokens_per_word": round(agg["tokens"] / agg["words"], 3),
                "chars_per_token": round(agg["chars"] / agg["tokens"], 3),
                "parity_corpus_vs_en": (
                    round(agg["tokens"] / en["tokens"], 3) if en and en["tokens"] else None
                ),
                # Cost to send this exact 1,012-sentence corpus as input, at
                # the rate card in pricing_2026-07-30.json. None = open-weight,
                # no owner-published API price - not zero cost, unknown cost.
                "total_cost_usd": round(agg["cost_usd"], 6) if priced else None,
                "cost_usd_per_1m_chars": (
                    round(agg["cost_usd"] / agg["chars"] * 1_000_000, 4) if priced and agg["chars"] else None
                ),
                "cost_penalty_vs_en": (
                    round(agg["cost_usd"] / en["cost_usd"], 3) if priced and en_priced and en["cost_usd"] else None
                ),
            }
        )
    summary.sort(key=lambda r: (r["model_id"], LANG_ORDER.index(r["lang"]) if r["lang"] in LANG_ORDER else 9))
    return summary


def print_matrix(summary, field, title, fmt="{:.2f}"):
    by_model = defaultdict(dict)
    names = {}
    for row in summary:
        by_model[row["model_id"]][row["lang"]] = row[field]
        names[row["model_id"]] = row["model_display_name"]

    langs = [l for l in LANG_ORDER if any(l in v for v in by_model.values())]
    # Rank by the mean over the Indic languages only - English is the baseline.
    indic = [l for l in langs if l != "en"]

    def indic_mean(mid):
        vals = [by_model[mid][l] for l in indic if by_model[mid].get(l) is not None]
        return sum(vals) / len(vals) if vals else float("inf")

    print(f"\n{title}")
    header = f"{'tokenizer':38s}" + "".join(f"{LANG_LABEL[l]:>10s}" for l in langs) + f"{'Indic avg':>11s}"
    print(header)
    print("-" * len(header))
    for mid in sorted(by_model, key=indic_mean):
        cells = ""
        for l in langs:
            v = by_model[mid].get(l)
            cells += f"{fmt.format(v):>10s}" if v is not None else f"{'-':>10s}"
        print(f"{names[mid][:37]:38s}{cells}{fmt.format(indic_mean(mid)):>11s}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dirs", default=None, help="Comma-separated result dirs (default: all five)")
    parser.add_argument("--out", default=None, help="Directory to write summary.csv/summary.json")
    args = parser.parse_args(argv)

    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    dirs = [d.strip() for d in args.dirs.split(",")] if args.dirs else RESULT_DIRS
    rows, missing = load_rows(dirs)
    if not rows:
        print("No results found - run flores_fertility.py first.", file=sys.stderr)
        return 1
    for d in missing:
        print(f"[pending] {d}/results.csv not found - that model is absent from this table.", file=sys.stderr)

    totals, names = aggregate(rows)
    summary = build_summary(totals, names)

    n_models = len({r["model_id"] for r in summary})
    n_sent = max(r["n_sentences"] for r in summary)
    print(f"FLORES+ devtest - {n_sent} sentences x 5 languages x {n_models} tokenizers")
    print("Envelope overhead removed from live-API models; see ENVELOPE_OVERHEAD in this file.")

    print_matrix(summary, "fertility_tokens_per_word", "FERTILITY (tokens per word - lower is better)")
    print_matrix(summary, "parity_corpus_vs_en", "PARITY vs ENGLISH (corpus-level token ratio - lower is better)")
    print_matrix(summary, "chars_per_token", "CHARS PER TOKEN (higher is better)")

    priced_summary = [r for r in summary if r["cost_usd_per_1m_chars"] is not None]
    if priced_summary:
        print_matrix(
            priced_summary, "cost_usd_per_1m_chars",
            "COST ($ per 1M input chars, 2026-07-30 rate card - lower is better; open-weight models omitted, no owner-published price)",
            fmt="{:.4f}",
        )
    else:
        print("\n[no priced models present in this result set - cost table skipped]", file=sys.stderr)

    uncorrected = [m for m in ENVELOPE_OVERHEAD if ENVELOPE_OVERHEAD[m] == 0 and any(r["model_id"] == m for r in summary)]
    if uncorrected:
        print(
            f"\n[warning] envelope overhead still 0 (unmeasured) for: {', '.join(uncorrected)}"
            " - their fertility is overstated and parity understated.",
            file=sys.stderr,
        )

    if args.out:
        out_dir = Path(args.out)
        out_dir.mkdir(parents=True, exist_ok=True)
        fields = list(summary[0])
        with open(out_dir / "summary.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(summary)
        with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f"\nWrote {len(summary)} summary rows to {out_dir}/summary.csv and summary.json")

    return 0


if __name__ == "__main__":
    sys.exit(main())
