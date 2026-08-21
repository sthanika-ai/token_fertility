"""Builds the per-sentence result table and writes it out as CSV/JSON, plus
a console summary aggregated by model x language."""

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

from . import metrics, pricing
from .backends import TokenizerUnavailable, count_tokens
from .cache import ResultCache

ROW_FIELDS = [
    "model_id",
    "model_display_name",
    "lang",
    "sentence_id",
    "domain",
    "token_count",
    "word_count",
    "char_count",
    "fertility_tokens_per_word",
    "chars_per_token",
    "parity_vs_english",
    "est_input_cost_usd",
    "cost_usd_per_1m_chars",
    "cost_penalty_vs_english",
]


def build_rows(corpus, model_specs, langs, price_table, cache: ResultCache):
    """Returns (rows, skipped) where skipped is {model_id: error_message}
    for any model whose tokenizer couldn't be loaded (e.g. missing HF_TOKEN)."""
    corpus_version = corpus["version"]
    rows = []
    skipped = {}

    for spec in model_specs:
        price_row = price_table.get(spec.id, {})
        model_failed = False
        # Gemini/Anthropic counts come from rate-limited live API calls, so a
        # full corpus takes hours - flush the cache after every one and print
        # progress, so a kill/crash resumes instead of starting over and a
        # multi-hour run isn't silent. Local tokenizers are fast enough that
        # the end-of-run flush in ResultCache.__exit__ is fine.
        is_live_api = spec.kind in ("gemini_api", "anthropic_api")
        total = len(corpus["sentences"])

        # English token counts per sentence, needed for parity/cost-penalty.
        en_tokens = {}
        for i, sentence in enumerate(corpus["sentences"], start=1):
            sid = sentence["id"]
            cached = cache.get(corpus_version, spec.id, "en", sid)
            if cached is not None:
                en_tokens[sid] = cached
                continue
            try:
                count = count_tokens(spec, sentence["en"])
            except TokenizerUnavailable as exc:
                skipped[spec.id] = str(exc)
                model_failed = True
                break
            cache.set(corpus_version, spec.id, "en", sid, count)
            en_tokens[sid] = count
            if is_live_api:
                cache.flush()
                print(f"[{spec.id}] en {i}/{total}", file=sys.stderr)

        if model_failed:
            continue

        for lang in langs:
            for i, sentence in enumerate(corpus["sentences"], start=1):
                sid = sentence["id"]
                text = sentence[lang]

                cached = cache.get(corpus_version, spec.id, lang, sid)
                if cached is not None:
                    token_count = cached
                else:
                    try:
                        token_count = count_tokens(spec, text)
                    except TokenizerUnavailable as exc:
                        skipped[spec.id] = str(exc)
                        model_failed = True
                        break
                    cache.set(corpus_version, spec.id, lang, sid, token_count)
                    if is_live_api:
                        cache.flush()
                        print(f"[{spec.id}] {lang} {i}/{total}", file=sys.stderr)

                wc = metrics.word_count(text)
                cc = metrics.char_count(text)
                fert = metrics.fertility(token_count, text)
                cpt = metrics.chars_per_token(token_count, text)
                parity = metrics.parity_vs_english(token_count, en_tokens[sid])

                est_cost = pricing.estimate_input_cost(token_count, price_row)
                cost_1m_chars = pricing.cost_per_1m_chars(token_count, cc, price_row)
                en_cost = pricing.estimate_input_cost(en_tokens[sid], price_row)
                cost_penalty = (
                    est_cost / en_cost if est_cost is not None and en_cost else None
                )

                rows.append(
                    {
                        "model_id": spec.id,
                        "model_display_name": spec.display_name,
                        "lang": lang,
                        "sentence_id": sid,
                        "domain": sentence.get("domain", ""),
                        "token_count": token_count,
                        "word_count": wc,
                        "char_count": cc,
                        "fertility_tokens_per_word": round(fert, 4),
                        "chars_per_token": round(cpt, 4),
                        "parity_vs_english": round(parity, 4),
                        "est_input_cost_usd": (
                            round(est_cost, 8) if est_cost is not None else None
                        ),
                        "cost_usd_per_1m_chars": (
                            round(cost_1m_chars, 6) if cost_1m_chars is not None else None
                        ),
                        "cost_penalty_vs_english": (
                            round(cost_penalty, 4) if cost_penalty is not None else None
                        ),
                    }
                )
            if model_failed:
                break

    return rows, skipped


def write_csv(rows, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=ROW_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def write_json(rows, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)


def summarize(rows):
    """Mean fertility / parity / cost-per-1M-chars per (model, lang)."""
    groups = defaultdict(list)
    for row in rows:
        groups[(row["model_id"], row["model_display_name"], row["lang"])].append(row)

    summary = []
    for (model_id, model_name, lang), group in groups.items():
        n = len(group)
        avg_fert = sum(r["fertility_tokens_per_word"] for r in group) / n
        avg_parity = sum(r["parity_vs_english"] for r in group) / n
        costs = [r["cost_usd_per_1m_chars"] for r in group if r["cost_usd_per_1m_chars"] is not None]
        avg_cost = sum(costs) / len(costs) if costs else None
        summary.append(
            {
                "model_id": model_id,
                "model_display_name": model_name,
                "lang": lang,
                "avg_fertility_tokens_per_word": round(avg_fert, 3),
                "avg_parity_vs_english": round(avg_parity, 3),
                "avg_cost_usd_per_1m_chars": round(avg_cost, 4) if avg_cost is not None else None,
            }
        )
    summary.sort(key=lambda r: (r["model_id"], r["lang"]))
    return summary


def print_summary(summary):
    headers = ["model", "lang", "fertility", "parity_vs_en", "cost/1M chars ($)"]
    col_widths = [24, 5, 10, 13, 18]

    def fmt_row(cells):
        return "  ".join(str(c).ljust(w) for c, w in zip(cells, col_widths))

    print(fmt_row(headers))
    print(fmt_row(["-" * w for w in col_widths]))
    for row in summary:
        cost = "n/a" if row["avg_cost_usd_per_1m_chars"] is None else f"{row['avg_cost_usd_per_1m_chars']:.4f}"
        print(
            fmt_row(
                [
                    row["model_display_name"],
                    row["lang"],
                    row["avg_fertility_tokens_per_word"],
                    row["avg_parity_vs_english"],
                    cost,
                ]
            )
        )
