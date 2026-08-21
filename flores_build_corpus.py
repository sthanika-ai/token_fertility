"""Builds a frozen FLORES+-derived corpus (English, Hindi, Telugu, Kannada,
Marathi) in the same schema as src/tokfertility/data/corpus_v1.json, so it
plugs straight into tokfertility.report (parity_vs_english, cost columns)
via flores_fertility.py.

FLORES+ (https://huggingface.co/datasets/openlanguagedata/flores_plus,
CC-BY-SA-4.0) is the maintained successor to FLORES-200: professionally
translated, sentence-aligned parallel text across ~200 languages, keyed by a
shared sentence id across languages. This is exactly the cross-check
corpus_v1.json's own provenance note calls for, since v1's sentences were
hand-authored with LLM-assisted translation rather than sourced from a
vetted parallel corpus.

Downloads the devtest split (1012 sentences) for eng_Latn, hin_Deva,
tel_Telu, kan_Knda, mar_Deva directly from the Hub (no HF_TOKEN needed -
verified anonymous access despite the repo showing gated:auto). By default
takes every sentence common to all 5 languages (the full devtest split);
pass --n for an evenly-spaced sub-sample instead. Writes
flores_subset/corpus_flores_v1.json.

Usage:
    python flores_build_corpus.py                # full devtest (1012 sentences) -> flores_subset/corpus_flores_v1.json
    python flores_build_corpus.py --n 100 --out flores_subset/corpus_flores_v1_n100.json
"""

import argparse
import json
import sys
from pathlib import Path

import truststore

truststore.inject_into_ssl()

from huggingface_hub import hf_hub_download

REPO_ID = "openlanguagedata/flores_plus"
SPLIT = "devtest"
LANG_CODES = {
    "en": "eng_Latn",
    "hi": "hin_Deva",
    "te": "tel_Telu",
    "kn": "kan_Knda",
    "mr": "mar_Deva",
}
CORPUS_VERSION = "flores_v1"
# Pinned dataset commit. Without it the corpus is rebuilt from whatever is at
# main, which would silently produce a corpus different from the frozen
# corpus_flores_v1.json that the published numbers were measured on.
FLORES_REVISION = "5fec6c13f9e5a4db2f745d4ec0d7c9721ddc4f06"


def load_lang_rows(lang_code: str) -> dict[int, dict]:
    path = hf_hub_download(
        REPO_ID, f"{SPLIT}/{lang_code}.jsonl", repo_type="dataset",
        revision=FLORES_REVISION,
    )
    rows = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            rows[row["id"]] = row
    return rows


def build_corpus(n: int | None) -> dict:
    by_lang = {lang: load_lang_rows(code) for lang, code in LANG_CODES.items()}

    common_ids = set(by_lang["en"])
    for rows in by_lang.values():
        common_ids &= set(rows)
    common_ids = sorted(common_ids)

    if n is None:
        sample_ids = common_ids
    else:
        if len(common_ids) < n:
            raise ValueError(f"Only {len(common_ids)} sentence ids are common to all 5 languages, need {n}")
        # Evenly-spaced stride sample (not a random sample) so the corpus
        # stays deterministic across rebuilds and spans the full devtest
        # set's domain mix instead of clustering in whatever domain happens
        # to come first.
        stride = len(common_ids) / n
        sample_ids = [common_ids[int(i * stride)] for i in range(n)]

    sentences = []
    for sid in sample_ids:
        sentence = {"id": sid, "domain": by_lang["en"][sid].get("domain", "")}
        for lang in LANG_CODES:
            sentence[lang] = by_lang[lang][sid]["text"]
        sentences.append(sentence)

    if n is None:
        coverage = (
            f"All {len(common_ids)} sentence ids common to all 5 languages in the "
            f"'{SPLIT}' split (the full devtest set)."
        )
    else:
        coverage = (
            f"{n} of {len(common_ids)} sentence ids common to all 5 languages, "
            f"taken at an even stride across the full '{SPLIT}' split."
        )

    return {
        "version": CORPUS_VERSION,
        "created": "2026-08-07",
        "languages": list(LANG_CODES),
        "provenance": (
            f"Sampled from FLORES+ (openlanguagedata/flores_plus, CC-BY-SA-4.0). "
            f"{coverage} Keyed by FLORES+'s own sentence id (stable across "
            f"languages). Serves as the FLORES-based cross-check that "
            f"corpus_v1.json's provenance note calls for."
        ),
        "sentences": sentences,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=None, help="Sentences to sample (default: all common to every language)")
    parser.add_argument("--out", default="flores_subset/corpus_flores_v1.json", help="Output path")
    args = parser.parse_args(argv)

    corpus = build_corpus(args.n)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(corpus, f, ensure_ascii=False, indent=2)

    print(
        f"Wrote {len(corpus['sentences'])} sentences x {len(corpus['languages'])} languages to {out_path}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
