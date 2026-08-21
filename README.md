# FLORES+ Tokenizer Fertility Benchmark

Measures how many tokens each model's tokenizer needs for the same sentence
in English, Hindi, Telugu, Kannada, and Marathi, and translates that into
serving cost. Corpus is the full FLORES+ devtest split (1,012
sentence-aligned rows, professionally translated, CC-BY-SA-4.0).

Every number comes from tokenizing locally or via a free `count_tokens`-style
call — nothing here generates text or spends money.

```
src/tokfertility/         library: registry, backends, cache, metrics, pricing, report
  data/pricing_*.json       dated $/1M-token rate cards
flores_build_corpus.py    downloads FLORES+ devtest -> flores_subset/
flores_fertility.py       runs fertility over the corpus for the selected models
flores_merge_report.py    merges per-model result dirs into one comparable table
flores_subset/            FLORES+ corpus, fetched by flores_build_corpus.py
```

The FLORES+ corpus, result tables, and token caches are all generated, not
committed — output lands in `flores_subset/`, `flores_results*/`, and
`flores_summary/` after you run the corresponding script (see "Data &
Licensing" for why the corpus isn't shipped).

## What this measures

For every (model, language, sentence) in the corpus:

- **token_count** — tokens the model's tokenizer produces for that sentence
- **fertility_tokens_per_word** — tokens / word (lower = more efficient for that language)
- **chars_per_token** — characters represented per token
- **parity_vs_english** — this sentence's token count / its English counterpart's (FLORES+ is a genuine parallel corpus, so this ratio is meaningful)
- **est_input_cost_usd** / **cost_usd_per_1m_chars** — cost from the dated pricing table; `null` for open-weight models with no owner-published API price

`flores_merge_report.py` corrects for live-API envelope overhead and reports
corpus-level parity/fertility (total tokens over all 1,012 sentences, not an
average of per-sentence ratios) — see [Known limitations](#known-limitations).

## Setup

macOS / Linux:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

Windows (PowerShell):

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -e .
```

`pip install -e .` alone also works, pulling current versions satisfying
`pyproject.toml`'s `>=` bounds; `requirements.txt` pins the exact versions
this benchmark was run with.

**Gated tokenizers** (Llama 3.1 8B, Gemma 2 9B, Gemma 3 4B) need `HF_TOKEN`
set to a Hugging Face token with the model licenses accepted.

**Live-API models** need `GEMINI_API_KEY` (free at
https://aistudio.google.com/apikey) and/or `ANTHROPIC_API_KEY`
(https://console.anthropic.com). All other registered models need no auth.

## Running

The corpus isn't shipped in this repo (see "Data & Licensing"). Fetch it
first — no HF token needed, despite the Hub UI showing the dataset as gated:

```bash
python flores_build_corpus.py    # -> flores_subset/corpus_flores_v1.json
```

**1. Local tokenizers** — a few minutes, no credentials beyond `HF_TOKEN` for
the gated ones:

```bash
python flores_fertility.py --models sarvam-1,gemma-2-9b,gemma-3-4b,brahmic-131k,mistral-nemo-tekken,qwen3-8b,llama-3.1-8b,cl100k-base,gpt-4o,sarvam-30b
```

List all ten in one command: each run overwrites `results.csv` in `--out`
rather than appending, so splitting them across invocations drops the earlier
rows. (`--local` covers only the first nine.)

**2. Live-API models** — one at a time, not in parallel. Each is ~5,060 calls
against a free rate-limited endpoint: roughly 1–1.5h per Claude model, up to
~6h for Gemini. Counts are cached per sentence, so an interrupted run resumes
where it stopped instead of restarting. Each model gets its own cache and
output directory:

```bash
python flores_fertility.py --models claude-sonnet-4-5 --cache-dir cache_live_sonnet45 --out flores_results_sonnet45
python flores_fertility.py --models claude-opus-4-8   --cache-dir cache_live_opus48   --out flores_results_opus48
python flores_fertility.py --models gemini-3.6-flash  --cache-dir cache_live_gemini36 --out flores_results_gemini36
```

**3. Merge** into `flores_summary/summary.csv` and `summary.json`:

```bash
python flores_merge_report.py --out flores_summary
```

Local tokenizer counts are deterministic with `requirements.txt` pinned — a
diff between two runs means a dependency version changed a tokenizer's
behavior. Live-API counts carry no such guarantee, since a provider can
revise its tokenizer without notice.

## Known limitations

- **Word count as a fertility denominator**: all four Indic languages are
  more agglutinative than English, so "tokens per whitespace-separated word"
  is a proxy, not a linguistically precise unit.
- **Open-weight model pricing**: Llama, Gemma, Qwen, Mistral, and Sarvam have
  no single official API price (they're self-hostable) — `pricing_available`
  is `false` and cost columns are `null` for those rows.
- **Live-API counts include per-request overhead** (8 tokens for Claude
  Sonnet 4.5, 7 for Claude Opus 4.8, 2 for Gemini 3.6 Flash, measured
  empirically) that `flores_fertility.py`'s raw output does not subtract.
  `flores_merge_report.py` corrects for it, so read those numbers from
  `flores_summary/` rather than hand-averaging the raw
  `flores_results*/results.csv` files.
- **Corpus-level vs per-sentence-mean parity**: averaging each sentence's own
  token ratio over-weights short sentences, where the ratio is noisiest. The
  corpus-level figure is what actually predicts a serving bill.
- **Pricing snapshots go stale fast**: `data/pricing_2026-08-17.json` is the
  latest rate card bundled here; re-verify against each provider's current
  published pricing before citing a cost figure.

## Extending it

- **Add a language**: edit `LANG_CODES` in `flores_build_corpus.py` and
  rebuild the corpus (FLORES+ covers ~200 languages). Bump `CORPUS_VERSION`
  so past results stay reproducible.
- **Add a model**: add a `ModelSpec` to `src/tokfertility/registry.py`, a
  matching row to the pricing file, and — for a new local tokenizer — to
  `DEFAULT_MODELS`/`LOCAL_MODELS` in `flores_fertility.py`.
- **Update pricing**: add a new `data/pricing_YYYY-MM-DD.json` and pass it
  with `--pricing` rather than editing an existing dated file.

## Data & Licensing

The code in this repo (everything under `src/`, the top-level scripts,
`pyproject.toml`) is MIT-licensed — see `LICENSE`. That license covers the
code only; it does not cover, and does not relicense, the third-party
dataset this benchmark measures:

- **[FLORES+](https://huggingface.co/datasets/openlanguagedata/flores_plus)**
  (`openlanguagedata/flores_plus`) is **CC BY-SA 4.0** and access-gated on
  the Hub. It is not bundled in this repo — `flores_subset/` is `.gitignore`d
  and fetched fresh by `flores_build_corpus.py`, pinned to a specific dataset
  revision so the fetched corpus reproduces byte-for-byte. If you redistribute
  the fetched corpus itself (not just this code), CC BY-SA 4.0's attribution
  and share-alike terms apply to it, separately from this repo's MIT license.

## License

Code: MIT — see `LICENSE`. This does not extend to the FLORES+ data — see
"Data & Licensing" above.
