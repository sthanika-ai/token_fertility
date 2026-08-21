"""On-disk cache keyed by (corpus_version, model_id, lang, sentence_id).

Tokenizing is deterministic and free once a tokenizer is loaded, but loading
HF tokenizers (download + init) is the slow part - this cache makes reruns
of the same corpus/model/sentence combination instant.
"""

import json
from pathlib import Path


class ResultCache:
    def __init__(self, cache_dir):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._path = self.cache_dir / "token_counts.json"
        self._data = self._load()
        self._dirty = False

    def _load(self):
        if self._path.exists():
            with open(self._path, encoding="utf-8") as f:
                return json.load(f)
        return {}

    def flush(self):
        if self._dirty:
            with open(self._path, "w", encoding="utf-8") as f:
                json.dump(self._data, f, ensure_ascii=False, indent=2, sort_keys=True)
            self._dirty = False

    @staticmethod
    def _key(corpus_version, model_id, lang, sentence_id):
        return f"{corpus_version}::{model_id}::{lang}::{sentence_id}"

    def get(self, corpus_version, model_id, lang, sentence_id):
        return self._data.get(self._key(corpus_version, model_id, lang, sentence_id))

    def set(self, corpus_version, model_id, lang, sentence_id, token_count):
        self._data[self._key(corpus_version, model_id, lang, sentence_id)] = token_count
        self._dirty = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.flush()
