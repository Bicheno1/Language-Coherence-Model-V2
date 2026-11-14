"""Data that used to come from WordNet at runtime, now READ FROM A JSON (data/wordnet_baked.json).

The JSON is generated ONCE with `python tools/bake_wordnet.py` (that step, and only that one, needs nltk and the WordNet corpus).
At runtime nothing is imported besides the standard library. Contents of the JSON (keys by lowercase word):
  synset        main sense of the word (name of the synset) or absent
  paths         hypernym paths of the main sense, root -> leaf, as lists of synset names (only for biological nodes)
  closure       all the hypernyms (strict ancestors, hypernym relation only) of the main sense, as synset names (biological nodes)
  ancestors     lemmas of the hypernyms of the main sense, from the closest to the farthest
  frequency     times the word appears in its main sense in the WordNet corpus (SemCor); 0 = obscure
  adverbs       words that WordNet knows as adverbs
  intransitive  verbs that have a use without an object ('Somebody ----s')
"""
import json, os
from functools import lru_cache

PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "wordnet_baked.json")


class MissingBakedData(RuntimeError):
    pass


@lru_cache(maxsize=1)
def _load(path=None):
    p = path or PATH
    if not os.path.exists(p):
        raise MissingBakedData(
            f"{os.path.relpath(p)} not found. Generate it once with:  python tools/bake_wordnet.py   "
            "(that step needs nltk + the WordNet corpus; the LCM itself does not).")
    with open(p, encoding="utf-8") as f:
        d = json.load(f)
    d["adverbs"] = set(d.get("adverbs", []))
    return d


def available(path=None):
    try:
        _load(path)
        return True
    except MissingBakedData:
        return False


def main_synset(word):
    return _load()["synset"].get(word)


def paths(word):
    return _load()["paths"].get(word, [])


def closure(word):
    return frozenset(_load()["closure"].get(word, ()))


def ancestors(word):
    return tuple(_load()["ancestors"].get(word, ()))


def frequency(word):
    return _load()["frequency"].get(word, 0)


def is_adverb(word):
    return word in _load()["adverbs"]


def intransitive(verb):
    return bool(_load()["intransitive"].get(verb, False))
