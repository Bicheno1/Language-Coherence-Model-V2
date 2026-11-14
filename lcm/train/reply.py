"""BASELINE reply by retrieval: returns what THAT character answered to similar inputs.
It is NOT the LCM and it does not generate text: it serves to check that the databases capture the character.
If the character has few replies, those of the others are mixed in (stepped)."""
import math
from collections import Counter
from .lexicon import tokenize, norm, PUNCT


class Retriever:
    def __init__(self, store):
        self.s = store
        df = Counter()
        for p in store.pairs: df.update(set(self._tokens(p["pt"])))
        N = max(len(store.pairs), 1)
        self.idf = lambda w: math.log(1 + N / (df.get(w, 0) + 1))
        self.vecs = [self._vec(p["pt"]) for p in store.pairs]

    def _tokens(self, text):
        return [self.s.lex.resolve(norm(t)) for t in tokenize(text) if norm(t) not in PUNCT]

    def _vec(self, text):
        return {w: (1 + math.log(n)) * self.idf(w) for w, n in Counter(self._tokens(text)).items()}

    @staticmethod
    def _cos(a, b):
        na = math.sqrt(sum(x * x for x in a.values())); nb = math.sqrt(sum(x * x for x in b.values()))
        return sum(x * b.get(w, 0.0) for w, x in a.items()) / (na * nb) if na and nb else 0.0

    def reply(self, speaker, text, k=3, lam=10, own_only=False):
        key = self.s._speaker_key(speaker); q = self._vec(text)
        qa = self.s.axis.predict(text)[0] if self.s.axis else None
        n_own = sum(1 for p in self.s.pairs if p["rs"] == key); wb = 0.0 if own_only else lam / (n_own + lam)
        out = []
        for p, v in zip(self.s.pairs, self.vecs):
            sim = self._cos(q, v) + (0.05 if qa and p["pa"] == qa else 0.0)
            w = 1.0 if p["rs"] == key else wb
            if w > 0 and sim > 0: out.append((sim * w, p["rt"], p["rs"], p["pt"]))
        return sorted(out, reverse=True)[:k]
