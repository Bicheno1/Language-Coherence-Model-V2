"""Replying AS A CHARACTER: it does not write free text, it chooses among the ways the character has answered in similar situations,
with a randomizer weighted by how many times she used each one (to 'hello' she answers 'hello comrade' or 'hello companion').
If the situation does not look like anything she has said, it abstains ('I do not know').  This is not the LCM's own generation:
it is the repertoire of the character. The LCM can use it to bias its choices (see lcm.train.export)."""
import random
from collections import defaultdict
from dataclasses import dataclass, field
from typing import List, Optional
from .reply import Retriever


@dataclass
class CharReply:
    text: Optional[str]
    kind: str                                   # 'repertoire' | 'abstain'
    confidence: float                           # similarity of the closest situation she has lived (0..1)
    alternatives: List[tuple] = field(default_factory=list)   # [(text, probability, from_character)]
    matched: Optional[str] = None               # the input that most resembles the new one


class Character:
    def __init__(self, store, name, tau=0.25, rel=0.7, min_own=3, lam=10):
        """tau: minimum similarity to answer. rel: only situations within `rel` of the best one count. min_own / lam: with fewer own
        pairs than `min_own`, the replies of the others are mixed in with weight lam / (n_own + lam)."""
        self.store, self.name = store, store._speaker_key(name)
        self.tau, self.rel, self.min_own, self.lam = tau, rel, min_own, lam
        self._r = Retriever(store)

    def n_pairs(self):
        return sum(1 for p in self.store.pairs if p["rs"] == self.name)

    def respond(self, text, seed=None, temperature=1.0):
        r = self._r
        q = r._vec(text)
        n_own = self.n_pairs()
        mix = n_own < self.min_own
        wb = self.lam / (n_own + self.lam) if mix else 0.0
        scored = []
        for p, v in zip(self.store.pairs, r.vecs):
            own = p["rs"] == self.name
            if not own and not mix:
                continue
            sim = r._cos(q, v)
            if sim > 0:
                scored.append((sim, p, own))
        if not scored:
            return CharReply(None, "abstain", 0.0)
        best = max(s for s, _, _ in scored)
        if best < self.tau:
            return CharReply(None, "abstain", round(best, 3))
        group = defaultdict(float); src = {}
        for sim, p, own in scored:
            if sim >= self.rel * best:
                k = p["rt"].strip().lower()
                group[k] += sim * (1.0 if own else wb); src.setdefault(k, (p["rt"].strip(), own, p["pt"]))
        keys = sorted(group)
        if temperature <= 0:
            weights = [1.0 if k == max(keys, key=group.get) else 0.0 for k in keys]
        else:
            weights = [group[k] ** (1.0 / temperature) for k in keys]
        tot = sum(weights) or 1.0
        probs = [w / tot for w in weights]
        pick = random.Random(seed).choices(keys, probs)[0]
        alts = sorted(((src[k][0], round(pr, 3), src[k][1]) for k, pr in zip(keys, probs)), key=lambda x: -x[1])
        return CharReply(src[pick][0], "repertoire", round(best, 3), alts, src[pick][2])
