"""Sentence analysis: format features (person, tense, mood, polarity) and relations.
LightAnalyzer: relations by POSITION only (adj / near); it works in any language.
LiteAnalyzer (below): TYPED relations (subject, object, adjective, adverb, preposition) through lcm.lite_nlp; English only."""
import re
from .lexicon import PUNCT

from .lang import format_features  # noqa: F401  (format features per language)


class LightAnalyzer:
    typed = False

    def relations(self, text, toks):
        """toks: tokens already resolved by the lexicon. Relations (rel, a, b): adj = next word, near = 2 or 3 words away."""
        sents, cur = [], []
        for t in toks:
            if t in (".", "?", "!", "…"):
                sents.append(cur); cur = []
            elif t not in PUNCT:
                cur.append(t)
        sents.append(cur)
        out = []
        for s in sents:
            for i, a in enumerate(s):
                if i + 1 < len(s): out.append(("adj", a, s[i + 1]))
                for j in (i + 2, i + 3):
                    if j < len(s): out.append(("near", a, s[j]))
        return out


class LiteAnalyzer:
    """TYPED relations (subject, object, adjective, adverb, preposition) with the in-house analyzer `lcm.lite_nlp` (no spaCy).
    Relations are (rel, dependent, head) on LEMMAS: ('S','dog','jump'), ('O','duck','like'), ('ADJ','sad','story'),
    ('ADV','yesterday','jump'), ('PREP_about','duck','think'). English only; it covers the constructions of lite_nlp."""
    typed = True

    def __init__(self, tables):
        self.tables = tables

    def relations(self, text, toks):
        from ..lite_nlp import analyze
        out = []
        for t in analyze(text, self.tables):
            if t.is_punct or t.head is t:
                continue
            a, h, d = t.lemma_.lower(), t.head.lemma_.lower(), t.dep_
            if d in ("nsubj", "nsubjpass"): out.append(("S", a, h))
            elif d in ("dobj", "obj"): out.append(("O", a, h))
            elif d == "dative": out.append(("IO", a, h))
            elif d == "amod": out.append(("ADJ", a, h))
            elif d in ("acomp", "attr"): out.append(("ATTR", a, h))
            elif d in ("advmod", "npadvmod"): out.append(("ADV", a, h))
            elif d == "pobj" and t.head.head is not t.head:
                out.append((f"PREP_{t.head.lower_}", a, t.head.head.lemma_.lower()))
        return out
