"""Mapping to the four poles. A classifier (Naive Bayes over words + format features) trained on labeled phrases assigns an axis to each phrase;
each word accumulates the votes of the phrases where it appears (center of mass of its uses) and the votes are normalized with lift against the corpus base rate.
It serves as 'mid-training'; the labels are those of post-training."""
import json, math, re, collections
from .lang import format_features

AX = ["N", "S", "E", "W"]                       # V (North), I (South), Lv (East), Gv (West)
AXNAME = {"N": "V", "S": "I", "E": "Lv", "W": "Gv"}


def _tk(text, lang="en"):
    pat = r"[a-záéíóúüñ']+|[?!¿¡]" if lang == "es" else r"[a-z']+|[?!]"
    return re.findall(pat, text.lower().replace("’", "'")) + format_features(text, lang)


class AxisModel:
    def __init__(self, lang="en"):
        self.lang = lang
        self.c = {a: collections.Counter() for a in AX}
        self.V, self.tot = set(), {a: 0 for a in AX}

    def fit(self, phrases):
        for a, L in phrases.items():
            for p in L:
                t = _tk(p, self.lang); self.c[a].update(t); self.V.update(t)
        self.tot = {a: sum(self.c[a].values()) for a in AX}
        return self

    @classmethod
    def from_seed(cls, path):
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        return cls(d.get("meta", {}).get("lang", "en")).fit(d["phrases"])

    def scores(self, text):
        t, V = _tk(text, self.lang), len(self.V) + 1
        return {a: sum(math.log((self.c[a][w] + .5) / (self.tot[a] + .5 * V)) for w in t) for a in AX}

    def predict(self, text):
        s = self.scores(text); o = sorted(AX, key=s.get, reverse=True)
        return o[0], s[o[0]] - s[o[1]]


def axis_profile(votes, base, min_n=5):
    """Four values 0-10 (V, I, Lv, Gv), a center (the dominant axis) and a dispersion (0 narrow .. 1 wide)."""
    n = sum(votes)
    if n == 0:
        return None
    lift = [(votes[i] / n) / max(base[i], 1e-3) for i in range(4)]
    m = max(lift); w = [round(10 * x / m) for x in lift]
    p = [x / sum(lift) for x in lift]
    H = -sum(x * math.log(x) for x in p if x > 0) / math.log(4)
    ok = n >= min_n
    return dict(n=n, V=w[0], I=w[1], Lv=w[2], Gv=w[3], center=AXNAME[AX[lift.index(m)]] if ok else None, dispersion=abs(round(H, 3)), confidence="ok" if ok else "low")
