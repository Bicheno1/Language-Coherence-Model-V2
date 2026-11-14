"""Word lookup -> nodes. The word is the unit; letters serve as a search index (typo tolerance).
An unknown word is PROVISIONAL (its own token, not merged). It is confirmed, or merged with a variant, after several occurrences by comparing
contexts (the same idea as convergence between pyramids). Inflections (file/files, use/used) are never merged: they carry tense and number."""
import math, re, collections, unicodedata
from dataclasses import dataclass, field
from .charset import Charset

TOKEN_RE = re.compile(r"[^\W\d_]+(?:['’][^\W\d_]+)*|\d+|[.,?!;:¿¡…«»—–\"()\-]")
PUNCT = set(".,?!;:¿¡…«»—–\"()-")
END = (".", "?", "!", "…")


def tokenize(text):
    return TOKEN_RE.findall(unicodedata.normalize("NFC", text))


def norm(tok):
    if tok in PUNCT:
        return tok
    if tok.isdigit():
        return "<num>"
    return tok.lower().replace("’", "'")


_SUF = ("s", "es", "ed", "d", "ing", "er", "ers", "ly", "est", "y")


def is_inflection(a, b, lang="en"):
    """Pairs that are NOT merged even if they are one letter apart: only an accent changes (si/sí), or the inflection (file/files; niño/niña, habla/hablo)."""
    if a == b:
        return False
    if Charset.fold(a) == Charset.fold(b):
        return True
    if lang == "es":
        n = min(len(a), len(b)); p = 0
        while p < n and a[p] == b[p]: p += 1
        return p >= 3 and p >= n - 2
    x, y = (a, b) if len(a) <= len(b) else (b, a)
    if any(y == x + s for s in _SUF):
        return True
    if x.endswith("e") and y in (x[:-1] + "ing", x[:-1] + "ed"):
        return True
    if x.endswith("y") and y in (x[:-1] + "ies", x[:-1] + "ied"):
        return True
    return len(x) > 2 and y in (x + x[-1] + "ed", x + x[-1] + "ing", x + x[-1] + "er")


def dels1(w):
    return {w} | {w[:i] + w[i + 1:] for i in range(len(w))}


def osa(a, b):
    la, lb = len(a), len(b)
    d = [[0] * (lb + 1) for _ in range(la + 1)]
    for i in range(la + 1):
        d[i][0] = i
    for j in range(lb + 1):
        d[0][j] = j
    for i in range(1, la + 1):
        for j in range(1, lb + 1):
            c = 0 if a[i - 1] == b[j - 1] else 1
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + c)
            if i > 1 and j > 1 and a[i - 1] == b[j - 2] and a[i - 2] == b[j - 1]:
                d[i][j] = min(d[i][j], d[i - 2][j - 2] + 1)
    return d[la][lb]


@dataclass
class WordInfo:
    count: int = 0
    status: str = "provisional"      # provisional | confirmed | name | merged
    ctx: collections.Counter = field(default_factory=collections.Counter)
    order: int = 0


class Lexicon:
    def __init__(self, tol_len=4, promote_n=3, tau_alias=0.05, tau_promote=0.15, long_no_ctx=6, lang="en", chars=None):
        self.lang, self.cs, self._codes = lang, Charset(chars), {}
        self.p = dict(tol_len=tol_len, promote_n=promote_n, tau_alias=tau_alias, tau_promote=tau_promote, long_no_ctx=long_no_ctx)
        self.info = {}
        self.alias = {}
        self.index = collections.defaultdict(set)
        self.typos = collections.Counter()
        self.total = 0

    # ---------- utilities ----------
    def known(self, words):
        for w in words:
            if w not in self.info:
                self.info[w] = WordInfo(0, "confirmed", collections.Counter(), len(self.info))
                self._index(w)

    def resolve(self, w):
        return self.alias.get(w, w)

    def code(self, w):
        """The cipher of a word: the sequence of the numbers of its characters."""
        if w not in self._codes: self._codes[w] = self.cs.encode(w)
        return self._codes[w]

    def _index(self, w):
        if len(w) >= self.p["tol_len"]:
            for d in dels1(self.code(w)):
                self.index[d].add(w)

    def _cnt(self, c):
        i = self.info.get(self.alias.get(c, c))
        return i.count if i else 0

    def _vec(self, ctx):
        N = max(self.total, 1)
        return {c: math.log(1 + n) * math.log(1 + N / (self._cnt(c) + 1)) for c, n in ctx.items()}

    @staticmethod
    def _cos(a, b):
        na = math.sqrt(sum(x * x for x in a.values()))
        nb = math.sqrt(sum(x * x for x in b.values()))
        return sum(x * b.get(c, 0.0) for c, x in a.items()) / (na * nb) if na and nb else 0.0

    def _best_candidate(self, w, ctx, tau):
        if len(w) < self.p["tol_len"]:
            return None
        cand, cw = set(), self.code(w)
        for d in dels1(cw):
            cand |= self.index.get(d, set())
        cands = [c for c in cand if c != w and self.info[c].status in ("confirmed",) and osa(cw, self.code(c)) <= 1 and not is_inflection(w, c, self.lang)]
        if not cands:
            return None
        cands.sort(key=lambda c: (-self.info[c].count, c))
        va, best, bs = self._vec(ctx), None, -1.0
        for c in cands[:5]:
            ci = self.info[c]
            s = (1.0 if len(w) >= self.p["long_no_ctx"] else 0.0) if not ci.ctx else self._cos(va, self._vec(ci.ctx))
            if s >= tau and s > bs:
                best, bs = c, s
        return best

    def _merge(self, w, c):
        wi, ci = self.info[w], self.info[c]
        ci.count += wi.count
        ci.ctx.update(wi.ctx)
        wi.status = "merged"
        self.alias[w] = c
        for k, v in list(self.alias.items()):
            if v == w:
                self.alias[k] = c
        self.typos[(w, c)] += wi.count

    def _decide(self, w):
        wi = self.info[w]
        c = self._best_candidate(w, wi.ctx, self.p["tau_promote"])
        if c is not None:
            self._merge(w, c)
        else:
            wi.status = "confirmed"
            self._index(w)

    # ---------- main entry point ----------
    def observe(self, w, ctx_words, is_name=False):
        """Records an occurrence of w with its context. Returns the canonical form to use in the counts."""
        w = self.alias.get(w, w)
        info = self.info.get(w)
        ctx = collections.Counter(ctx_words)
        if info is None:
            cand = None if is_name else self._best_candidate(w, ctx, self.p["tau_alias"])
            if cand is not None:                       # probable typo: counted as the known word
                self.alias[w] = cand
                self.typos[(w, cand)] += 1
                self._touch(cand, ctx)
                return cand
            info = self.info[w] = WordInfo(0, "name" if is_name else "provisional", collections.Counter(), len(self.info))
        self._touch(w, ctx)
        if info.status == "provisional" and info.count >= self.p["promote_n"]:
            self._decide(w)
        return self.resolve(w)

    def _touch(self, w, ctx):
        i = self.info[w]
        i.count += 1
        i.ctx.update(ctx)
        self.total += 1

    # ---------- export ----------
    def ids(self):
        """Stable number per canonical word (in order of first appearance), starting at 1000. Punctuation marks use their character number (1-299)."""
        live = sorted((i.order, w) for w, i in self.info.items() if i.status != "merged")
        return {w: 1000 + n for n, (_, w) in enumerate(live)}

    def to_dict(self):
        return dict(params=self.p, lang=self.lang, chars=self.cs.dynamic(), total=self.total, alias=self.alias,
                    info={w: [i.count, i.status, dict(i.ctx), i.order] for w, i in self.info.items()},
                    typos=[[a, b, n] for (a, b), n in self.typos.items()])

    @classmethod
    def from_dict(cls, d):
        lx = cls(**d["params"], lang=d.get("lang", "en"), chars=d.get("chars"))
        lx.total, lx.alias = d["total"], d["alias"]
        for w, (c, st, ctx, o) in d["info"].items():
            lx.info[w] = WordInfo(c, st, collections.Counter(ctx), o)
            if st == "confirmed":
                lx._index(w)
        lx.typos = collections.Counter({(a, b): n for a, b, n in d["typos"]})
        return lx
