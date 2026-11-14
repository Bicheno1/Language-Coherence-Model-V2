"""Store: the databases. Everything is ADDITIVE counts (ingesting in batches = ingesting everything at once). Common layer + one layer per speaker.
Forms are stored as they were seen; merges of variants (typos) are applied when compiling (resolve)."""
import json, math, os
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from .script import parse_script, parse_auto, make_pairs, NotDialogueError
from .lexicon import Lexicon, tokenize, norm, PUNCT, END
from .lang import detect_lang, LANGS
from .analyzer import LightAnalyzer
from .axes import AxisModel, AX, axis_profile

DATA = os.path.join(os.path.dirname(__file__), "data")
SEED = os.path.join(DATA, "axis_seed.json")                      # English (kept for compatibility)


def seed_for(lang):
    p = os.path.join(DATA, f"axis_seed_{lang}.json")
    return p if os.path.exists(p) else (SEED if lang == "en" else None)


@dataclass
class Speaker:
    n: int = 0
    words: int = 0
    openers: Counter = field(default_factory=Counter)
    closers: Counter = field(default_factory=Counter)
    axis: list = field(default_factory=lambda: [0, 0, 0, 0])


class Store:
    def __init__(self, lang=None, axis_seed="auto", analyzer=None, keep_text=True, window=3, lex_params=None):
        """lang: 'en', 'es' or None (detected on the first ingestion). axis_seed: path, 'auto' (by language) or None (no axes)."""
        self.lang, self.axis_seed, self.lex_params = lang, axis_seed, dict(lex_params or {})
        self.lex = Lexicon(**self.lex_params, lang=lang or "en")
        self.analyzer = analyzer or LightAnalyzer()
        self.axis = None
        if lang: self._init_axis()
        self.keep_text, self.window = keep_text, window
        self.uni, self.uni_s = Counter(), defaultdict(Counter)
        self.rel, self.rel_s = Counter(), defaultdict(Counter)
        self.votes, self.utt_axis = defaultdict(lambda: [0, 0, 0, 0]), [0, 0, 0, 0]
        self.spk, self.pairs = defaultdict(Speaker), []
        self.n_utts = self.n_pairs = 0

    @classmethod
    def for_lcm(cls, tables=None, lang="en", **kw):
        """A store connected to the LCM: typed relations with the in-house analyzer and the 9,263 words of the tables
        as the known vocabulary (so typos of known words are caught from the first sentence)."""
        from .. import load
        from .analyzer import LiteAnalyzer
        tables = tables if tables is not None else load()
        s = cls(lang=lang, analyzer=LiteAnalyzer(tables) if lang == "en" else None, **kw)
        words = {n["word"].lower() for tb in ("subjects", "concepts", "connectors") for n in tables[tb].nodes if " " not in n["word"]}
        s.lex.known(sorted(words))
        s.tables = tables
        return s

    def _init_axis(self):
        path = seed_for(self.lang) if self.axis_seed == "auto" else self.axis_seed
        self.axis = AxisModel.from_seed(path) if path else None
        self.lex.lang = self.lang

    def set_lang(self, text, lang=None):
        """Sets the language of the base (one base = one language). If it is already set and the text is in another, it raises an error."""
        found = lang or detect_lang(text)
        if self.lang is None:
            if found not in LANGS: raise ValueError(f"could not detect the language (found {found!r}); use lang='en' or 'es'")
            self.lang = found; self._init_axis()
        elif found in LANGS and found != self.lang and lang is None:
            raise ValueError(f"this base is in language {self.lang!r} and the text looks {found!r}; use a separate base or force the language")
        return self.lang

    # ---------------- ingestion ----------------
    def _process(self, speaker, text):
        raw = tokenize(text); toks = [norm(t) for t in raw]
        idx = [i for i, t in enumerate(toks) if t not in PUNCT]
        W = self.window
        for pos, i in enumerate(idx):
            ctx = [toks[j] for j in idx[max(0, pos - W):pos] + idx[pos + 1:pos + 1 + W]]
            is_name = len(raw[i]) > 1 and raw[i][0].isupper() and raw[i][1:].islower() and i > 0 and toks[i - 1] not in (".", "?", "!")
            self.lex.observe(toks[i], ctx, is_name)
        words = [toks[i] for i in idx]                  # the form as seen is counted; merging of variants is applied when compiling
        ax = self.axis.predict(text)[0] if (self.axis and words) else None
        rels = self.analyzer.relations(text, toks)
        self.uni.update(words); self.uni_s[speaker].update(words)
        self.rel.update(rels); self.rel_s[speaker].update(rels)
        sp = self.spk[speaker]; sp.n += 1; sp.words += len(words)
        if toks:
            sp.openers[self.lex.resolve(toks[0])] += 1; sp.closers[self.lex.resolve(toks[-1])] += 1
        if ax is not None:
            k = AX.index(ax); self.utt_axis[k] += 1; sp.axis[k] += 1
            for w in set(words): self.votes[w][k] += 1
        self.n_utts += 1
        return ax

    def ingest_turns(self, turns, source="script"):
        axes = {t.index: self._process(t.speaker, t.text) for t in turns}
        pairs = make_pairs(turns)
        if self.keep_text:
            for p in pairs:
                self.pairs.append(dict(ps=p.prompt.speaker, pt=p.prompt.text, pa=axes[p.prompt.index],
                                       rs=p.reply.speaker, rt=p.reply.text, ra=axes[p.reply.index], src=source))
        self.n_pairs += len(pairs)
        return dict(turns=len(turns), pairs=len(pairs), speakers=sorted({t.speaker for t in turns}))

    def ingest_script(self, text, source="script", fmt="auto", monologue=None, lang=None, unlabeled="auto"):
        """Recognizes the format (names, chat, dash) and the language; if the text is not a conversation it raises NotDialogueError."""
        turns, rep = parse_auto(text, fmt, monologue, unlabeled)
        self.set_lang(text, lang)
        out = self.ingest_turns(turns, source); out["format"] = rep; out["lang"] = self.lang
        return out

    def fix_alias(self, variant, canonical):
        """Fixes a variant merge (e.g. 'helo' -> 'hello'). canonical=None undoes it. Counts are rebuilt when compiling."""
        from .lexicon import WordInfo
        lx, n = self.lex, self.uni.get(variant, 0)
        old = lx.alias.get(variant)
        if old is not None:
            lx.typos.pop((variant, old), None)
            if old in lx.info: lx.info[old].count = max(0, lx.info[old].count - n)
        if canonical is None:
            lx.alias.pop(variant, None)
            lx.info.setdefault(variant, WordInfo(n, "confirmed", Counter(), len(lx.info)))
            if lx.info[variant].status == "merged": lx.info[variant].status = "confirmed"
            return
        lx.known([canonical]); lx.alias[variant] = canonical; lx.typos[(variant, canonical)] += n; lx.info[canonical].count += n
        if variant in lx.info: lx.info[variant].status = "merged"

    # ---------------- derived data ----------------
    def _speaker_key(self, name):
        for k in self.spk:
            if k.casefold() == name.casefold():
                return k
        raise KeyError(f"unknown character: {name!r} (available: {', '.join(self.spk)})")

    def _merged(self):
        R = self.lex.resolve; uni, votes, rel = Counter(), defaultdict(lambda: [0] * 4), Counter()
        for w, c in self.uni.items(): uni[R(w)] += c
        for w, v in self.votes.items():
            t = votes[R(w)]
            for i in range(4): t[i] += v[i]
        for (r, a, b), c in self.rel.items(): rel[(r, R(a), R(b))] += c
        return uni, votes, rel

    def base_rates(self):
        tot = sum(self.utt_axis) or 1
        return [max(x / tot, 1e-3) for x in self.utt_axis]

    def table(self, min_count=1, top_rel=6, min_axis_n=5):
        """Table for the LCM: per word, number, count, four axis values, center, dispersion and main relations."""
        uni, votes, rel = self._merged(); ids, base = self.lex.ids(), self.base_rates()
        by = defaultdict(list)
        for (r, a, b), c in rel.items():
            by[a].append((c, r, ">", b)); by[b].append((c, r, "<", a))
        words = []
        for w, c in uni.most_common():
            if c < min_count: continue
            words.append(dict(id=ids.get(w), word=w, cipher=list(self.lex.code(w)), count=c, status=self.lex.info[w].status if w in self.lex.info else "confirmed",
                              axes=axis_profile(votes[w], base, min_axis_n) if w in votes else None,
                              relations=[dict(rel=r, dir=d, word=o, n=n) for n, r, d, o in sorted(by.get(w, []), reverse=True)[:top_rel]]))
        return dict(meta=dict(lang=self.lang, utterances=self.n_utts, pairs=self.n_pairs, relation_channel="typed" if self.analyzer.typed else "position-only (adj/near)",
                              axis_base_rates=dict(zip(["V", "I", "Lv", "Gv"], [round(x, 3) for x in base]))),
                    chars={c: n for c, n in sorted(self.lex.cs.table.items(), key=lambda x: x[1])},
                    words=words, speakers={s: self.speaker_profile(s) for s in self.spk},
                    typo_merges=[dict(variant=a, word=b, n=n) for (a, b), n in self.lex.typos.most_common()])

    def speaker_profile(self, name):
        key = self._speaker_key(name); sp = self.spk[key]; n = sum(sp.axis) or 1
        return dict(speaker=key, utterances=sp.n, words_per_utterance=round(sp.words / max(sp.n, 1), 2),
                    axis_mix=dict(zip(["V", "I", "Lv", "Gv"], [round(x / n, 2) for x in sp.axis])),
                    top_openers=sp.openers.most_common(5), top_closers=sp.closers.most_common(5), vocabulary=len(self.uni_s[key]))

    def neighbors(self, word, k=5, min_count=3):
        """Convergence: words with similar relation profiles (PPMI + cosine). With LightAnalyzer it is by position only."""
        uni, _, rel = self._merged(); word = self.lex.resolve(norm(word)); feats = defaultdict(Counter)
        for (r, a, b), c in rel.items():
            feats[a][(r, ">", b)] += c; feats[b][(r, "<", a)] += c
        ws = [w for w, c in uni.items() if c >= min_count and w in feats]
        if word not in ws: return []
        tot = sum(sum(feats[w].values()) for w in ws); fc = Counter()
        for w in ws:
            for f, c in feats[w].items(): fc[f] += c
        def vec(w):
            rs = sum(feats[w].values()); v = {}
            for f, c in feats[w].items():
                p = math.log((c * tot) / (rs * fc[f]))
                if p > 0: v[f] = p
            return v
        vt = vec(word); nt = math.sqrt(sum(x * x for x in vt.values())) or 1.0; out = []
        for w in ws:
            if w == word: continue
            v = vec(w); nv = math.sqrt(sum(x * x for x in v.values())) or 1.0
            out.append((sum(x * v.get(f, 0) for f, x in vt.items()) / (nt * nv), w))
        return [(w, round(s, 3)) for s, w in sorted(out, reverse=True)[:k] if s > 0]

    # ---------------- save / load ----------------
    def to_dict(self):
        return dict(version=2, lang=self.lang, axis_seed=self.axis_seed, window=self.window, keep_text=self.keep_text, typed=self.analyzer.typed, lex=self.lex.to_dict(),
                    uni=dict(self.uni), uni_s={s: dict(c) for s, c in self.uni_s.items()},
                    rel=[[r, a, b, c] for (r, a, b), c in self.rel.items()],
                    rel_s={s: [[r, a, b, c] for (r, a, b), c in cn.items()] for s, cn in self.rel_s.items()},
                    votes={w: v for w, v in self.votes.items()}, utt_axis=self.utt_axis,
                    spk={s: dict(n=v.n, words=v.words, openers=dict(v.openers), closers=dict(v.closers), axis=v.axis) for s, v in self.spk.items()},
                    pairs=self.pairs, n_utts=self.n_utts, n_pairs=self.n_pairs)

    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False)

    @classmethod
    def load(cls, path, analyzer=None):
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        s = cls(lang=d.get("lang"), axis_seed=d.get("axis_seed", "auto"), analyzer=analyzer, keep_text=d["keep_text"], window=d["window"])
        s.lex = Lexicon.from_dict(d["lex"])
        s.uni = Counter(d["uni"]); s.uni_s = defaultdict(Counter, {k: Counter(v) for k, v in d["uni_s"].items()})
        s.rel = Counter({(r, a, b): c for r, a, b, c in d["rel"]})
        s.rel_s = defaultdict(Counter, {k: Counter({(r, a, b): c for r, a, b, c in v}) for k, v in d["rel_s"].items()})
        for w, v in d["votes"].items(): s.votes[w] = list(v)
        s.utt_axis = d["utt_axis"]
        for k, v in d["spk"].items(): s.spk[k] = Speaker(v["n"], v["words"], Counter(v["openers"]), Counter(v["closers"]), v["axis"])
        s.pairs, s.n_utts, s.n_pairs = d["pairs"], d["n_utts"], d["n_pairs"]
        return s
