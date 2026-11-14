"""English inflection without external libraries (replaces lemminflect): rules + tables of irregular forms.

It exposes the same three functions that the LCM used from lemminflect (getInflection, getAllInflections, getLemma), so the
modules that call them only change the import. Plus helpers for the analyzer: plural(), singular_candidates(),
verb_lemma_candidates().  Coverage: regular morphology and the ~130 most common irregular verbs and the usual irregular plurals.
"""
import re

# base: (past, past participle)
IRREG_VERBS = {
    "be": ("was", "been"), "have": ("had", "had"), "do": ("did", "done"), "go": ("went", "gone"), "say": ("said", "said"), "see": ("saw", "seen"),
    "tell": ("told", "told"), "make": ("made", "made"), "get": ("got", "gotten"), "take": ("took", "taken"), "come": ("came", "come"),
    "know": ("knew", "known"), "think": ("thought", "thought"), "give": ("gave", "given"), "find": ("found", "found"), "run": ("ran", "run"),
    "sit": ("sat", "sat"), "stand": ("stood", "stood"), "write": ("wrote", "written"), "read": ("read", "read"), "eat": ("ate", "eaten"),
    "drink": ("drank", "drunk"), "sleep": ("slept", "slept"), "fall": ("fell", "fallen"), "feel": ("felt", "felt"), "hold": ("held", "held"),
    "keep": ("kept", "kept"), "leave": ("left", "left"), "lose": ("lost", "lost"), "meet": ("met", "met"), "pay": ("paid", "paid"),
    "sell": ("sold", "sold"), "send": ("sent", "sent"), "speak": ("spoke", "spoken"), "spend": ("spent", "spent"), "swim": ("swam", "swum"),
    "teach": ("taught", "taught"), "win": ("won", "won"), "fly": ("flew", "flown"), "sing": ("sang", "sung"), "ring": ("rang", "rung"),
    "begin": ("began", "begun"), "break": ("broke", "broken"), "bring": ("brought", "brought"), "build": ("built", "built"), "buy": ("bought", "bought"),
    "catch": ("caught", "caught"), "choose": ("chose", "chosen"), "draw": ("drew", "drawn"), "drive": ("drove", "driven"), "fight": ("fought", "fought"),
    "forget": ("forgot", "forgotten"), "hide": ("hid", "hidden"), "hit": ("hit", "hit"), "hurt": ("hurt", "hurt"), "lead": ("led", "led"),
    "lend": ("lent", "lent"), "let": ("let", "let"), "put": ("put", "put"), "ride": ("rode", "ridden"), "rise": ("rose", "risen"), "set": ("set", "set"),
    "shake": ("shook", "shaken"), "shoot": ("shot", "shot"), "shut": ("shut", "shut"), "sink": ("sank", "sunk"), "steal": ("stole", "stolen"),
    "swing": ("swung", "swung"), "throw": ("threw", "thrown"), "understand": ("understood", "understood"), "wake": ("woke", "woken"),
    "wear": ("wore", "worn"), "bear": ("bore", "borne"), "beat": ("beat", "beaten"), "become": ("became", "become"), "bend": ("bent", "bent"),
    "bite": ("bit", "bitten"), "blow": ("blew", "blown"), "burn": ("burnt", "burnt"), "cost": ("cost", "cost"), "cut": ("cut", "cut"),
    "dig": ("dug", "dug"), "dream": ("dreamt", "dreamt"), "feed": ("fed", "fed"), "freeze": ("froze", "frozen"), "grow": ("grew", "grown"),
    "hang": ("hung", "hung"), "hear": ("heard", "heard"), "kneel": ("knelt", "knelt"), "lay": ("laid", "laid"), "lie": ("lay", "lain"),
    "light": ("lit", "lit"), "mean": ("meant", "meant"), "quit": ("quit", "quit"), "seek": ("sought", "sought"), "shine": ("shone", "shone"),
    "slide": ("slid", "slid"), "spin": ("spun", "spun"), "split": ("split", "split"), "spread": ("spread", "spread"), "spring": ("sprang", "sprung"),
    "stick": ("stuck", "stuck"), "sting": ("stung", "stung"), "strike": ("struck", "struck"), "swear": ("swore", "sworn"), "sweep": ("swept", "swept"),
    "tear": ("tore", "torn"), "wind": ("wound", "wound"), "forgive": ("forgave", "forgiven"), "overcome": ("overcame", "overcome"),
    "undo": ("undid", "undone"), "withdraw": ("withdrew", "withdrawn"), "arise": ("arose", "arisen"), "awake": ("awoke", "awoken"),
    "dive": ("dove", "dived"), "flee": ("fled", "fled"), "fling": ("flung", "flung"), "forbid": ("forbade", "forbidden"), "grind": ("ground", "ground"),
    "lean": ("leant", "leant"), "leap": ("leapt", "leapt"), "learn": ("learnt", "learnt"), "spell": ("spelt", "spelt"), "spill": ("spilt", "spilt"),
    "spit": ("spat", "spat"), "weave": ("wove", "woven"), "weep": ("wept", "wept"), "bid": ("bid", "bid"), "bind": ("bound", "bound"),
    "bleed": ("bled", "bled"), "breed": ("bred", "bred"), "cling": ("clung", "clung"), "creep": ("crept", "crept"), "deal": ("dealt", "dealt"),
}
BE_FORMS = {"am": "VBP", "is": "VBZ", "are": "VBP", "was": "VBD", "were": "VBD", "been": "VBN", "being": "VBG", "be": "VB"}
IRREG_PLURALS = {"man": "men", "woman": "women", "child": "children", "person": "people", "mouse": "mice", "foot": "feet", "tooth": "teeth",
                 "goose": "geese", "ox": "oxen", "louse": "lice", "die": "dice", "sheep": "sheep", "fish": "fish", "deer": "deer",
                 "species": "species", "series": "series", "aircraft": "aircraft", "moose": "moose", "swine": "swine"}
_F_TO_VES = {"wolf", "knife", "life", "wife", "leaf", "half", "shelf", "thief", "loaf", "calf", "self", "elf", "scarf", "wharf", "hoof"}
_O_TO_ES = {"hero", "potato", "tomato", "echo", "veto", "torpedo", "volcano"}
_DOUBLE_LONG = {"begin", "admit", "commit", "prefer", "refer", "occur", "regret", "permit", "submit", "control", "equip", "forget", "upset",
                "transfer", "compel", "expel", "rebel", "repel", "propel", "patrol", "format", "target", "output", "input"}
_NOT_PLURAL = {"news", "series", "species", "gas", "bus", "lens", "chaos", "atlas", "canvas", "bias", "pass", "class", "glass", "grass", "boss",
               "kiss", "miss", "mass", "cross", "loss", "moss", "dress", "stress", "success", "business", "address", "princess", "mess", "less",
               "this", "his", "was", "has", "does", "is", "yes", "plus", "thus", "us", "always", "perhaps", "sometimes", "across", "unless",
               "status", "virus", "bonus", "chorus", "campus", "genius", "analysis", "crisis", "basis", "thesis", "axis", "oasis", "iris"}
_IREG_VERB_REV = {}
for _b, (_p, _pp) in IRREG_VERBS.items():
    _IREG_VERB_REV.setdefault(_p, []).append((_b, "VBD"))
    if _pp != _p:
        _IREG_VERB_REV.setdefault(_pp, []).append((_b, "VBN"))
_IREG_PLURAL_REV = {v: k for k, v in IRREG_PLURALS.items() if v != k}
_VOWELS = "aeiou"


def _cvc_double(base):
    """stop -> stopped, plan -> planned, run -> running; not: visit -> visited, open -> opened, play -> played."""
    if base in _DOUBLE_LONG:
        return True
    if len(base) < 3 or base[-1] in "wxy" or base[-1] in _VOWELS or base[-2] not in _VOWELS or base[-3] in _VOWELS:
        return False
    return len(re.findall(r"[aeiouy]+", base)) == 1


def plural(noun):
    n = noun.lower()
    if n in IRREG_PLURALS:
        return IRREG_PLURALS[n]
    if n in _F_TO_VES:
        return n[:-2] + "ves" if n.endswith("fe") else n[:-1] + "ves"
    if re.search(r"[^aeiou]y$", n):
        return n[:-1] + "ies"
    if re.search(r"(s|x|z|ch|sh)$", n) or n in _O_TO_ES:
        return n + "es"
    return n + "s"


def singular_candidates(word):
    """Ordered candidates for the singular of `word` (the caller keeps the first one that is a known word)."""
    w = word.lower()
    if w in _IREG_PLURAL_REV:
        return [_IREG_PLURAL_REV[w]]
    out = []
    if w.endswith("ies") and len(w) > 4:
        out.append(w[:-3] + "y")
    if w.endswith("ves") and len(w) > 4:
        out += [w[:-3] + "f", w[:-3] + "fe"]
    if re.search(r"(ss|x|z|ch|sh)es$", w):
        out.append(w[:-2])
    if w.endswith("oes") and len(w) > 4:
        out.append(w[:-2])
    if w.endswith("s") and not w.endswith("ss") and len(w) > 2:
        out.append(w[:-1])
    return out or [w]


def is_plural_form(word, known=None):
    w = word.lower()
    if w in _NOT_PLURAL:
        return False
    if w in _IREG_PLURAL_REV:
        return True
    if not w.endswith("s") or w.endswith(("ss", "us", "is")):
        return False
    return known is None or any(known(c) for c in singular_candidates(w))


def verb_forms(lemma):
    """{'VB','VBP','VBZ','VBD','VBN','VBG'} of a verb."""
    b = lemma.lower()
    if b == "be":
        return {"VB": "be", "VBP": "are", "VBZ": "is", "VBD": "was", "VBN": "been", "VBG": "being"}
    if b == "have":
        return {"VB": "have", "VBP": "have", "VBZ": "has", "VBD": "had", "VBN": "had", "VBG": "having"}
    if re.search(r"[^aeiou]y$", b):
        z = b[:-1] + "ies"
    elif re.search(r"(s|x|z|ch|sh|o)$", b):
        z = b + "es"
    else:
        z = b + "s"
    if b.endswith("ie"):
        g = b[:-2] + "ying"
    elif b.endswith("e") and not b.endswith(("ee", "ye", "oe")):
        g = b[:-1] + "ing"
    elif _cvc_double(b):
        g = b + b[-1] + "ing"
    else:
        g = b + "ing"
    if b in IRREG_VERBS:
        p, pp = IRREG_VERBS[b]
    else:
        if b.endswith("e"):
            p = b + "d"
        elif re.search(r"[^aeiou]y$", b):
            p = b[:-1] + "ied"
        elif _cvc_double(b):
            p = b + b[-1] + "ed"
        else:
            p = b + "ed"
        pp = p
    return {"VB": b, "VBP": b, "VBZ": z, "VBD": p, "VBN": pp, "VBG": g}


def verb_lemma_candidates(word):
    """[(lemma, tag)] possible readings of an inflected form, most likely first (the caller keeps the first known verb)."""
    w = word.lower()
    out = []
    if w in BE_FORMS:
        return [("be", BE_FORMS[w])]
    for lem, tag in _IREG_VERB_REV.get(w, []):
        out.append((lem, tag))
    if w.endswith("ing") and len(w) > 4:
        s = w[:-3]
        out += [(s, "VBG"), (s + "e", "VBG")] + ([(s[:-1], "VBG")] if len(s) > 1 and s[-1] == s[-2] else []) + ([(s[:-2] + "ie", "VBG")] if s.endswith("y") else [])
    if w.endswith("ied") and len(w) > 4:
        out.append((w[:-3] + "y", "VBD"))
    elif w.endswith("ed") and len(w) > 3:
        s = w[:-2]
        out += [(s, "VBD"), (s + "e", "VBD")] + ([(s[:-1], "VBD")] if len(s) > 1 and s[-1] == s[-2] else [])
    if w.endswith("ies") and len(w) > 4:
        out.append((w[:-3] + "y", "VBZ"))
    elif w.endswith("es") and len(w) > 3:
        out += [(w[:-2], "VBZ"), (w[:-1], "VBZ")]
    elif w.endswith("s") and not w.endswith("ss") and len(w) > 2:
        out.append((w[:-1], "VBZ"))
    out.append((w, "VB"))

    def penalty(c):
        lem = c[0]
        if len(lem) < 2:
            return 2
        return int((len(lem) > 2 and lem[-1] == lem[-2] and lem[-1] in "bdgmnprt") or lem[-1] in "cv")
    return sorted(out, key=penalty)        # stable sort: keeps the generation order among equals


# ---------- API compatible with lemminflect (only what the LCM used) ----------
def getInflection(lemma, tag, **_):
    if tag in ("NNS", "NNPS"):
        return (plural(lemma),)
    if tag in ("NN", "NNP"):
        return (lemma,)
    f = verb_forms(lemma)
    return (f[tag],) if tag in f else ()


def getAllInflections(lemma, upos="VERB", **_):
    if upos == "VERB":
        return {t: (f,) for t, f in verb_forms(lemma).items()}
    if upos == "NOUN":
        return {"NN": (lemma,), "NNS": (plural(lemma),)}
    return {}


def getLemma(word, upos="NOUN", **_):
    if upos == "NOUN":
        return (singular_candidates(word)[0],) if is_plural_form(word) else (word,)
    return (verb_lemma_candidates(word)[0][0],)
