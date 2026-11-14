"""lite_nlp: grammatical analyzer WITHOUT external libraries (it replaces spaCy in `lcm.parse`).

It offers what `parse` consumes from a document: tokens with text, lower_, lemma_, pos_, tag_, dep_, children and morph, and the
sentences. It is a deterministic grammar by PATTERNS (imperative, question with an auxiliary, copular, let's / let me, declarative),
not a statistical model. To decide what is a verb, a noun or an adjective it uses the categories of the LCM tables (action, quality,
biological / non_biological) plus a short list of common verbs and the morphology of `lcm.inflect`.
LIMIT: it covers the constructions that the LCM answers; free-form sentences outside those patterns are analyzed in a degraded way
(it keeps the first verb or noun as root and hangs the rest as 'dep').
"""
import re
from .inflect import BE_FORMS, IRREG_VERBS, verb_lemma_candidates, singular_candidates, is_plural_form

PRON_PERSONAL = {"i", "me", "you", "he", "him", "she", "her", "it", "we", "us", "they", "them", "myself", "yourself", "himself", "herself",
                 "itself", "ourselves", "themselves", "mine", "yours"}
SUBJ_PRON = {"i", "you", "he", "she", "it", "we", "they"}
POSS = {"my", "your", "his", "its", "our", "their"}                # "her" is handled apart (pronoun or possessive)
DET = {"a", "an", "the", "this", "that", "these", "those", "some", "any", "each", "every", "no", "another", "all", "both", "many", "few",
       "several", "much", "more", "most", "other"}
NUMS = {"zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve", "twenty", "hundred"}
WP = {"what", "who", "whom", "which", "whose"}
WRB = {"when", "where", "why", "how"}
AUX_BE = {"am", "is", "are", "was", "were", "be", "been", "being", "'m", "'re"}
AUX_DO = {"do", "does", "did"}
AUX_HAVE = {"have", "has", "had"}
MODALS = {"will", "would", "shall", "should", "can", "could", "may", "might", "must", "wo", "ca", "'ll", "'d"}
PREP = {"about", "of", "on", "to", "for", "in", "at", "over", "under", "with", "without", "from", "by", "regarding", "into", "onto", "through",
        "around", "near", "after", "before", "against", "between", "among", "during", "toward", "towards", "across", "behind", "beside", "like", "than"}
NOUN_PREP = {"about", "of", "on", "regarding", "with"}             # these prepositions attach to the nearest noun phrase
TIME_NP = {"yesterday", "today", "tomorrow", "tonight"}            # time nouns: dep npadvmod
TIME_ADV = {"now", "later", "soon", "ago", "earlier", "once", "previously", "eventually", "currently", "again", "always", "often", "sometimes",
            "still", "just", "also", "too", "very", "really", "here", "there", "then", "never", "before", "after"}
NEG = {"not", "n't", "never"}
CCONJ = {"and", "or", "but"}
COMMON_VERBS = set(IRREG_VERBS) | set((
    "like love hate want need play jump climb walk talk help ask answer call try use work live look show start stop turn open close move watch "
    "listen wait visit learn remember believe enjoy prefer miss hope wish dance laugh cry smile paint cook clean kill save study travel shout "
    "whisper count mix pick pull push pass kick kiss hug hunt race rest roll shop skate wonder imagine explain describe discuss mention "
    "suppose guess agree allow appear arrive ask carry change check decide enter expect follow happen invite join kick knock love notice "
    "offer order own plan point promise protect reach realize receive reply return smell stay taste thank touch trust wash wish worry "
    "tell go say see feel think know").split())
CONTR_N = {"don't": ("do", "n't"), "doesn't": ("does", "n't"), "didn't": ("did", "n't"), "won't": ("wo", "n't"), "can't": ("ca", "n't"),
           "isn't": ("is", "n't"), "aren't": ("are", "n't"), "wasn't": ("was", "n't"), "weren't": ("were", "n't"), "couldn't": ("could", "n't"),
           "wouldn't": ("would", "n't"), "shouldn't": ("should", "n't"), "haven't": ("have", "n't"), "hasn't": ("has", "n't")}
TOKEN_RE = re.compile(r"[A-Za-z]+(?:-[A-Za-z]+)*(?:['’][A-Za-z]+)?|\d+(?:[.,]\d+)*|[^\sA-Za-z0-9]")


class Morph(dict):
    def get(self, key, default=None):
        v = dict.get(self, key)
        return list(v) if v is not None else ([] if default is None else default)


class Token:
    def __init__(self, i, text):
        self.i, self.text, self.lower_ = i, text, text.lower().replace("’", "'")
        self.lemma_, self.pos_, self.tag_, self.dep_ = self.lower_, "X", "NN", "dep"
        self.head, self.children, self.morph = self, [], Morph()
        self.is_punct = not any(c.isalnum() for c in text)

    def __repr__(self):
        return f"{self.text}/{self.pos_}/{self.dep_}"


class Span(list):
    pass


class Doc:
    def __init__(self, tokens, sents):
        self.tokens, self._sents = tokens, sents

    def __iter__(self): return iter(self.tokens)
    def __len__(self): return len(self.tokens)
    def __getitem__(self, i): return self.tokens[i]
    @property
    def sents(self): return iter(self._sents)


def _attach(child, head, dep):
    child.head, child.dep_ = head, dep
    if child is not head:
        head.children.append(child)


def tokenize(text):
    out = []
    for m in TOKEN_RE.findall(text):
        low = m.lower().replace("’", "'")
        if low in CONTR_N:
            a, b = CONTR_N[low]
            out += [m[:len(a)], "n't"]
        elif "'" in low and low.split("'")[1] in ("s", "ll", "m", "re", "ve", "d") and low.split("'")[0]:
            stem, suf = m.split("'") if "'" in m else m.split("’")
            out += [stem, "'" + suf]
        else:
            out.append(m)
    return out


class Analyzer:
    def __init__(self, tables):
        self.t, self._hits_cache = tables, {}

    # ---------- lexical information from the tables ----------
    def _hits(self, w):
        if w not in self._hits_cache:
            hs = self.t.find(w) if self.t is not None else []
            noun = any(n.get("type") in ("biological", "non_biological") for _, n in hs)
            verb = any(n.get("type") == "action" or n.get("narrative_verb") for _, n in hs)
            adj = any(n.get("type") == "quality" for _, n in hs)
            self._hits_cache[w] = (noun, verb, adj, bool(hs))
        return self._hits_cache[w]

    def known_verb(self, lemma):
        return lemma in COMMON_VERBS or self._hits(lemma)[1]

    def verb_reading(self, w):
        """(lemma, tag) if `w` can be a verb form of a known verb, else None."""
        if w in BE_FORMS:
            return "be", BE_FORMS[w]
        if w in AUX_DO:
            return "do", {"do": "VBP", "does": "VBZ", "did": "VBD"}[w]
        if w in AUX_HAVE:
            return "have", {"have": "VBP", "has": "VBZ", "had": "VBD"}[w]
        for lem, tag in verb_lemma_candidates(w):
            if self.known_verb(lem):
                return lem, ("VB" if lem == w else tag)
        return None

    def _is_known_noun(self, w):
        return self._hits(w)[3] or w in COMMON_VERBS

    def is_adjective(self, w):
        n, v, a, any_ = self._hits(w)
        return a and not n

    # ---------- NP ----------
    def _nominal(self, t, after_mod):
        lw = t.lower_
        if lw in (AUX_BE | AUX_DO | AUX_HAVE | MODALS | PREP | CCONJ | NEG | WP | WRB | TIME_NP | TIME_ADV) or t.is_punct:
            return False
        n, v, a, any_ = self._hits(lw)
        if n or a:
            return True
        vr = self.verb_reading(lw)
        if vr is not None:                                   # verb reading only
            return after_mod and vr[1] == "VB"                # "a duck": a base verb after a determiner or adjective is a noun
        return True                                           # unknown word: noun

    def np(self, words, i):
        """Noun phrase from i: (head, [(tok, dep)], next_i) or None."""
        n = len(words)
        if i >= n:
            return None
        w0 = words[i]
        if w0.lower_ in PRON_PERSONAL and w0.lower_ != "her":
            return w0, [], i + 1
        if w0.lower_ == "her":
            nxt = words[i + 1] if i + 1 < n else None
            if not (nxt is not None and self._nominal(nxt, True) and nxt.lower_ not in PREP):
                return w0, [], i + 1                          # "her" pronoun
        mods, j, det_seen = [], i, False
        w = words[j]
        if w.lower_ in POSS or w.lower_ == "her":
            mods.append((w, "poss")); j += 1; det_seen = True
        elif w.lower_ in NUMS or w.text.isdigit():
            mods.append((w, "nummod")); j += 1; det_seen = True
        elif w.lower_ in DET:
            mods.append((w, "det")); j += 1; det_seen = True
        run = []
        while j < n and self._nominal(words[j], det_seen or bool(run)):
            run.append(words[j]); j += 1
        while len(run) >= 2 and self.is_adjective(run[-1].lower_):     # "the sky blue": the final adjective is the complement, not the head
            run.pop(); j -= 1
        if not run:
            return None
        head = run[-1]
        for m in run[:-1]:
            mods.append((m, "amod" if self.is_adjective(m.lower_) else "compound"))
        return head, mods, j

    def _tag_np(self, head, mods, dep_head):
        lw = head.lower_
        if lw in PRON_PERSONAL or lw == "her":
            head.pos_, head.tag_ = "PRON", "PRP"
        else:
            cap = head.text[:1].isupper() and head.i > 0
            plural = is_plural_form(lw, self._is_known_noun)
            if cap:
                head.pos_, head.tag_ = "PROPN", "NNP"
            else:
                head.pos_, head.tag_ = "NOUN", "NNS" if plural else "NN"
            cands = singular_candidates(lw) if plural else [lw]
            head.lemma_ = next((c for c in cands if self._hits(c)[3]), cands[0])
            if plural:
                head.morph["Number"] = ["Plur"]
        for m, dep in mods:
            m.dep_ = dep
            if dep == "det": m.pos_, m.tag_ = "DET", "DT"
            elif dep == "poss": m.pos_, m.tag_ = "PRON" if m.lower_ != "her" else "PRON", "PRP$"
            elif dep == "nummod": m.pos_, m.tag_ = "NUM", "CD"
            elif dep == "amod": m.pos_, m.tag_ = "ADJ", "JJ"
            else: m.pos_, m.tag_ = "NOUN", "NN"
            _attach(m, head, dep)

    # ---------- verbs ----------
    def _set_verb(self, tok, lemma, tag, aux=False, imperative=False):
        tok.lemma_, tok.tag_ = lemma, tag
        tok.pos_ = "AUX" if aux else "VERB"
        if tag in ("VBD",) or tok.lower_ in ("was", "were", "had", "did"):
            tok.morph["Tense"] = ["Past"]; tok.morph["VerbForm"] = ["Fin"]
        elif tag in ("VBZ", "VBP"):
            tok.morph["Tense"] = ["Pres"]; tok.morph["VerbForm"] = ["Fin"]
        elif tag == "VBG":
            tok.morph["VerbForm"] = ["Part"]; tok.morph["Tense"] = ["Pres"]
        elif tag == "VB":
            tok.morph["VerbForm"] = ["Inf"]
        if imperative:
            tok.morph["Mood"] = ["Imp"]

    def _set_aux(self, tok):
        lw = tok.lower_
        if lw in AUX_BE:
            self._set_verb(tok, "be", BE_FORMS.get(lw, "VBP"), aux=True)
        elif lw in AUX_DO:
            self._set_verb(tok, "do", {"do": "VBP", "does": "VBZ", "did": "VBD"}[lw], aux=True)
        elif lw in AUX_HAVE:
            self._set_verb(tok, "have", {"have": "VBP", "has": "VBZ", "had": "VBD"}[lw], aux=True)
        else:
            lem = {"wo": "will", "ca": "can", "'ll": "will", "'d": "would"}.get(lw, lw)
            tok.lemma_, tok.pos_, tok.tag_ = lem, "AUX", "MD"
            tok.morph["VerbForm"] = ["Fin"]
            if lw in ("would", "should", "could", "might"): pass

    # ---------- arguments of a verb ----------
    def _args(self, root, words, k, last_np=None):
        n = len(words)
        be = root.lemma_ == "be"
        while k < n:
            w = words[k]; lw = w.lower_
            if lw in NEG:
                w.pos_, w.tag_ = "PART", "RB"; _attach(w, root, "neg"); k += 1; continue
            if lw in TIME_NP:
                w.pos_, w.tag_ = "NOUN", "NN"; _attach(w, root, "npadvmod"); k += 1; continue
            if lw in TIME_ADV:
                w.pos_, w.tag_ = "ADV", "RB"; _attach(w, root, "advmod"); k += 1; continue
            if lw in CCONJ:
                w.pos_, w.tag_ = "CCONJ", "CC"; _attach(w, root, "cc"); k += 1; continue
            if lw == "to" and k + 1 < n and self.verb_reading(words[k + 1].lower_) and not self._hits(words[k + 1].lower_)[0]:
                w.pos_, w.tag_ = "PART", "TO"; v = words[k + 1]
                lem, _tag = self.verb_reading(v.lower_); self._set_verb(v, lem, "VB")
                _attach(v, root, "xcomp"); _attach(w, v, "aux"); k = self._args(v, words, k + 2); continue
            if lw in PREP:
                w.pos_, w.tag_ = "ADP", "IN"
                r = self.np(words, k + 1)
                if r is None:
                    _attach(w, root, "prep"); k += 1; continue
                head, mods, nk = r
                self._tag_np(head, mods, "pobj")
                anchor = last_np if (lw in NOUN_PREP and last_np is not None) else root
                _attach(w, anchor, "prep"); _attach(head, w, "pobj"); k = nk; continue
            if be and self.is_adjective(lw) or (be and self._hits(lw)[2] and (k + 1 >= n or words[k + 1].is_punct or words[k + 1].lower_ in TIME_NP | TIME_ADV | PREP)):
                w.pos_, w.tag_ = "ADJ", "JJ"; _attach(w, root, "acomp"); k += 1; continue
            r = self.np(words, k)
            if r is not None:
                head, mods, nk = r
                is_pron = head.lower_ in PRON_PERSONAL or head.lower_ == "her"
                nxt = self.np(words, nk) if (is_pron and nk < n and words[nk].lower_ not in PREP) else None
                self._tag_np(head, mods, "dobj")
                if be:
                    _attach(head, root, "attr")
                elif is_pron and nxt is not None:
                    _attach(head, root, "dative")            # "tell ME a story"
                else:
                    _attach(head, root, "dobj")              # "like ducks", "tell me about X" (me = object)
                last_np = None if is_pron else head
                k = nk; continue
            w.pos_, w.tag_ = ("ADV", "RB") if w.is_punct is False and self._hits(lw)[2] else ("X", "NN")
            _attach(w, root, "dep" if not w.is_punct else "punct"); k += 1
        return k

    # ---------- sentence ----------
    def sentence(self, toks):
        words = [t for t in toks if not t.is_punct]
        puncts = [t for t in toks if t.is_punct]
        if not words:
            for p in puncts: p.pos_, p.tag_, p.dep_ = "PUNCT", ".", "ROOT" if p is puncts[0] else "punct"
            return
        n, i, wh = len(words), 0, None
        root = None
        if words[0].lower_ in WP | WRB:
            wh = words[0]; i = 1
            wh.pos_, wh.tag_ = ("PRON", "WP") if wh.lower_ in WP else ("ADV", "WRB")
        w = words[i] if i < n else None
        lw = w.lower_ if w is not None else ""

        # --- let's / let me ---
        if lw == "let" and i + 1 < n and words[i + 1].lower_ in ("'s", "me", "us"):
            self._set_verb(w, "let", "VB", imperative=True); w.dep_ = "ROOT"; root = w
            subj = words[i + 1]; subj.lemma_ = "us" if subj.lower_ == "'s" else subj.lower_; subj.pos_, subj.tag_ = "PRON", "PRP"
            vr = self.verb_reading(words[i + 2].lower_) if i + 2 < n else None
            if vr:
                v = words[i + 2]; self._set_verb(v, vr[0], "VB"); _attach(v, root, "ccomp"); _attach(subj, v, "nsubj")
                self._args(v, words, i + 3)
            else:
                _attach(subj, root, "dobj")
        # --- negative imperative: Don't tell ... ---
        elif lw in AUX_DO and i + 2 < n and words[i + 1].lower_ == "n't" and self.verb_reading(words[i + 2].lower_):
            lem, _ = self.verb_reading(words[i + 2].lower_); v = words[i + 2]
            self._set_verb(v, lem, "VB", imperative=True); v.dep_ = "ROOT"; root = v
            self._set_aux(w); _attach(w, v, "aux"); words[i + 1].pos_, words[i + 1].tag_ = "PART", "RB"; _attach(words[i + 1], v, "neg")
            self._args(v, words, i + 3)
        # --- question with do / modal: [wh] AUX SUBJ [not] V ... ---
        elif (lw in AUX_DO | MODALS or (lw in AUX_HAVE and wh is None)) and i + 1 < n and (self.np(words, i + 1) is not None):
            r = self.np(words, i + 1); head, mods, k = r
            self._tag_np(head, mods, "nsubj")
            neg = words[k] if k < n and words[k].lower_ in NEG else None
            k2 = k + (1 if neg else 0)
            vr = self.verb_reading(words[k2].lower_) if k2 < n else None
            if vr is not None:
                v = words[k2]; self._set_verb(v, vr[0], "VB"); v.dep_ = "ROOT"; root = v
                self._set_aux(w); _attach(w, v, "aux"); _attach(head, v, "nsubj")
                if neg: neg.pos_, neg.tag_ = "PART", "RB"; _attach(neg, v, "neg")
                if wh is not None and wh.lower_ in WP:
                    _attach(wh, v, "dobj")
                elif wh is not None:
                    _attach(wh, v, "advmod")
                self._args(v, words, k2 + 1, None)
            else:
                root = None                                  # not a verb after the subject: falls to the degraded case
        # --- copular: [wh] BE SUBJ complement ---
        elif lw in AUX_BE and i + 1 < n and self.np(words, i + 1) is not None:
            r = self.np(words, i + 1); head, mods, k = r
            self._tag_np(head, mods, "nsubj")
            vr = self.verb_reading(words[k].lower_) if k < n and words[k].lower_.endswith("ing") else None
            if vr is not None and vr[1] == "VBG":            # "How are you doing?": progressive
                v = words[k]; self._set_verb(v, vr[0], "VBG"); v.dep_ = "ROOT"; root = v
                self._set_aux(w); _attach(w, v, "aux"); _attach(head, v, "nsubj")
                if wh is not None: _attach(wh, v, "dobj" if wh.lower_ in WP else "advmod")
                self._args(v, words, k + 1, None)
            else:
                self._set_aux(w); w.dep_ = "ROOT"; root = w
                _attach(head, w, "nsubj")
                if wh is not None: _attach(wh, w, "attr" if wh.lower_ in WP else "advmod")
                self._args(w, words, k, head if mods or head.pos_ == "NOUN" else None)
        # --- imperative: V ... ---
        elif w is not None and wh is None:
            vr = self.verb_reading(lw)
            nxt = words[i + 1].lower_ if i + 1 < n else ""
            declarative_hint = nxt in (AUX_BE | AUX_DO | AUX_HAVE | MODALS) or (self.verb_reading(nxt) and self.verb_reading(nxt)[1] in ("VBD", "VBZ") and not self._hits(nxt)[0])
            if vr is not None and vr[1] == "VB" and lw not in PRON_PERSONAL and lw not in DET and not declarative_hint:
                self._set_verb(w, vr[0], "VB", imperative=True); w.dep_ = "ROOT"; root = w
                self._args(w, words, i + 1)
        # --- declarative: SUBJ [aux] [not] V/complement ---
        if root is None and i < n:
            r = self.np(words, i)
            if r is not None:
                head, mods, k = r
                self._tag_np(head, mods, "nsubj")
                aux_toks = []
                while k < n and words[k].lower_ in (AUX_BE | AUX_DO | AUX_HAVE | MODALS):
                    aux_toks.append(words[k]); k += 1
                negs = []
                while k < n and words[k].lower_ in NEG:
                    negs.append(words[k]); k += 1
                vr = self.verb_reading(words[k].lower_) if k < n else None
                going = bool(aux_toks and aux_toks[-1].lower_ == "going")
                if aux_toks and aux_toks[-1].lower_ in AUX_BE and (vr is None or vr[1] not in ("VBG", "VBN", "VB", "VBD") or not k < n or words[k].lower_ in PREP) :
                    # copular: SUBJ be [not] complement
                    cop = aux_toks[-1]; self._set_aux(cop); cop.dep_ = "ROOT"; root = cop
                    for a in aux_toks[:-1]: self._set_aux(a); _attach(a, cop, "aux")
                    _attach(head, cop, "nsubj")
                    for ng in negs: ng.pos_, ng.tag_ = "PART", "RB"; _attach(ng, cop, "neg")
                    if wh is not None: _attach(wh, cop, "advmod")
                    self._args(cop, words, k, head if head.pos_ in ("NOUN", "PROPN") else None)
                elif vr is not None:
                    v = words[k]
                    tag = vr[1] if not aux_toks else ("VBG" if vr[1] == "VBG" else "VB" if vr[1] in ("VB", "VBP", "VBD") and aux_toks[-1].lower_ in (MODALS | AUX_DO) else vr[1])
                    if not aux_toks and tag == "VB":
                        tag = "VBP"                           # present with I / you / we / they
                    self._set_verb(v, vr[0], tag); v.dep_ = "ROOT"; root = v
                    for a in aux_toks: self._set_aux(a); _attach(a, v, "aux")
                    _attach(head, v, "nsubj")
                    for ng in negs: ng.pos_, ng.tag_ = "PART", "RB"; _attach(ng, v, "neg")
                    if wh is not None: _attach(wh, v, "advmod")
                    self._args(v, words, k + 1, None)
                elif aux_toks:
                    cop = aux_toks[-1]; self._set_aux(cop); cop.dep_ = "ROOT"; root = cop
                    _attach(head, cop, "nsubj"); self._args(cop, words, k, None)
        # --- degraded case: first verb, else first noun ---
        if root is None:
            vtok = next((t for t in words if self.verb_reading(t.lower_) and not self._hits(t.lower_)[0]), None)
            if vtok is not None:
                lem, tag = self.verb_reading(vtok.lower_); self._set_verb(vtok, lem, tag); root = vtok
            else:
                root = words[0]; root.pos_, root.tag_ = "X", "NN"
                lw0 = root.lower_
                if self._hits(lw0)[0] or lw0 not in PRON_PERSONAL: root.pos_ = "NOUN"
            root.dep_ = "ROOT"
            for t in words:
                if t is not root and t.head is t and t.dep_ == "dep":
                    t.pos_ = t.pos_ if t.pos_ != "X" else "X"; _attach(t, root, "dep")
        for p in puncts:
            p.pos_, p.tag_ = "PUNCT", "."; _attach(p, root, "punct")
        for t in toks:
            t.children.sort(key=lambda c: c.i)


_AZ = {}


def _analyzer_for(tables):
    """One analyzer per tables object (it keeps a cache of the lookups, so analyzing many sentences is fast)."""
    hit = _AZ.get(id(tables))
    if hit is None or hit[0] is not tables:
        hit = _AZ[id(tables)] = (tables, Analyzer(tables))
    return hit[1]


def analyze(text, tables=None):
    """Text -> Doc (tokens with grammatical attributes, same as the ones `lcm.parse` used from spaCy)."""
    toks = [Token(i, s) for i, s in enumerate(tokenize(text))]
    az = _analyzer_for(tables)
    sents, cur = [], []
    for k, t in enumerate(toks):
        cur.append(t)
        if t.text in (".", "?", "!") and k + 1 < len(toks):
            sents.append(cur); cur = []
    if cur:
        sents.append(cur)
    for s in sents:
        az.sentence(s)
    return Doc(toks, [Span(s) for s in sents])
