"""Language profiles (English, Spanish): generic format features (person, tense, mood, negation) and language detection.
The lists are short and of grammatical function, not of meaning. The lexicon thresholds were measured ONLY on technical English."""
import re, unicodedata

LANGS = {
 "en": dict(i={"i", "me", "my", "mine", "i'm", "i'll", "i'd", "i've", "myself"}, you={"you", "your", "yours", "you're", "you've", "yourself"},
            we={"we", "us", "our", "ours", "let's", "we'll", "we're"}, it={"it", "they", "them", "there", "it's", "they're"},
            neg={"no", "not", "never", "nothing", "nobody", "none"}, past={"was", "were", "had", "did"}, fut={"will", "would", "could", "might", "maybe", "going"}),
 "es": dict(i={"yo", "me", "mi", "mí", "mis", "mío", "mía", "mismo", "conmigo"}, you={"tú", "tu", "tus", "te", "ti", "vos", "usted", "ustedes", "vosotros", "contigo", "tuyo", "tuya"},
            we={"nosotros", "nosotras", "nos", "nuestro", "nuestra", "nuestros", "nuestras", "vamos", "juntos", "juntas"}, it={"él", "ella", "ellos", "ellas", "eso", "esto", "hay", "se"},
            neg={"no", "nunca", "nada", "nadie", "ninguno", "ninguna", "jamás", "tampoco", "sin"},
            past={"fue", "fueron", "era", "eran", "había", "habían", "hubo", "estaba", "estaban", "estuvo", "tuvo", "hizo", "dijo", "vino", "vio", "murió"},
            fut={"mañana", "quizás", "quizá", "ojalá", "futuro", "posible", "podría", "podríamos", "será", "serán", "iremos", "podremos"}),
}
_STOP = {"en": {"the", "and", "you", "is", "to", "of", "that", "it", "i", "in", "was", "for", "are", "with", "this"},
         "es": {"el", "la", "los", "las", "de", "que", "y", "en", "un", "una", "es", "por", "con", "no", "para", "yo", "tú", "mi", "te", "se", "lo", "pero", "como"}}


def _words(text, lang):
    t = unicodedata.normalize("NFC", text.lower().replace("’", "'"))
    return re.findall(r"[a-záéíóúüñ']+" if lang == "es" else r"[a-z']+", t)


def detect_lang(text):
    w = re.findall(r"[^\W\d_]+", unicodedata.normalize("NFC", text.lower()))
    en, es = sum(x in _STOP["en"] for x in w), sum(x in _STOP["es"] for x in w)
    es += 2 * sum(c in "ñ¿¡" for c in text)
    return "es" if es > en * 1.2 else "en" if en > es * 1.2 else "unknown"


def format_features(text, lang="en"):
    L = LANGS[lang]; w = _words(text, lang); t = []
    first = next(("i" if x in L["i"] else "you" if x in L["you"] else "we" if x in L["we"] else "it" for x in w if x in L["i"] | L["you"] | L["we"] | L["it"]), "none")
    t.append("__first_" + first)
    for nm in ("i", "you", "we", "it"): t.append(f"__n{nm}{min(sum(x in L[nm] for x in w), 2)}")
    if "?" in text or "¿" in text: t.append("__q")
    if "!" in text or "¡" in text: t.append("__excl")
    if re.search(r"\b[A-ZÁÉÍÓÚÑ]{2,}\b", text): t.append("__caps")
    if any(x in L["neg"] or x.endswith("n't") for x in w): t.append("__neg")
    if lang == "en": past = any(x in L["past"] or (x.endswith("ed") and len(x) > 4) for x in w)
    else: past = any(x in L["past"] or (len(x) > 4 and x.endswith(("aba", "abas", "aban", "ió", "ieron", "aron", "ían"))) for x in w)
    if past: t.append("__past")
    if lang == "en": fut = any(x in L["fut"] or x.endswith("'ll") for x in w)
    else: fut = any(x in L["fut"] or (len(x) > 4 and x.endswith(("ré", "rás", "rá", "remos", "rán", "ría", "ríamos"))) for x in w) or "voy a" in text.lower() or "vamos a" in text.lower()
    if fut: t.append("__fut")
    return t
