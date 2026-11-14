# db/db_lexicon_overrides.py
#
# Sense corrections (homonyms) on top of db/db_lexicon.py.
#
# The lexicon was built taking the first WordNet sense of each word (synsets[0]). WordNet orders senses by
# frequency in general English (SemCor), but story text often uses a different sense, so words whose dominant
# sense in a story isn't the first one ended up misclassified:
#
#   "blade"  -> plant/biological  (blade of grass)   -> taken as a living subject
#   "raven"  -> action            (verb "to devour") -> rejected as a noun
#   "staff"  -> group/biological  (personnel/staff)
#   "club"   -> group/biological  (baseball club)
#   ...
#
# LEXICON_OVERRIDES: replaces fields of existing entries.
# LEXICON_ADDITIONS: adds common story words the lexicon didn't cover (without them resolve_concept returns None
#                    and the word is invisible). No emotional load of their own, they only make the word
#                    visible/classifiable.
# apply_overrides: called at the end of db/db_lexicon.py.
#
# db_lexicon.py is generated (1.4 MB) and not edited by hand: the corrections live here, where they are easy to
# review and extend.
#
# Criterion: only cases where the story sense is clearly the dominant one. The truly ambiguous ones (bolt, heart,
# spike, spell, mark...) are not touched here: they are listed in AMBIGUOUS_FOR_REVIEW, to be decided.

# field to replace -> new value. Only the fields that appear are overwritten.
LEXICON_OVERRIDES = {
    # ---------- weapons / objects that WordNet resolves to another sense ----------
    "blade":  {"category": "object", "node_type": "non_biological", "subtype": "weapon",
               "synonyms": ["blade", "sword", "steel"]},
    "club":   {"category": "object", "node_type": "non_biological", "subtype": "weapon",
               "synonyms": ["club", "cudgel", "bludgeon"]},
    "staff":  {"category": "object", "node_type": "non_biological", "subtype": "misc",
               "synonyms": ["staff", "quarterstaff", "walking stick"]},
    "arrow":  {"category": "object", "node_type": "non_biological", "subtype": "weapon",
               "synonyms": ["arrow", "shaft"]},
    "crown":  {"category": "object", "node_type": "non_biological", "subtype": "misc",
               "synonyms": ["crown", "diadem", "coronet"]},
    "ring":   {"category": "object", "node_type": "non_biological", "subtype": "misc",
               "synonyms": ["ring", "band", "signet"]},
    "chain":  {"category": "object", "node_type": "non_biological", "subtype": "misc",
               "synonyms": ["chain", "shackle", "fetter"]},

    # ---------- living beings wrongly resolved ----------
    # (raven: was a VERB -> it was discarded as a subject)
    "raven":  {"category": "animal", "node_type": "biological",
               "synonyms": ["raven", "corvid"]},
    "bear":   {"category": "animal", "node_type": "biological",
               "synonyms": ["bear", "grizzly"]},
    "queen":  {"category": "person", "node_type": "biological",
               "synonyms": ["queen", "female monarch", "sovereign"]},
    "dragon": {"category": "animal", "node_type": "biological",
               "synonyms": ["dragon", "wyrm", "drake"]},
}

# Common story words that were not in the lexicon. Same schema as
# the rest: {category, node_type, subtype?, synonyms}.
LEXICON_ADDITIONS = {
    # weapons
    "dagger": {"category": "object", "node_type": "non_biological", "subtype": "weapon",
               "synonyms": ["dagger", "dirk", "stiletto"]},
    "spear":  {"category": "object", "node_type": "non_biological", "subtype": "weapon",
               "synonyms": ["spear", "pike", "lance"]},
    "mace":   {"category": "object", "node_type": "non_biological", "subtype": "weapon",
               "synonyms": ["mace", "flail", "cudgel"]},
    "sling":  {"category": "object", "node_type": "non_biological", "subtype": "weapon",
               "synonyms": ["sling", "slingshot"]},
    "sabre":  {"category": "object", "node_type": "non_biological", "subtype": "weapon",
               "synonyms": ["sabre", "saber", "cutlass"]},
    "saber":  {"category": "object", "node_type": "non_biological", "subtype": "weapon",
               "synonyms": ["saber", "sabre", "cutlass"]},
    "dart":   {"category": "object", "node_type": "non_biological", "subtype": "weapon",
               "synonyms": ["dart", "javelin"]},
    "torch":  {"category": "object", "node_type": "non_biological", "subtype": "misc",
               "synonyms": ["torch", "brand", "firebrand"]},
    # creatures
    "hound":  {"category": "animal", "node_type": "biological", "synonyms": ["hound", "dog"]},
    "stag":   {"category": "animal", "node_type": "biological", "synonyms": ["stag", "hart", "deer"]},
    "boar":   {"category": "animal", "node_type": "biological", "synonyms": ["boar", "wild pig"]},
    "nun":    {"category": "person", "node_type": "biological", "synonyms": ["nun", "sister"]},
    # creature parts
    "fang":   {"category": "body_part", "node_type": "biological", "subtype": "external",
               "synonyms": ["fang", "tusk"]},
    "claw":   {"category": "body_part", "node_type": "biological", "subtype": "external",
               "synonyms": ["claw", "talon"]},
    "talon":  {"category": "body_part", "node_type": "biological", "subtype": "external",
               "synonyms": ["talon", "claw"]},
    # environment
    "mist":   {"category": "weather", "node_type": "non_biological",
               "synonyms": ["mist", "haze", "vapor"]},
}

# truly ambiguous: both senses are plausible in stories and choosing
# one is a design call, not a bug. Not touched here, listed for review.
AMBIGUOUS_FOR_REVIEW = {
    "bolt":  "lightning (today: weather) vs. crossbow bolt / door bolt (object)",
    "heart": "feeling (today: cognition) vs. organ (body_part)",
    "spike": "today: event; usually a sharp object in a scene",
    "spell": "today: state (trance); in fantasy it is a magic spell (act/object)",
    "mark":  "today: cognition (grade); in a scene it is usually a mark/scar",
    "bark":  "tree bark (plant) vs. dog bark (event) — separate verb",
}


def apply_overrides(lexicon: dict) -> dict:
    """Applies LEXICON_OVERRIDES and LEXICON_ADDITIONS on `lexicon` (in place).
    Returns the same dict. Idempotent: calling it twice changes nothing."""
    for word, fields in LEXICON_OVERRIDES.items():
        entry = lexicon.get(word)
        if entry is None:
            lexicon[word] = dict(fields)
            continue
        # if the override has no subtype, the old one does not apply to the new sense
        if "subtype" not in fields:
            entry.pop("subtype", None)
        entry.update(fields)
    for word, fields in LEXICON_ADDITIONS.items():
        if word not in lexicon:
            lexicon[word] = dict(fields)
    return lexicon
