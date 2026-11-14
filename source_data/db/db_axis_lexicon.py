# db_axis_lexicon.py
#
# Lexicon classified by cardinal axis (north/south/east/west). This is separate from the thematic lexicon
# (db_lexicon.py: animal, person, object...): it classifies words by their geometric function in the model,
# not by topic.
#
#   north (viability)    -> future / possibility / "will do"   | subject: we, us
#   south (inviability)  -> past / impossibility / "done it"   | subject: that, it (impersonal)
#   east  (local / I)    -> raw perception from one sense, no comparison between parts
#                           (how it feels, looks, sounds, tastes, smells, as is)
#   west  (global / you) -> relation between parts / judgment against a standard
#                           (structure, proportion, aesthetic or moral evaluation)
#
# Quick test for a new quality word (east vs west):
#   single-sense raw reading, nothing compared       -> east
#   compares parts, or measures against a standard   -> west
#
# First pass only: to be reviewed and expanded with synonyms (WordNet synsets, thesaurus or a manual list).


# ---------- vertical axis: north / south (tense/modality, sentence subject) ----------

AXIS_NORTH = {
    "subjects": [
        "we", "us", "our", "ours", "ourselves",
    ],
    # "will do" - future / possibility / intention
    "modal_verbs": [
        "will", "would", "can", "could", "may", "might", "shall",
        "going to", "about to", "plan to", "intend to", "hope to",
        "aim to", "expect to", "look forward to",
    ],
    # concepts that only exist projected forward
    "concepts": [
        "hope", "expectation", "project", "goal", "plan", "dream",
        "ambition", "opportunity", "promise", "potential", "future",
        "prospect", "aspiration", "possibility", "vision", "intention",
    ],
    # interrogatives - a question opens toward possibility, not a closed fact
    "interrogatives": [
        "what", "who", "when", "where", "why", "how", "which",
        "whose", "what if",
    ],
    # affirmatives - clean, resolved agreement (no hedging, no negation)
    "affirmatives": [
        "yes", "sure", "of course", "why not", "absolutely",
        "definitely", "certainly", "i don't mind", "no problem",
        "gladly", "sounds good", "agreed",
    ],
}

AXIS_SOUTH = {
    "subjects": [
        "that", "it", "those", "them", "itself",
    ],
    # "done it" - past / impossibility / negation
    "modal_verbs": [
        "was", "were", "did", "had", "couldn't", "cannot", "can't",
        "won't", "wouldn't", "shouldn't", "didn't", "wasn't", "isn't",
        "never", "no longer", "impossible", "unable to", "failed to",
        "ended", "finished", "gave up",
    ],
    # negative auxiliary contractions - fixed forms, don't depend on the base
    # verb, so they live here instead of in the generated per-verb lexicon
    "negative_auxiliaries": [
        "isn't", "aren't", "wasn't", "weren't",
        "don't", "doesn't", "didn't",
        "haven't", "hasn't", "hadn't",
        "won't", "wouldn't",
        "can't", "cannot", "couldn't",
        "shouldn't", "mustn't", "mightn't", "shan't",
        "not",
    ],
    "concepts": [
        "failure", "loss", "end", "impossibility", "negation", "regret",
        "past", "history", "ruin", "defeat", "resignation", "dead end",
        "limit", "collapse", "downfall",
    ],
    # exclamations - a declaration slammed shut, no room left to negotiate
    "exclamations": [
        "no!", "never!", "stop!", "enough!", "how dare you!",
        "unbelievable!", "get out!", "absolutely not!", "that's it!",
    ],
}


# ---------- horizontal axis: east / west (qualities: raw perception vs. relation/judgment) ----------

AXIS_EAST = {
    "subjects": [
        "I", "me", "my", "mine", "myself",
    ],
    "touch": [
        "hard", "soft", "rough", "smooth", "heavy", "light", "hot",
        "cold", "warm", "icy", "wet", "dry", "sticky", "sharp",
        "pointed", "slippery", "firm", "flexible", "rigid", "coarse",
    ],
    "sight": [
        "red", "blue", "green", "yellow", "black", "white", "gray",
        "dark", "bright", "opaque", "transparent", "big", "large",
        "small", "tiny", "tall", "short", "wide", "narrow", "round",
        "square", "curved", "straight", "thin", "thick",
    ],
    "hearing": [
        "loud", "quiet", "silent", "high-pitched", "low-pitched",
        "noisy", "shrill", "muffled", "deafening",
    ],
    "taste": [
        "sweet", "salty", "bitter", "sour", "spicy", "bland",
        "tasteless", "stale", "savory",
    ],
    "smell": [
        "fragrant", "aromatic", "smelly", "stinky", "odorless",
        "pungent", "musty",
    ],
    "perceived_motion_time": [
        "fast", "slow", "quick", "swift", "sluggish", "abrupt", "brisk",
        "sudden",
    ],
}

AXIS_WEST = {
    "subjects": [
        "you", "your", "yours", "yourself",
    ],
    "aesthetic": [
        "beautiful", "pretty", "ugly", "elegant", "harmonious",
        "symmetrical", "asymmetrical", "balanced", "unbalanced",
        "proportionate", "disproportionate", "attractive",
        "sophisticated", "vulgar", "refined", "graceful",
    ],
    "moral_value": [
        "good", "bad", "fair", "unfair", "right", "wrong", "honest",
        "dishonest", "noble", "cruel", "generous", "selfish", "loyal",
        "treacherous", "decent", "corrupt", "virtuous",
    ],
    "general_evaluation": [
        "interesting", "boring", "admirable", "mediocre", "excellent",
        "terrible", "valuable", "useless", "impressive",
        "disappointing", "outstanding", "worthy", "unworthy",
        "remarkable",
    ],
}


# ---------- dynamic subjects (no fixed axis position) ----------
# Unlike fixed-value nouns (dog, grandfather, uncle), these "descriptive subjects" behave like pronouns: a speaker
# can address someone through them, but their position is not hardcoded here. It depends on the relationship
# between the speakers, and once that puts one of them near a quadrant it gets named the way that quadrant names
# things (We/You/I/That).
# They are their own category so they are neither dropped like plain nouns nor fixed like real pronouns.
DYNAMIC_SUBJECTS = [
    "friend", "enemy", "villain", "hero", "rival", "ally", "stranger",
    "traitor", "companion", "leader", "follower", "mentor", "student",
    "protector", "threat", "target", "guardian", "opponent", "partner",
    "outsider", "intruder", "captor", "hostage", "witness", "accomplice",
]


# ---------- negation as an operator: inversion vs. dampening ----------
# Negation ("not"/"don't"/"never"...) doesn't always push to south by itself. It acts on whatever follows it:
#
#   1. inversion: negating a word that is already negative ("mind", "dislike", "refuse", "doubt") flips the
#      sign, south x south = north.
#         "I don't mind"   -> don't (south) + mind (already negative) -> resolved affirmation -> north
#
#   2. plain push: negating a neutral/positive verb goes straight to south.
#         "I don't want"   -> don't (south) + want (neutral/positive) -> clean negation -> south
#
#   3. dampening (a different operator, not negation): a possibility modal (might/could/may) before a negation
#      doesn't invert the sign, it reduces the magnitude, so the result stays near the center: low-confidence
#      doubt.
#         "I might not mind" -> might (dampens, doesn't flip) -> near center -> doubt, not north
#
# In practice:
#   - verbs that are inherently negative (mind, dislike, refuse, doubt, resent, deny, reject...) need a flag in
#     the lexicon so the inversion rule can find them. TODO: tag them in db_axis_lexicon_generated.py (not
#     marked yet).
#   - possibility modals (might/could/may/perhaps/maybe) are never treated as negation: they change the
#     magnitude, not the sign.
INHERENTLY_NEGATIVE_VERBS = [
    "mind", "dislike", "refuse", "doubt", "resent", "deny", "reject",
    "hate", "avoid", "distrust", "regret", "resist", "oppose",
]

POSSIBILITY_DAMPENERS = [
    "might", "could", "may", "perhaps", "maybe", "possibly",
    "presumably", "conceivably",
]


# ---------- notes ----------
# - This file doesn't replace db_lexicon.py (the thematic lexicon built from WordNet lexnames). It's a layer
#   on top: any "quality" word from db_lexicon.py should end up tagged AXIS_EAST or AXIS_WEST with the test above.
# - Open questions:
#     can a word belong to more than one axis depending on context? ("heavy" physically = east, but
#     "a heavy situation" = west?)
#     do we auto-expand each list with WordNet similar_tos(), or keep a hand-picked core and only expand that?
#     (safer, less noise)
# - Next: run the "quality" words of db_lexicon.py through the quick test one by one instead of writing this list
#   from scratch, so nothing in the real lexicon gets duplicated or missed.

