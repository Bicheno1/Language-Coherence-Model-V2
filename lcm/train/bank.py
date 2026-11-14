"""Emotional banks: a character's dialogues grouped by axis (North / South / East / West) and category.

A bank is plain text. Two layouts are read, and both give the same result:

  My own layout (as in `data/lucia/lucia_bank.txt`)     The simplified layout
      NORTH (V) — We                                                   ## NORTH
      100 dialogues · ...   (free text is ignored)                     # 1 Fear of losing someone
      1. Fear of losing someone                                        Bernard: I'm leaving tomorrow, Lucia.
      1. Bernard: I'm leaving tomorrow, Lucia.                         Lucia: But... we just started spending time together.
      Lucia: But... we just started spending time together.

Rules
  * An axis heading is a line that starts with NORTH, SOUTH, EAST or WEST and has no colon. Axis letters follow `axes.AX`: N, S, E, W.
  * A category heading is `N. title` (no colon in the title) or `# N title`. Dialogues are numbered inside their category (1, 2, ...).
  * A dialogue is two consecutive lines: someone else says something, then the character answers (`Name: text`). A leading `12.` number is ignored.
  * Anything else (titles, descriptions, blank lines) is ignored. A category title that contains a colon would be read as a spoken line.
  * The label of a dialogue is the axis of the CHARACTER'S REPLY, as I assigned it. It is not computed here.

Standard library only.
"""
import re
import unicodedata

AXIS_LETTER = {"NORTH": "N", "SOUTH": "S", "EAST": "E", "WEST": "W"}
_AXIS = re.compile(r"^(?:#{1,2}\s*)?(NORTH|SOUTH|EAST|WEST)\b[^:]*$")
_CAT_AUTHOR = re.compile(r"^(\d+)\.\s+([^:]+)$")
_CAT_SIMPLE = re.compile(r"^#\s*(\d+)\b\s*(.*)$")
_LINE = re.compile(r"^(?:\d+\.\s*)?([^:]{1,40}):\s*(.+)$")


def _fold(s):
    """lower case without accents, so 'Lucía' and 'Lucia' are the same speaker"""
    return "".join(c for c in unicodedata.normalize("NFD", s.lower()) if unicodedata.category(c) != "Mn").strip()


def parse_bank(text, character="Lucia"):
    """text -> list of dicts: ax (N/S/E/W), cat (1..), cat_title, n (1.. inside the category), who, prompt, reply."""
    out, ax, cat, title, n, pending = [], None, 0, "", 0, None
    me = _fold(character)
    for no, raw in enumerate(text.splitlines(), 1):
        line = raw.strip()
        if not line:
            pending = None; continue
        m = _AXIS.match(line)
        if m:
            ax, cat, title, n, pending = AXIS_LETTER[m.group(1)], 0, "", 0, None; continue
        m = _CAT_SIMPLE.match(line) or _CAT_AUTHOR.match(line)
        if m and ":" not in line:
            cat, title, n, pending = cat + 1, m.group(2).strip(), 0, None; continue
        m = _LINE.match(line)
        if not m:
            continue
        who, said = m.group(1).strip(), m.group(2).strip()
        if _fold(who) == me:
            if pending is not None:
                if ax is None:
                    raise ValueError(f"line {no}: a dialogue appears before any axis heading (NORTH / SOUTH / EAST / WEST)")
                n += 1
                out.append(dict(ax=ax, cat=cat, cat_title=title, n=n, who=pending[0], prompt=pending[1], reply=said))
            pending = None
        else:
            pending = (who, said)
    return out


def load_bank(path, character="Lucia"):
    with open(path, encoding="utf-8") as f:
        return parse_bank(f.read(), character)


def bank_summary(dialogues):
    """counts per axis and per (axis, category), for a quick sanity check of a bank"""
    ax, cat = {}, {}
    for d in dialogues:
        ax[d["ax"]] = ax.get(d["ax"], 0) + 1
        cat[(d["ax"], d["cat"])] = cat.get((d["ax"], d["cat"]), 0) + 1
    return dict(dialogues=len(dialogues), per_axis=ax, per_category=cat)


def bank_to_script(dialogues, character="Lucia"):
    """the bank as an ordinary script (`Name: text`, scenes separated by `---`), ready for `Store.ingest_script`"""
    return "\n\n---\n".join(f"{d['who']}: {d['prompt']}\n{character}: {d['reply']}" for d in dialogues) + "\n"


_AXIS_NOTE = {"N": "V north: we, future, possibility, connection", "S": "I south: cold, fear, absence, past, hard logic",
              "E": "Lv east: me first, defense of the self, subjective opinion", "W": "Gv west: empathy, calm, toward you"}


def axis_seed_from_bank(dialogues, field="reply", bank_name="lucia_bank.txt"):
    """a seed for `AxisModel.from_seed` / `Store(axis_seed=...)`: the bank's own lines labeled by the bank's own axis.
    field='reply' uses the character's lines; field='prompt' uses what the others say to her."""
    phrases = {a: [d[field] for d in dialogues if d["ax"] == a] for a in "NSEW"}
    meta = dict(source="author_bank", lang="en", field=field,
                note=(f"{sum(len(v) for v in phrases.values())} lines from {bank_name}, labeled by the axis the bank assigns to the character's reply. "
                      "Bank and training by me."),
                axes=_AXIS_NOTE)
    return dict(meta=meta, phrases=phrases)
