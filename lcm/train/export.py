"""From the training to the LCM.

  related_overlay(store)   -> {word: [[target, count], ...]}  the relations learned from the text, in the format of the field
                              `related_oanc` that `lcm.tables` already reads (channel 'corpus', weight 0.4, off by default).
  apply_overlay(tables, o) -> writes it into the nodes of the tables and switches `USE_CORPUS` on.
  compare_axes(store, t)   -> agreement between the center of each word learned from the text and the dominant axis of the
                              hand-made values (V, I, Lv, Gv) of the tables. It is the check of the training against your tables.
"""
import json
from collections import Counter, defaultdict
from .axes import AX, AXNAME

_TYPED_OR_POSITION = None            # all channels (typed ones or by position) feed the overlay


def related_overlay(store, top=8, min_count=2, channels=None):
    uni, votes, rel = store._merged()
    tables = getattr(store, "tables", None)
    by = defaultdict(Counter)
    for (r, a, b), c in rel.items():
        if channels is not None and not any(r == ch or r.startswith(ch) for ch in channels):
            continue
        if r == "near":
            c = c * 0.5                                          # a word two or three positions away weighs less
        by[a][b] += c
        by[b][a] += c
    out = {}
    for w, cnt in by.items():
        if tables is not None and not tables.find(w):
            continue                                             # only words that are nodes: the rest has nowhere to go
        pairs = [[t, int(round(c))] for t, c in cnt.most_common() if c >= min_count and t != w and (tables is None or tables.find(t))][:top]
        if pairs:
            out[w] = pairs
    return out


def apply_overlay(tables, overlay, enable=True):
    n = 0
    for w, pairs in overlay.items():
        for _, node in tables.find(w):
            node["related_oanc"] = pairs; n += 1
    if enable:
        tables.USE_CORPUS = True
    return n


def save_overlay(overlay, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(overlay, f, ensure_ascii=False)


def load_overlay(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def table_center(node):
    """Dominant axis of the hand-made values of a node (None if there is a tie at the top)."""
    v = {"V": node["V"], "I": node["I"], "Lv": node["Lv"], "Gv": node["Gv"]}
    m = max(v.values()); top = [a for a, x in v.items() if x == m]
    return top[0] if len(top) == 1 else None


def compare_axes(store, tables, min_n=5):
    """How many of the words whose center was learned from the text (n >= min_n occurrences) have the same dominant axis in the tables."""
    t = store.table(min_axis_n=min_n)
    conf = Counter(); rows = []
    for r in t["words"]:
        ax = r["axes"]
        if not ax or not ax["center"]:
            continue
        hits = tables.find(r["word"])
        if not hits:
            continue
        ref = table_center(hits[0][1])
        if ref is None:
            continue
        conf[(ref, ax["center"])] += 1
        rows.append((r["word"], ref, ax["center"], ax["n"]))
    n = len(rows); agree = sum(1 for _, a, b, _n in rows if a == b)
    return dict(compared=n, agree=agree, agreement=(agree / n if n else None), chance=0.25,
                confusion={f"table {a} -> text {b}": c for (a, b), c in sorted(conf.items())},
                disagreements=[r for r in rows if r[1] != r[2]][:20])
