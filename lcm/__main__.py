"""Usage (v2):  python -m lcm train ingest|detect|profile|word|reply|tables|alias ...  |  lucia --say \"TEXT\"|--demo|--chat|--info|--make-seed OUT  |  character NAME \"SENTENCE\" [--db F] [--seed N] [--temperature T] [--tau X]  |  overlay [--db F] [--out F]  |  compare [--db F] [--min-n N]  |  app
       Also:  python -m lcm stats | lookup WORD | cell TABLE ROW COL | parse \"SENTENCE\" | walk \"SENTENCE\" [STEPS] | path A B | decompose \"SENTENCE\" | recompose \"SENTENCE\" [--trace] [--seed N] [--randomize] | story \"SENTENCE\" [SENTENCES] [--trace] [--uppercase] [--randomize] [--seed N] [--temperature T] [--top-k K] | reply \"SENTENCE\" [--trace] [--uppercase] [--seed N] [--temperature T] [--disposition friendly|neutral]"""
import json
import sys
from . import load


def _fmt(table, n):
    ej = f"{n['word']}  [{table}]"
    if "group" in n:   # connector
        meta = f"{n['group']} / {n['relation']} / {n['class']}"
    else:
        meta = f"{n['type']} / {n['category']}" + (f" / {n['subtype']}" if n.get("subtype") else "")
    return (f"{ej}\n    {meta}\n    V{n['V']} I{n['I']} Lv{n['Lv']} Gv{n['Gv']}  ->  x={n['x']:+.1f} y={n['y']:+.1f}  "
            f"cell row {n['row']} col {n['col']}  (tension {n['tension']})")


def _opts(argv, names):
    opts, rest, i = {}, [], 0
    while i < len(argv):
        if argv[i] in names and i + 1 < len(argv):
            opts[argv[i][2:]] = argv[i + 1]; i += 2
        else:
            rest.append(argv[i]); i += 1
    return opts, rest


def main(argv):
    if argv and argv[0] == "train":                                   # the training module (does not need the tables)
        from .train.cli import main as train_main
        return train_main(argv[1:])
    if argv and argv[0] == "lucia":                                   # a character answering from her emotional bank (does not need the tables)
        from .train.bankbot import main as bank_main
        return bank_main(argv[1:])
    t = load()
    if argv and argv[0] == "character":
        opts, rest = _opts(argv[1:], ("--db", "--seed", "--temperature", "--tau"))
        if len(rest) < 2:
            print("usage: character NAME \"SENTENCE\" [--db forge_db.json] [--seed N] [--temperature T] [--tau X]"); return 2
        from .train import Store
        from .train.character import Character
        c = Character(Store.load(opts.get("db", "forge_db.json")), rest[0], tau=float(opts.get("tau", 0.25)))
        r = c.respond(rest[1], seed=int(opts["seed"]) if "seed" in opts else None, temperature=float(opts.get("temperature", 1.0)))
        print(r.text if r.text is not None else "(I do not know: nothing in her repertoire looks like that)")
        print(f"  [{r.kind}, similarity {r.confidence}]" + "".join(f"\n  {p:.2f}  {tx}" for tx, p, _ in r.alternatives))
        return 0
    if argv and argv[0] == "overlay":
        opts, _ = _opts(argv[1:], ("--db", "--out", "--min-count"))
        from .train import Store
        from .train.export import related_overlay, save_overlay
        s = Store.load(opts.get("db", "forge_db.json")); s.tables = t
        ov = related_overlay(s, min_count=int(opts.get("min-count", 2))); save_overlay(ov, opts.get("out", "overlay_related.json"))
        print(f"{len(ov)} words with relations learned from text -> {opts.get('out', 'overlay_related.json')}  (use: lcm.load(overlay=...))")
        return 0
    if argv and argv[0] == "compare":
        opts, _ = _opts(argv[1:], ("--db", "--min-n"))
        from .train import Store
        from .train.export import compare_axes
        r = compare_axes(Store.load(opts.get("db", "forge_db.json")), t, min_n=int(opts.get("min-n", 5)))
        print(json.dumps(r, ensure_ascii=False, indent=1)); return 0
    if not argv or argv[0] == "stats":
        for name, s in t.stats().items():
            print(f"{name:10s} {s['nodes']:5d} nodes | {s['occupied_cells']:3d}/100 cells occupied | largest: row {s['largest_cell']['row']} col {s['largest_cell']['col']} ({s['largest_cell']['nodes']})")
    elif argv[0] == "lookup" and len(argv) > 1:
        hits = t.find(argv[1])
        if not hits:
            print("not found"); return 1
        for table, n in hits:
            print(_fmt(table, n))
            vec = [w["word"] for w in t[table].near(n["word"], 1)][:8]
            print("    neighbors (±1 cell):", ", ".join(vec) or "—")
            print("    synonyms:", ", ".join(n.get("synonyms", [])[:6]) or "—")
            links = t.links(n["word"])
            print("    related:", ", ".join(f"{x['word']} ({via})" for x, _, via, _ in links[:10]) or "—",
                  f"(+{len(links) - 10} more)" if len(links) > 10 else "",
                  ("| no node: " + ", ".join(n["related_labels"])) if n.get("related_labels") else "")
            opp = [x["word"] for _, x in t.opposites(n["word"])]
            print("    opposites:", ", ".join(opp) or "— (no antonym; geometric mirror: lcm mirror)")
    elif argv[0] == "parse" and len(argv) > 1:
        from .parse import parse
        print(json.dumps(parse(argv[1], t).to_dict(), ensure_ascii=False, indent=1, default=list))
    elif argv[0] == "walk" and len(argv) > 1:
        from .parse import parse
        from .walk import walk
        f = parse(argv[1], t)
        r = walk(t, f, **({"max_steps": int(argv[2])} if len(argv) > 2 else {}))
        print(f"{f.mode} · tense {f.tense} ({f.tense_reason}) · quantity {f.quantity}\ntarget x={r.target[0]:+.2f} y={r.target[1]:+.2f} · coherence {r.coherence} · stop: {r.stop}")
        for p in r.steps:
            print(f"  {p.word:14s} [{p.table[:4]}] cell {p.cell}  dist {p.distance:4.2f}  " + (f"{p.via} <- {p.from_word}" if p.from_word else "anchor"))
    elif argv[0] == "story" and len(argv) > 1:
        from .parse import parse
        from .narrate import stories
        flags, pos, opts, i = set(), [], {}, 1
        while i < len(argv):
            a = argv[i]
            if a in ("--seed", "--temperature", "--top-k") and i + 1 < len(argv):
                opts[a[2:]] = argv[i + 1]
                i += 2
                continue
            (flags if a.startswith("--") else pos).add(a) if a.startswith("--") else pos.append(a)
            i += 1
        kw = {"randomize": "--randomize" in flags}
        if "temperature" in opts:
            kw["temperature"] = float(opts["temperature"])
        if "top-k" in opts:
            kw["top_k"] = int(opts["top-k"])
        f = parse(pos[0], t)
        for r in stories(t, f, int(pos[1]) if len(pos) > 1 else 4, seed=int(opts["seed"]) if "seed" in opts else None, **kw):
            if r.error:
                print("no story:", r.error)
                continue
            extra = f" · seed {r.seed}" if r.seed is not None else ""
            print((r.text.upper() if "--uppercase" in flags else r.text) + f"    [coherence {r.coherence}{extra}]")
            if "--trace" in flags:
                for o in r.sentences:
                    print("   " + o.text)
                    for z in o.trace:
                        print("      · " + z)
    elif argv[0] == "reply" and len(argv) > 1:
        from .parse import parse
        from .reply import reply
        flags, pos, opts, i = set(), [], {}, 1
        while i < len(argv):
            a = argv[i]
            if a in ("--seed", "--temperature", "--top-k", "--disposition") and i + 1 < len(argv):
                opts[a[2:]] = argv[i + 1]
                i += 2
                continue
            flags.add(a) if a.startswith("--") else pos.append(a)
            i += 1
        kw = {}
        if "seed" in opts:
            kw["seed"] = int(opts["seed"])
        if "temperature" in opts:
            kw["temperature"] = float(opts["temperature"])
        if "top-k" in opts:
            kw["top_k"] = int(opts["top-k"])
        if "disposition" in opts:
            kw["disposition"] = opts["disposition"]
        for r in reply(t, parse(pos[0], t), **kw):
            if r.error:
                print("no answer:", r.error)
                continue
            extra = f" · seed {r.seed}" if r.seed is not None else ""
            print((r.text.upper() if "--uppercase" in flags else r.text) + f"    [{r.kind} · coherence {r.coherence}{extra}]")
            if "--trace" in flags:
                for z in r.trace:
                    print("      · " + z)
    elif argv[0] == "decompose" and len(argv) > 1:
        from .parse import parse
        from .decompose import decompose
        f = parse(argv[1], t)
        print(f"input: {f.input_text!r} · mode {f.mode} · tense {f.tense} ({f.tense_reason}) · form {f.form}")
        for fi in decompose(t, f):
            x, y = (fi.node["x"], fi.node["y"]) if fi.node else (0, 0)
            print(f"\n  {fi.role_obj.upper():12s} {fi.word!r}" + (f"  cell {fi.node['row']},{fi.node['col']}  x={x:+.1f} y={y:+.1f}  axis {fi.axis}  (height marks: {fi.tense_by_height})" if fi.node else ""))
            if fi.tense_requested:
                print(f"      the sentence asks for it in: {fi.tense_requested}")
            if fi.perspective:
                print(f"      perspective: the LCM reads it as '{fi.perspective}' ({fi.perspective_side})")
            if fi.forms:
                print(f"      forms: {fi.forms}  ·  inflected forms that are nodes: {fi.inflected}")
            for via, ws in fi.related.items():
                print(f"      related ({via}): {', '.join(ws)}")
            if fi.synonyms:
                print(f"      synonyms: {', '.join(fi.synonyms)}")
            print(f"      opposite: {', '.join(fi.opposites) or "— (no lexical opposite)"}")
            for n in fi.notes:
                print(f"      · {n}")
    elif argv[0] == "recompose" and len(argv) > 1:
        from .parse import parse
        from .recompose import recompose
        flags, pos, opts, i = set(), [], {}, 1
        while i < len(argv):
            a = argv[i]
            if a in ("--seed", "--temperature") and i + 1 < len(argv):
                opts[a[2:]] = argv[i + 1]
                i += 2
                continue
            flags.add(a) if a.startswith("--") else pos.append(a)
            i += 1
        kw = {"randomize": "--randomize" in flags}
        if "seed" in opts:
            kw["seed"] = int(opts["seed"])
        if "temperature" in opts:
            kw["temperature"] = float(opts["temperature"])
        r = recompose(t, parse(pos[0], t), **kw)
        if r.error:
            print("no recomposition:", r.error)
        else:
            print((r.text.upper() if "--uppercase" in flags else r.text) + f"    [coherence {r.coherence}" + (f" · seed {r.seed}" if r.seed is not None else "") + "]")
            print(f"   tense {r.tense} · {r.closed_by}")
            if "--trace" in flags:
                for c in r.clauses:
                    print("   " + " ".join(p.text for p in c))
                    for p in c:
                        print(f"      {p.text:<12s} <- {p.rule}")
    elif argv[0] == "path" and len(argv) == 3:
        from .walk import path
        p = path(t, argv[1], argv[2])
        print(" -> ".join(f"{w}" + (f" ({v})" if v != "origin" else "") for w, v in p) if p else "no path")
    elif argv[0] == "cell" and len(argv) == 4:
        nodes = t[argv[1]].cell(int(argv[2]), int(argv[3]))
        print(f"{len(nodes)} nodes:", ", ".join(n["word"] for n in nodes[:40]), "…" if len(nodes) > 40 else "")
    else:
        print(__doc__); return 2
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
