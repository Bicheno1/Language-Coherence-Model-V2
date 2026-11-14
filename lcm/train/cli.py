import argparse, json, os, sys
from .store import Store
from .reply import Retriever
from .script import detect_format, NotDialogueError


def main(argv=None):
    ap = argparse.ArgumentParser(prog="lcm train", description="Builds the LCM databases from conversations (English or Spanish).")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def db(p):
        p.add_argument("--db", default="forge_db.json")

    p = sub.add_parser("detect", help="say whether a file looks like a conversation, and its format and language"); p.add_argument("script")
    p = sub.add_parser("ingest", help="read a conversation and add it to the database"); p.add_argument("script"); db(p)
    p.add_argument("--no-text", action="store_true", help="do not store the text of the pairs (counts only)")
    p.add_argument("--format", default="auto", choices=["auto", "named", "chat", "dash"]); p.add_argument("--monologue", metavar="NAME", help="text by a single person")
    p.add_argument("--lang", choices=["en", "es"], help="force the language (detected by default)")
    p = sub.add_parser("profile", help="a character in numbers"); p.add_argument("speaker"); db(p)
    p = sub.add_parser("word", help="cipher, axis values and relations of a word"); p.add_argument("word"); db(p)
    p = sub.add_parser("reply", help="baseline: what that character answered to something similar"); p.add_argument("speaker"); p.add_argument("text"); db(p); p.add_argument("-k", type=int, default=3)
    p = sub.add_parser("tables", help="export the tables for the LCM"); db(p); p.add_argument("--out", default="LCM_forge_tables.json"); p.add_argument("--min-count", type=int, default=1)
    p = sub.add_parser("alias", help="fix a variant merge: alias helo hello  (or 'none' to undo it)"); p.add_argument("variant"); p.add_argument("canonical"); db(p)
    a = ap.parse_args(argv)
    if a.cmd == "detect":
        from .lang import detect_lang
        t = open(a.script, encoding="utf-8").read(); print(json.dumps(dict(detect_format(t), lang=detect_lang(t)), ensure_ascii=False, indent=2)); return
    if a.cmd == "ingest":
        s = Store.load(a.db) if os.path.exists(a.db) else Store(keep_text=not a.no_text)
        try: r = s.ingest_script(open(a.script, encoding="utf-8").read(), source=a.script, fmt=a.format, monologue=a.monologue, lang=a.lang)
        except (NotDialogueError, ValueError) as e: print("ERROR:", e); sys.exit(2)
        s.save(a.db); print(json.dumps(r, ensure_ascii=False)); return
    s = Store.load(a.db)
    if a.cmd == "alias":
        s.fix_alias(a.variant.lower(), None if a.canonical.lower() == "none" else a.canonical.lower()); s.save(a.db); print("ok")
    elif a.cmd == "profile": print(json.dumps(s.speaker_profile(a.speaker), ensure_ascii=False, indent=2))
    elif a.cmd == "word":
        t = s.table(); w = s.lex.resolve(a.word.lower()); row = next((x for x in t["words"] if x["word"] == w), None)
        print(json.dumps(row, ensure_ascii=False, indent=2)); print("neighbors:", s.neighbors(w))
    elif a.cmd == "reply":
        for sc, rt, rs, pt in Retriever(s).reply(a.speaker, a.text, a.k): print(f"{sc:.2f}  [{rs}] {rt}   (answering: {pt!r})")
    elif a.cmd == "tables":
        json.dump(s.table(min_count=a.min_count), open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1); print("written", a.out)
