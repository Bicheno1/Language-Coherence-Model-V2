"""Generates data/wordnet_baked.json from WordNet.   BUILD-TIME ONLY: it needs `pip install nltk` and the WordNet corpus
(python tools/setup_nltk.py). Run it ONCE (and again only if the tables change). After that the LCM does not need nltk.

It replicates exactly what lcm/narrate.py used to ask WordNet at runtime (the original `_first_sense`, `_frequency`, `_closing`,
`_ancestors`, `_intransitive` and `_is_adverb`), for every word that is a node of the tables.   NOT TESTED in the environment where
it was written (there was no nltk): run it, then run  python tests/test_tables.py  and compare with the original v0.5 results.
"""
import json, os, sys, datetime

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))


def main(out=None):
    try:
        from nltk.corpus import wordnet as wn
        wn.synsets("dog")
        import nltk
    except Exception as e:
        sys.exit("This tool needs nltk and the WordNet corpus: pip install nltk && python tools/setup_nltk.py   (" + type(e).__name__ + ")")
    import lcm
    t = lcm.load()
    words = {}
    for tb in ("subjects", "concepts", "connectors"):
        for n in t[tb].nodes:
            w = n["word"].lower()
            if w not in words or n.get("type") == "biological":
                words[w] = n
    synset, paths, closure, ancestors, frequency, adverbs, intransitive = {}, {}, {}, {}, {}, [], {}
    for w, n in sorted(words.items()):
        ss = wn.synsets(w.replace(" ", "_"), "n")
        sy = ss[0] if ss else None                                           # _first_sense
        if sy is not None:
            synset[w] = sy.name()
        frequency[w] = sum(l.count() for l in sy.lemmas() if l.name().lower() == w) if sy else 0          # _frequency
        dist = {}
        if sy is not None:                                                    # _ancestors
            for path in sy.hypernym_paths():
                for d, anc in enumerate(reversed(path[:-1]), 1):
                    for l in anc.lemmas():
                        a = l.name().lower().replace("_", " ")
                        dist[a] = min(dist.get(a, 99), d)
        ancestors[w] = [a for a, _ in sorted(dist.items(), key=lambda x: (x[1], x[0]))]
        if sy is not None and n.get("type") == "biological":                  # for _descendants (hypernym closure = inverse of _closing)
            paths[w] = [[x.name() for x in p] for p in sy.hypernym_paths()]
            closure[w] = sorted(x.name() for x in sy.closure(lambda x: x.hypernyms()))
        if wn.synsets(w, "r"):                                                # _is_adverb
            adverbs.append(w)
        for s in wn.synsets(w, "v")[:3]:                                      # _intransitive
            if any(l.name().lower() == w and set(l.frame_ids()) & {1, 2} for l in s.lemmas()):
                intransitive[w] = True
                break
    data = dict(meta=dict(source="WordNet via nltk", nltk=nltk.__version__, words=len(words), generated=datetime.date.today().isoformat()),
                synset=synset, paths=paths, closure=closure, ancestors=ancestors, frequency=frequency, adverbs=adverbs, intransitive=intransitive)
    out = out or os.path.join(HERE, "..", "data", "wordnet_baked.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
    print(f"written {os.path.relpath(out)}: {len(words)} words, {len(synset)} with a main sense, {len(paths)} biological with paths, "
          f"{len(adverbs)} adverbs, {len(intransitive)} verbs with an intransitive use")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
