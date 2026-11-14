"""LCM v2: tests of what is NEW (analyzer without spaCy, inflection without lemminflect, data baked from WordNet,
training connected to the LCM). The old suites (test_tables.py, test_arithmetic.py) check that nothing regressed."""
import json, os, sys, tempfile, unittest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import lcm
from lcm.inflect import getInflection, getAllInflections, getLemma, verb_lemma_candidates
from lcm.lite_nlp import analyze
from lcm import wn_baked
from lcm.train import Store
from lcm.train.analyzer import LiteAnalyzer
from lcm.train.character import Character
from lcm.train.export import related_overlay, apply_overlay, compare_axes, table_center

with open(os.path.join(os.path.dirname(__file__), "train_examples", "demo_script.txt"), encoding="utf-8") as f:
    DEMO = f.read()
LUCIA = """Ben: Hello Lucia.
Lucia: Hello companion.
Joan: Hello Lucia, how are you?
Lucia: Hello comrade, I am fine, and you?
Ben: Good morning, Lucia.
Lucia: Hello companion, see, the day has begun.
Lucia walks into the room and puts her coat on the chair.
Joan: Hello Lucia.
Lucia: Hello comrade.
Ben: Hello, Lucia, do you have a minute?
Lucia: See, you always arrive when I am busy.
Joan: Lucia, are you coming to the meeting?
Lucia: I am coming, comrade. I am coming.
Ben: Hello Lucia.
Lucia: Hello companion.
"""
T = lcm.load()


class TestInflect(unittest.TestCase):
    def test_forms(self):
        for lem, tag, want in [("tell", "VBD", "told"), ("tell", "VBZ", "tells"), ("stop", "VBD", "stopped"), ("stop", "VBG", "stopping"), ("try", "VBD", "tried"),
                               ("dance", "VBG", "dancing"), ("go", "VBZ", "goes"), ("sleep", "VBD", "slept"), ("lie", "VBG", "lying"), ("story", "NNS", "stories"),
                               ("woman", "NNS", "women"), ("wolf", "NNS", "wolves"), ("box", "NNS", "boxes"), ("person", "NNS", "people")]:
            self.assertEqual(getInflection(lem, tag)[0], want, (lem, tag))
        self.assertEqual(getAllInflections("tell")["VBD"], ("told",))
        self.assertEqual(getLemma("stories", "NOUN")[0], "story")

    def test_lemma_candidates_include_the_right_one(self):
        for w, lem in [("stopped", "stop"), ("dancing", "dance"), ("running", "run"), ("went", "go"), ("tries", "try")]:
            self.assertIn(lem, [c[0] for c in verb_lemma_candidates(w)])


class TestAnalyzer(unittest.TestCase):
    def test_trees(self):
        d = analyze("Do you like ducks?", T)
        root = next(t for t in d if t.dep_ == "ROOT")
        self.assertEqual((root.lemma_, [c.dep_ for c in root.children if c.dep_ in ("nsubj", "dobj", "aux")]), ("like", ["aux", "nsubj", "dobj"]))
        d = analyze("Why is the sky blue?", T); root = next(t for t in d if t.dep_ == "ROOT")
        self.assertEqual(root.lemma_, "be"); self.assertEqual({c.dep_: c.lower_ for c in root.children if c.dep_ in ("nsubj", "acomp", "advmod")}, {"advmod": "why", "nsubj": "sky", "acomp": "blue"})
        d = analyze("Tell me a sad story about a king.", T); root = next(t for t in d if t.dep_ == "ROOT")
        self.assertEqual(root.tag_, "VB"); self.assertEqual(sorted(c.dep_ for c in root.children if c.dep_ in ("dative", "dobj")), ["dative", "dobj"])

    def test_unknown_words_do_not_break(self):
        for s in ["Zzqx blorf", "zzqx", "", "?", "Tell me about zzqx", "!!!"]:
            analyze(s, T)

    def test_typed_relations(self):
        a = LiteAnalyzer(T)
        self.assertEqual(a.relations("The dog jumped over the fence yesterday.", []), [("S", "dog", "jump"), ("PREP_over", "fence", "jump"), ("ADV", "yesterday", "jump")])
        self.assertIn(("O", "duck", "like"), a.relations("Do you like ducks?", []))
        self.assertIn(("ATTR", "happy", "be"), a.relations("I am happy today.", []))


class TestTrainingConnectedToLcm(unittest.TestCase):
    def setUp(self):
        self.s = Store.for_lcm(); self.s.ingest_script(LUCIA)

    def test_typed_channel_and_known_vocabulary(self):
        self.assertEqual(self.s.table()["meta"]["relation_channel"], "typed")
        self.assertEqual(self.s.lex.info["happy"].status, "confirmed")             # a word of the LCM tables is known from the start

    def test_narration_is_not_glued(self):
        self.assertFalse(any("walks into the room" in p["rt"] or "walks into the room" in p["pt"] for p in self.s.pairs))

    def test_character_repertoire_with_frequencies(self):
        c = Character(self.s, "lucia"); n = 900
        out = [c.respond("Hello Lucia", seed=i).text for i in range(n)]
        share = out.count("Hello companion.") / n
        self.assertTrue(0.58 < share < 0.75, share)                                  # she said it 2 of 3 times
        self.assertEqual(set(out), {"Hello companion.", "Hello comrade."})
        self.assertEqual(c.respond("Hello Lucia", seed=5).text, c.respond("Hello Lucia", seed=5).text)   # same seed, same answer
        self.assertEqual(c.respond("Hello Lucia", temperature=0).text, "Hello companion.")

    def test_character_abstains(self):
        c = Character(self.s, "Lucia")
        r = c.respond("Quantum chromodynamics of bread", seed=1)
        self.assertEqual((r.kind, r.text), ("abstain", None))

    def test_overlay_goes_into_the_lcm(self):
        ov = related_overlay(self.s, min_count=1)
        self.assertTrue(ov and all(T.find(w) for w in ov))                             # only words that are nodes
        t = lcm.load()
        self.assertFalse(t.USE_CORPUS)
        with tempfile.TemporaryDirectory() as d:
            from lcm.train.export import save_overlay
            p = os.path.join(d, "o.json"); save_overlay(ov, p); t2 = lcm.load(overlay=p)
        self.assertTrue(t2.USE_CORPUS)
        w = next(iter(ov)); self.assertEqual(t2.find(w)[0][1]["related_oanc"], ov[w])

    def test_compare_axes(self):
        s = Store.for_lcm(); s.ingest_script(DEMO)
        r = compare_axes(s, T, min_n=2)
        self.assertGreater(r["compared"], 5); self.assertEqual(r["chance"], 0.25)
        self.assertEqual(table_center(T.find("happy")[0][1]), "Gv")                    # V7 I1 Lv2 Gv8 -> Gv is the highest


class TestBakedWordnet(unittest.TestCase):
    """The plumbing of the data baked from WordNet, with a toy tree (it does NOT check that bake_wordnet.py reproduces WordNet)."""
    TREE = {"meta": {"source": "toy"}, "synset": {"animal": "animal.n.01", "dog": "dog.n.01", "cat": "cat.n.01", "puppy": "puppy.n.01", "rock": "rock.n.01"},
            "closure": {"dog": ["animal.n.01"], "cat": ["animal.n.01"], "puppy": ["dog.n.01", "animal.n.01"]},
            "paths": {"dog": [["entity.n.01", "animal.n.01", "dog.n.01"]], "cat": [["entity.n.01", "animal.n.01", "cat.n.01"]],
                      "puppy": [["entity.n.01", "animal.n.01", "dog.n.01", "puppy.n.01"]]},
            "ancestors": {"dog": ["animal", "organism"]}, "frequency": {"dog": 42}, "adverbs": ["slowly"], "intransitive": {"climb": True}}

    def setUp(self):
        self.d = tempfile.TemporaryDirectory(); self.p = os.path.join(self.d.name, "w.json")
        json.dump(self.TREE, open(self.p, "w")); self.old = wn_baked.PATH
        wn_baked.PATH = self.p; wn_baked._load.cache_clear()

    def tearDown(self):
        wn_baked.PATH = self.old; wn_baked._load.cache_clear(); self.d.cleanup()

    def test_api(self):
        self.assertTrue(wn_baked.available()); self.assertEqual(wn_baked.main_synset("dog"), "dog.n.01"); self.assertEqual(wn_baked.frequency("dog"), 42)
        self.assertEqual(wn_baked.frequency("zzz"), 0); self.assertTrue(wn_baked.is_adverb("slowly")); self.assertFalse(wn_baked.is_adverb("dog"))
        self.assertTrue(wn_baked.intransitive("climb")); self.assertFalse(wn_baked.intransitive("eat")); self.assertEqual(wn_baked.ancestors("dog"), ("animal", "organism"))
        self.assertIn("animal.n.01", wn_baked.closure("puppy"))

    def test_descendants_use_the_baked_tree(self):
        from lcm import narrate
        class Tb:
            def __init__(self, nodes): self.nodes = nodes
        node = {"word": "animal", "type": "biological", "category": "animal"}
        nodes = [node] + [{"word": w, "type": "biological", "category": "animal"} for w in ("dog", "cat", "puppy")] + [{"word": "rock", "type": "non_biological", "category": "thing"}]
        tables = {"subjects": Tb(nodes), "concepts": Tb([])}
        got = sorted(n["word"] for n in narrate._descendants(tables, node, leaves_only=False))
        self.assertEqual(got, ["cat", "dog", "puppy"])                                  # the rock and the node itself are out

    def test_missing_file_says_what_to_do(self):
        wn_baked.PATH = os.path.join(self.d.name, "nope.json"); wn_baked._load.cache_clear()
        self.assertFalse(wn_baked.available())
        with self.assertRaises(wn_baked.MissingBakedData) as cm: wn_baked.is_adverb("x")
        self.assertIn("bake_wordnet.py", str(cm.exception))


class TestNoExternalDependencies(unittest.TestCase):
    def test_runtime_code_imports_nothing_external(self):
        import re, glob
        bad = []
        for f in glob.glob(os.path.join(os.path.dirname(__file__), "..", "lcm", "**", "*.py"), recursive=True):
            with open(f, encoding="utf-8") as fh:
                for i, l in enumerate(fh, 1):
                    if re.match(r"\s*(import|from)\s+(spacy|nltk|lemminflect|numpy)\b", l):
                        bad.append((os.path.relpath(f), i, l.strip()))
        self.assertEqual(bad, [])


if __name__ == "__main__":
    unittest.main()
