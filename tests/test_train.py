import unittest, os, sys, tempfile, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from lcm.train import parse_script, make_pairs, Lexicon, Store
from lcm.train.lexicon import is_inflection
from lcm.train.axes import AxisModel
from lcm.train.store import SEED
from lcm.train.reply import Retriever
from lcm.train.charset import Charset
from lcm.train.script import detect_format, parse_chat, parse_dash, NotDialogueError
from lcm.train.lexicon import tokenize
from lcm.train.store import seed_for

def EX(n):
    with open(os.path.join(os.path.dirname(__file__), "train_examples", n), encoding="utf-8") as f:
        return f.read()
DEMO, DEMO_ES = EX("demo_script.txt"), EX("demo_script_es.txt")


class TestScript(unittest.TestCase):
    def test_parse_and_pairs(self):
        t = parse_script("Lucía: Hola, estas bien. (sonríe)\nBen: Sí,\nclaro que sí\n\n\nLucía: Otra escena\nBen: Ok\nBen: Y mas")
        self.assertEqual([x.speaker for x in t], ["Lucía", "Ben", "Lucía", "Ben", "Ben"])
        self.assertEqual(t[0].text, "Hola, estas bien.")
        self.assertEqual(t[1].text, "Sí, claro que sí")
        self.assertEqual([x.scene for x in t], [0, 0, 1, 1, 1])
        pairs = make_pairs(t)
        self.assertEqual(len(pairs), 2)           # (Lucía->Ben) in scene 0, (Lucía->Ben) in scene 1; Ben->Ben is not a pair
        self.assertEqual(pairs[0].reply.speaker, "Ben")


class TestLexicon(unittest.TestCase):
    def test_inflection(self):
        for a, b in [("file", "files"), ("use", "used"), ("stop", "stopped"), ("make", "making"), ("city", "cities")]:
            self.assertTrue(is_inflection(a, b), (a, b))
        for a, b in [("form", "from"), ("cat", "cut"), ("thank", "think")]:
            self.assertFalse(is_inflection(a, b), (a, b))

    def test_typo_alias_after_confirmation(self):
        lx = Lexicon()
        for _ in range(3): lx.observe("thank", ["you", "for"])
        self.assertEqual(lx.info["thank"].status, "confirmed")
        self.assertEqual(lx.observe("thnak", ["you", "for"]), "thank")
        self.assertEqual(lx.typos[("thnak", "thank")], 1)

    def test_inflection_not_merged(self):
        lx = Lexicon()
        for _ in range(3): lx.observe("file", ["open", "the"])
        self.assertEqual(lx.observe("files", ["open", "the"]), "files")

    def test_context_blocks_wrong_merge(self):
        lx = Lexicon()
        for _ in range(3): lx.observe("form", ["filled", "out", "paper"])
        self.assertEqual(lx.observe("from", ["came", "home", "tomorrow"]), "from")

    def test_unknown_stays_provisional(self):
        lx = Lexicon(); lx.observe("zorblax", ["the", "big"])
        self.assertEqual(lx.info["zorblax"].status, "provisional")
        self.assertIn("zorblax", lx.ids())


class TestAxes(unittest.TestCase):
    def test_seed_loads_and_predicts(self):
        m = AxisModel.from_seed(SEED)
        for text, ax in [("We will make it through together.", "N"), ("The world is a cold place.", "S"), ("I think I am right about this.", "E"), ("I hope you have a wonderful day.", "W")]:
            self.assertEqual(m.predict(text)[0], ax, text)


class TestStore(unittest.TestCase):
    def test_incremental_equals_batch(self):
        a, b = Store(), Store()
        for part in DEMO.split("\n---\n"): a.ingest_script(part)
        b.ingest_script(DEMO)
        self.assertEqual(a.uni, b.uni); self.assertEqual(a.rel, b.rel); self.assertEqual(a.utt_axis, b.utt_axis)
        self.assertEqual(dict(a.votes), dict(b.votes))

    def test_save_load_roundtrip(self):
        s = Store(); s.ingest_script(DEMO)
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "db.json"); s.save(p); t = Store.load(p)
        self.assertEqual(json.dumps(s.table(), sort_keys=True), json.dumps(t.table(), sort_keys=True))

    def test_typo_merges_in_demo(self):
        s = Store(); s.ingest_script(DEMO)
        self.assertIn(("thnak", "thank"), set(s.lex.typos))                     # shared context: it merges
        self.assertEqual(s.lex.resolve("tomorow"), "tomorow")                   # no common context: it stays provisional (safe)
        self.assertEqual(s.lex.info["tomorow"].status, "provisional")

    def test_fix_alias_is_auditable(self):
        s = Store(); s.ingest_script(DEMO)
        tot = lambda t: sum(r["count"] for r in t["words"] if r["word"] in ("hello", "help", "helo"))
        before = tot(s.table())
        s.fix_alias("helo", "hello"); t = s.table()
        self.assertEqual(tot(t), before)                                         # no data is lost or invented
        self.assertNotIn("helo", [r["word"] for r in t["words"]])
        self.assertIn(("helo", "hello"), {(m["variant"], m["word"]) for m in t["typo_merges"]})
        s.fix_alias("helo", None)
        self.assertIn("helo", [r["word"] for r in s.table()["words"]])

    def test_speaker_profile_case_insensitive(self):
        s = Store(); s.ingest_script(DEMO)
        self.assertEqual(s.speaker_profile("lucia")["speaker"], "Lucia")
        with self.assertRaises(KeyError): s.speaker_profile("nadie")

    def test_demo_characters_get_their_axis(self):
        s = Store(); s.ingest_script(DEMO); want = {"Lucia": "Gv", "Ben": "Lv", "Mara": "I", "Ava": "V"}; ok = 0
        for n, ax in want.items():
            mix = s.speaker_profile(n)["axis_mix"]; ok += max(mix, key=mix.get) == ax
        self.assertGreaterEqual(ok, 3, "at least 3 of 4 characters must have their dominant axis")

    def test_reply_baseline_uses_own_lines(self):
        s = Store(); s.ingest_script(DEMO); r = Retriever(s).reply("Ben", "Are you okay today?", own_only=True)
        self.assertTrue(r and all(x[2] == "Ben" for x in r))


class TestCharset(unittest.TestCase):
    def test_numbers(self):
        c = Charset()
        self.assertEqual(c.encode("año"), (1, 27, 15)); self.assertEqual(c.encode("Ñ"), (27,))
        self.assertEqual(c.encode("sí"), (19, 30)); self.assertNotEqual(c.encode("si"), c.encode("sí"))
        self.assertEqual((c.num("¿"), c.num("."), c.num(",")), (59, 51, 52)); self.assertEqual(c.decode(c.encode("niño, sí.")), "niño, sí.")

    def test_decomposed_text_is_normalized(self):
        self.assertEqual(tokenize("an\u0303o"), ["año"])

    def test_dynamic_chars_persist(self):
        c = Charset(); n = c.num("€"); self.assertGreaterEqual(n, 200); self.assertEqual(Charset(c.dynamic()).num("€"), n)


class TestFormat(unittest.TestCase):
    def test_detect_each_format(self):
        for f, want in [("demo_script.txt", "named"), ("demo_script_es.txt", "named"), ("dash_dialogue_es.txt", "dash"), ("whatsapp_chat_es.txt", "chat"), ("plain_text_es.txt", "plain")]:
            self.assertEqual(detect_format(EX(f))["format"], want, f)

    def test_plain_rejected_but_monologue_ok(self):
        s = Store()
        with self.assertRaises(NotDialogueError): s.ingest_script(EX("plain_text_es.txt"))
        r = s.ingest_script(EX("plain_text_es.txt"), monologue="Narradora")
        self.assertEqual((r["turns"], r["pairs"]), (2, 0))

    def test_single_label_is_not_a_speaker(self):
        self.assertEqual(detect_format("Nota: esto es un texto\nOtra linea de prosa sin nombre\nMas prosa por aqui")["format"], "plain")

    def test_chat_and_dash_parse(self):
        t = parse_chat(EX("whatsapp_chat_es.txt"))
        self.assertEqual([x.speaker for x in t], ["Lucía", "Beto", "Lucía", "Beto"]); self.assertIn("y no sabía qué decirte", t[2].text)
        d = parse_dash(EX("dash_dialogue_es.txt")); self.assertEqual([x.speaker for x in d], ["A", "B", "A", "B", "A"])


class TestSpanish(unittest.TestCase):
    def test_accent_and_gender_are_not_merged(self):
        lx = Lexicon(lang="es")
        for _ in range(3): lx.observe("esta", ["dijo", "que"]); lx.observe("niño", ["el", "pequeño"])
        self.assertEqual(lx.observe("está", ["dijo", "que"]), "está")
        self.assertEqual(lx.observe("niña", ["el", "pequeño"]), "niña")

    def test_missing_enye_is_a_typo(self):
        lx = Lexicon(lang="es")
        for _ in range(3): lx.observe("años", ["hace", "muchos"])
        self.assertEqual(lx.observe("anos", ["hace", "muchos"]), "años")

    def test_axes_seed_es(self):
        m = AxisModel.from_seed(seed_for("es"))
        for text, ax in [("Juntos, todo es posible.", "N"), ("El mundo es un lugar frío.", "S"), ("En mi opinión, estás exagerando.", "E"), ("Gracias, eres muy amable.", "W")]:
            self.assertEqual(m.predict(text)[0], ax, text)

    def test_spanish_store(self):
        s = Store(); r = s.ingest_script(DEMO_ES); self.assertEqual(r["lang"], "es")
        row = next(x for x in s.table()["words"] if x["word"] == "mañana"); self.assertEqual(row["cipher"], [13, 1, 27, 1, 14, 1])
        want = {"Lucía": "Gv", "Beto": "Lv", "Mara": "I", "Ana": "V"}; ok = 0
        for n, ax in want.items():
            mix = s.speaker_profile(n)["axis_mix"]; ok += max(mix, key=mix.get) == ax
        self.assertGreaterEqual(ok, 3)
        with self.assertRaises(ValueError): s.ingest_script(DEMO)          # one base = one language
    def test_spanish_roundtrip_keeps_chars(self):
        s = Store(); s.ingest_script(DEMO_ES); s.lex.cs.num("€")
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "es.json"); s.save(p); t = Store.load(p)
        self.assertEqual(t.lang, "es"); self.assertEqual(t.lex.cs.num("€"), s.lex.cs.num("€")); self.assertEqual(t.lex.lang, "es")


if __name__ == "__main__":
    unittest.main()
