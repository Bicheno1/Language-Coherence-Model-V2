"""Tests for the emotional bank, the BankBot and the seeds.  python -m unittest tests.test_bankbot"""
import contextlib
import io
import json
import os
import sys
import unittest

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
from lcm.train.bank import parse_bank, load_bank, bank_summary, axis_seed_from_bank, bank_to_script
from lcm.train.bankbot import BankBot, DEFAULT_BANK
from lcm.train.axes import AxisModel

AUTHOR_LAYOUT = """Lucia — Emotional dialogue bank
NORTH (V) — We
100 dialogues · Connection, future and hope
This axis does not mean constant happiness.
1. Fear of losing someone
1. Bernard: I'm leaving tomorrow, Lucia.
Lucia: But... we just started spending time together.

2. Mary: What if we never see each other again?
Lucia: Don't say that! We'll find a way.
2. Desperation to save a relationship
11. Bernard: Just let me go!
Lucia: I CAN'T! Not without knowing we tried everything!
EAST (Lv) — I / Me
1. Anger and irritation
1. Bernard: You're overreacting.
Lucia: DON'T TELL ME HOW TO FEEL!
"""
SIMPLE_LAYOUT = """## NORTH
# 1 Fear of losing someone
Bernard: I'm leaving tomorrow, Lucia.
Lucia: But... we just started spending time together.

Mary: What if we never see each other again?
Lucia: Don't say that! We'll find a way.
# 2 Desperation to save a relationship
Bernard: Just let me go!
Lucia: I CAN'T! Not without knowing we tried everything!
## EAST
# 1 Anger and irritation
Bernard: You're overreacting.
Lucia: DON'T TELL ME HOW TO FEEL!
"""
KEY = lambda d: (d["ax"], d["cat"], d["n"], d["who"], d["prompt"], d["reply"])


class TestParser(unittest.TestCase):
    def test_both_layouts_give_the_same_dialogues(self):
        a, b = parse_bank(AUTHOR_LAYOUT), parse_bank(SIMPLE_LAYOUT)
        self.assertEqual([KEY(x) for x in a], [KEY(x) for x in b])
        self.assertEqual(len(a), 4)
        self.assertEqual([(d["ax"], d["cat"], d["n"]) for d in a], [("N", 1, 1), ("N", 1, 2), ("N", 2, 1), ("E", 1, 1)])

    def test_title_of_the_category_is_kept_and_descriptions_are_ignored(self):
        a = parse_bank(AUTHOR_LAYOUT)
        self.assertEqual(a[0]["cat_title"], "Fear of losing someone")
        self.assertNotIn("constant happiness", json.dumps(a, ensure_ascii=False))

    def test_character_name_ignores_accents_and_case(self):
        self.assertEqual(len(parse_bank(AUTHOR_LAYOUT, "Lucía")), 4)
        self.assertEqual(len(parse_bank(AUTHOR_LAYOUT, "LUCIA")), 4)
        self.assertEqual(len(parse_bank(AUTHOR_LAYOUT, "Someone else")), 0)

    def test_a_dialogue_before_any_axis_heading_is_an_error(self):
        with self.assertRaises(ValueError):
            parse_bank("Bernard: Hi.\nLucia: Hello.\n")

    def test_script_form_is_an_ordinary_script(self):
        s = bank_to_script(parse_bank(SIMPLE_LAYOUT))
        self.assertIn("Bernard: I'm leaving tomorrow, Lucia.\nLucia: But...", s)
        self.assertEqual(s.count("---"), 3)


class TestShippedBank(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = load_bank(DEFAULT_BANK)
        cls.bot = BankBot(cls.data, "Lucia")
        cls.replies = {d["reply"] for d in cls.data}

    def test_shape_is_4_axes_x_10_categories_x_10_dialogues(self):
        s = bank_summary(self.data)
        self.assertEqual(s["dialogues"], 400)
        self.assertEqual(s["per_axis"], {"N": 100, "S": 100, "E": 100, "W": 100})
        self.assertEqual(sorted(set(s["per_category"].values())), [10])
        self.assertEqual(len(s["per_category"]), 40)

    def test_she_recalls_her_own_reply_for_at_least_95_percent_of_her_bank_prompts(self):
        ok = sum(self.bot.reply(d["prompt"]).text == d["reply"] for d in self.data)
        self.assertGreaterEqual(ok / len(self.data), 0.95, f"{ok}/400")

    def test_she_only_ever_says_sentences_from_her_bank(self):
        probes = ["Could you ever forgive me, Lucia?", "I failed my exam and I feel like a complete idiot.", "What is the capital of France?",
                  "Nobody ever answers my calls any more.", "How dare you read my messages!", "Hello", "", "?!"]
        for t in probes:
            a = self.bot.reply(t)
            self.assertTrue(a.text is None or a.text in self.replies, (t, a))

    def test_unrelated_requests_get_no_reply(self):
        for t in ("What is the capital of France?", "Write a python function to sort a list", "Tell me the price of gold"):
            self.assertEqual(self.bot.reply(t).mode, "silent", t)

    def test_the_three_modes_are_reachable(self):
        self.assertEqual(self.bot.reply("Could you ever forgive me, Lucia?").mode, "recall")
        a = self.bot.reply("I failed my exam and I feel like a complete idiot.")
        self.assertEqual((a.mode, a.axis), ("axis", "W"))
        self.assertEqual(self.bot.reply("What is the capital of France?").mode, "silent")

    def test_deterministic(self):
        t = "Lucia, I am terrified."
        self.assertEqual(self.bot.reply(t), BankBot(self.data, "Lucia").reply(t))

    def test_empty_bank_is_refused(self):
        with self.assertRaises(ValueError):
            BankBot([], "Lucia")


class TestSeeds(unittest.TestCase):
    D = os.path.join(ROOT, "lcm", "train", "data")

    def test_original_seeds_say_they_are_first_version(self):
        for n in ("axis_seed.json", "axis_seed_es.json"):
            with open(os.path.join(self.D, n), encoding="utf-8") as f:
                self.assertEqual(json.load(f)["meta"].get("source"), "first_version", n)

    def test_lucia_seed_is_the_bank_and_loads(self):
        path = os.path.join(self.D, "axis_seed_lucia.json")
        with open(path, encoding="utf-8") as f:
            shipped = json.load(f)
        again = axis_seed_from_bank(load_bank(DEFAULT_BANK), "reply", "lucia_bank.txt")
        self.assertEqual(shipped["phrases"], again["phrases"])
        self.assertEqual(shipped["meta"]["source"], "author_bank")
        self.assertIn("E", AxisModel.from_seed(path).predict("DON'T TELL ME HOW TO FEEL!")[0])


class TestCommandLine(unittest.TestCase):
    def run_cli(self, *args):
        import lcm.__main__ as M
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = M.main(["lucia", *args])
        return rc, buf.getvalue()

    def test_say(self):
        rc, out = self.run_cli("--say", "Could you ever forgive me, Lucia?")
        self.assertEqual(rc, 0)
        self.assertIn("I can try. But I need some time.", out)
        self.assertIn("recall", out)

    def test_silent_is_said_out_loud(self):
        rc, out = self.run_cli("--say", "What is the capital of France?")
        self.assertIn("she does not answer", out)

    def test_info_and_missing_bank(self):
        rc, out = self.run_cli("--info")
        self.assertEqual(json.loads(out)["dialogues"], 400)
        rc, out = self.run_cli("--bank", "/nonexistent/bank.txt", "--info")
        self.assertEqual(rc, 2)


if __name__ == "__main__":
    unittest.main()
