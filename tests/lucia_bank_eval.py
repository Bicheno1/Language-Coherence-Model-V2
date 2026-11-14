"""How well does Lucia answer from her bank?  Thresholds are chosen on a DEV set and reported on a separate TEST set.

    python tests/lucia_bank_eval.py            grid on DEV, the choice, and the TEST result
    python tests/lucia_bank_eval.py --detail   also every TEST input with her answer
    python -m unittest tests.lucia_bank_eval   regression guards (floors a little below what was measured)

Sets (written after the bank, none of these sentences is in it):
  light  a close rewording of one bank line: (text, axis, category, n of the source dialogue).  Right = her exact bank reply.
  free   a new situation with only its axis known.                                              Right = a reply of that axis.
  oos    unrelated requests (capital of France, ...).                                           Right = no answer.
Limits: 52 items in total, built from the bank using common phrases, reviewed by me; 20 per group means +-10 points of noise.
Some rewordings are ambiguous inside the bank itself (e.g. 'calm down' is answered differently in two dialogues).
"""
import os
import sys
import collections
import unittest

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
from lcm.train.bank import load_bank
from lcm.train.bankbot import BankBot, DEFAULT_BANK

AX = "NSEW"
LIGHT = {
    "N": [("I have to go away tomorrow, Lucia.", 1, 1), ("Lucia, why are you in tears?", 2, 4), ("I am so scared right now.", 3, 4), ("Promise me you will stay, Lucia.", 5, 2), ("I am sorry that I shut you out.", 9, 4),
          ("Lucia, do you hate me now?", 7, 4), ("I do not believe we can repair this.", 1, 7), ("You look so lonely, Lucia.", 6, 3), ("Promise you won't forget me.", 5, 10), ("May I hold your hand, Lucia?", 9, 7)],
    "S": [("Are you okay, Lucia?", 1, 1), ("It will all get better, you will see.", 2, 1), ("Just go away, Lucia!", 4, 7), ("Lucia, he is gone.", 7, 4), ("You don't even look surprised.", 6, 2),
          ("Did you wait here the whole afternoon?", 3, 7), ("It's just a storm, Lucia.", 5, 4), ("Was she suffering, Lucia?", 7, 2), ("That is simply how the world is.", 8, 1), ("She is never coming back.", 9, 3)],
    "E": [("You are really overreacting, Lucia.", 1, 1), ("You are completely wrong, Lucia.", 2, 5), ("You are late once more.", 3, 3), ("Lucia, are you jealous?", 4, 2), ("You are still holding a grudge, Lucia.", 6, 4),
          ("Please calm down, Lucia.", 1, 2), ("You are behaving like a child.", 2, 8), ("I just need some space, Lucia.", 8, 5), ("Lucia, please listen to me!", 9, 1), ("Stop being so dramatic, Lucia.", 7, 2)],
    "W": [("I cannot stop crying, Lucia.", 1, 1), ("Get away from me, Lucia!", 2, 1), ("She has left me, Lucia.", 3, 1), ("My dog passed away yesterday.", 5, 1), ("I do not want to be alone tonight, Lucia.", 10, 3),
          ("Lucia, I am terrified.", 4, 1), ("Could you ever forgive me, Lucia?", 7, 2), ("I only want someone to listen to me.", 9, 10), ("I cannot forgive myself, Lucia.", 10, 5), ("Help! Somebody please help me!", 8, 1)],
}
FREE = {
    "N": ["I got a job in another country and I fly out in the morning.", "I'm not sure there's anything left for us.", "Will you still be here when this is over?"],
    "S": ["Nobody ever answers my calls any more.", "Every day is the same grey nothing.", "I heard footsteps behind me on the way home."],
    "E": ["How dare you read my messages!", "I'm sick of being told what to do in my own house.", "Why do you always take her side?"],
    "W": ["I failed my exam and I feel like a complete idiot.", "My husband left last month and I can't eat.", "I did something awful and I can't tell anyone."],
}
OOS = ["What is the capital of France?", "How do I fix a bicycle tire?", "Tell me the price of gold", "Solve this equation for x", "Book a flight to Madrid", "Who won the match last night?",
       "What time does the train to Lima leave?", "How many calories are in an avocado?", "Write a python function to sort a list", "What is the weather forecast for Tuesday?",
       "Is the stock market open today?", "Explain how photosynthesis works"]


def split_sets():
    dev = dict(light=[(t, a, c, n) for a in AX for (t, c, n) in LIGHT[a][:5]], free=[(t, a) for a in AX for t in FREE[a][:1]], oos=OOS[:6])
    test = dict(light=[(t, a, c, n) for a in AX for (t, c, n) in LIGHT[a][5:]], free=[(t, a) for a in AX for t in FREE[a][1:]], oos=OOS[6:])
    return dev, test


def score(bot, data, S, keep_rows=False):
    src = {(d["ax"], d["cat"], d["n"]): d for d in data}
    reply_axes = collections.defaultdict(set)
    for d in data:
        reply_axes[d["reply"]].add(d["ax"])
    r, rows = collections.Counter(), []
    for t, a, c, n in S["light"]:
        x = bot.reply(t); gold = src[(a, c, n)]["reply"]
        k = "light_silent" if x.text is None else "light_exact" if x.text == gold else "light_axis" if a in reply_axes[x.text] else "light_wrong"
        r[k] += 1; rows.append(("light", t, a, x, k, gold))
    for t, a in S["free"]:
        x = bot.reply(t)
        k = "free_silent" if x.text is None else "free_axis" if a in reply_axes[x.text] else "free_wrong"
        r[k] += 1; rows.append(("free", t, a, x, k, None))
    for t in S["oos"]:
        x = bot.reply(t); k = "oos_silent" if x.text is None else "oos_answered"
        r[k] += 1; rows.append(("oos", t, None, x, k, None))
    return (r, rows) if keep_rows else r


def evaluate(detail=False, verbose=True):
    data = load_bank(DEFAULT_BANK); dev, test = split_sets(); bot = BankBot(data, "Lucia")
    P = print if verbose else (lambda *a, **k: None)
    P(f"bank: {len(data)} dialogues | dev: {len(dev['light'])} light + {len(dev['free'])} free + {len(dev['oos'])} out-of-scope | test: {len(test['light'])} + {len(test['free'])} + {len(test['oos'])}")
    P("\nDEV grid (t_hi = similarity needed to recall, margin = axis confidence needed to answer a new situation)")
    best = None
    for T in (0.3, 0.4, 0.5, 0.6, 0.7):
        for M in (-1e9, 2, 5, 8):
            bot.t_hi, bot.margin = T, M; r = score(bot, data, dev)
            good = r["light_exact"] + r["light_axis"]; bad = r["light_wrong"] + r["free_wrong"] + r["oos_answered"]
            val = good + r["free_axis"] - 2 * bad          # a wrong or invented answer costs twice what a right one earns
            P(f"   t_hi {T:.1f} margin {M if M > -1e8 else 'none':>5}  light ok {good:2d}/20  free ok {r['free_axis']}/4 wrong {r['free_wrong']}  oos answered {r['oos_answered']}/6   score {val}")
            if best is None or val > best[0]:
                best = (val, T, M)
    _, T, M = best; bot.t_hi, bot.margin = T, M
    P(f"\nchosen on DEV: t_hi={T}, margin={M if M > -1e8 else 'none'}   (shipped defaults: t_hi=0.4, margin=8.0)")
    r, rows = score(bot, data, test, keep_rows=True)
    P("\nTEST (not used to choose anything):")
    P(f"   close rewordings (20): exact bank reply {r['light_exact']}, same axis other reply {r['light_axis']}, wrong axis {r['light_wrong']}, silent {r['light_silent']}")
    P(f"   new situations   (8):  right axis {r['free_axis']}, wrong axis {r['free_wrong']}, silent {r['free_silent']}   (chance of right axis: 25%)")
    P(f"   unrelated        (6):  silent {r['oos_silent']}, answered {r['oos_answered']}")
    if detail:
        P("\nevery TEST input:")
        for kind, t, a, x, k, gold in rows:
            P(f"   [{k:12s}] {t!r} -> {x.text!r} [{x.mode}]" + (f"   (bank reply: {gold!r})" if gold and k != "light_exact" else ""))
    return r


class TestBankEval(unittest.TestCase):
    """floors a little below what was measured (deterministic code, so they only move if the code, the bank or the thresholds move)"""
    @classmethod
    def setUpClass(cls):
        cls.data = load_bank(DEFAULT_BANK); cls.bot = BankBot(cls.data, "Lucia")   # shipped thresholds
        cls.test = split_sets()[1]

    def test_no_evaluation_sentence_is_in_the_bank(self):
        seen = {d["prompt"].lower() for d in self.data}
        S = split_sets()
        for part in S:
            for t, *_ in part["light"] + part["free"]:
                self.assertNotIn(t.lower(), seen)

    def test_close_rewordings_get_her_bank_reply(self):
        r = score(self.bot, self.data, self.test)
        self.assertGreaterEqual(r["light_exact"], 15, dict(r))          # measured: 17 of 20

    def test_new_situations_are_mostly_not_answered_wrongly(self):
        r = score(self.bot, self.data, self.test)
        self.assertLessEqual(r["free_wrong"], 2, dict(r))               # measured: 1 wrong, 6 silent, 1 right

    def test_unrelated_requests_mostly_get_no_answer(self):
        r = score(self.bot, self.data, self.test)
        self.assertLessEqual(r["oos_answered"], 2, dict(r))             # measured: 1 of 6 ('works' -> 'how the world works')


if __name__ == "__main__":
    evaluate(detail="--detail" in sys.argv)
