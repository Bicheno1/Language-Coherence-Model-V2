"""BankBot: a character who answers as written in her emotional bank, and says nothing when she does not know.

    from lcm.train.bank import load_bank
    from lcm.train.bankbot import BankBot
    bot = BankBot(load_bank("data/lucia/lucia_bank.txt"), "Lucia")
    a = bot.reply("Could you ever forgive me, Lucia?")
    a.text, a.mode            # ('I can try. But I need some time.', 'recall')

Three modes, tried in this order (all deterministic):
  recall  the input is close to something in her bank (word similarity >= t_hi)
          -> the reply she gave there. She never produces a sentence she did not have.
  axis    the input is new, but the axis classifier (trained on what the others say in the bank) is confident (margin >= margin)
          -> the closest reply inside that emotional axis.
  silent  neither is confident -> no reply (text is None). This is deliberate: an invented reply is worse than none.

Defaults t_hi=0.4 and margin=8.0 were chosen on a DEV set of rewordings and reported on a separate TEST set
(see docs/LUCIA.md and tests/lucia_bank_eval.py). They are prototype values, not design decisions.

Command line (also `python -m lcm lucia ...`):
    --say "text"        one answer and how she decided        --demo     a short scripted conversation
    --chat              talk to her (empty line to quit)      --info     what is loaded
    --make-seed OUT     write an axis seed from the bank      [--field reply|prompt]
    options: --bank PATH  --character NAME  --t-hi X  --margin X
Standard library only.
"""
import os
import sys
import json
from collections import namedtuple

from .axes import AxisModel, AX
from .bank import load_bank, bank_to_script, bank_summary, axis_seed_from_bank
from .character import Character
from .store import Store

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_BANK = os.path.normpath(os.path.join(HERE, "..", "..", "data", "lucia", "lucia_bank.txt"))

Answer = namedtuple("Answer", "text mode similarity axis margin matched")
Answer.__doc__ = ("text: the reply or None | mode: 'recall' | 'axis' | 'silent' | similarity: word similarity of the best bank prompt (0-1) | "
                  "axis: predicted axis (N/S/E/W) or None | margin: axis classifier confidence or None | matched: the bank prompt she answered to, or None")


class BankBot:
    def __init__(self, dialogues, character="Lucia", t_hi=0.4, margin=8.0, lang="en"):
        if not dialogues:
            raise ValueError("the bank has no dialogues")
        self.character, self.t_hi, self.margin, self.dialogues = character, t_hi, margin, list(dialogues)
        self._all = self._char(self.dialogues, character, lang)
        self._by_axis = {a: self._char([d for d in self.dialogues if d["ax"] == a], character, lang) for a in AX
                         if any(d["ax"] == a for d in self.dialogues)}
        self._axis = AxisModel(lang).fit({a: [d["prompt"] for d in self.dialogues if d["ax"] == a] for a in AX})

    @staticmethod
    def _char(ds, character, lang):
        st = Store(lang=lang)
        st.ingest_script(bank_to_script(ds, character), source="bank")
        return Character(st, character, tau=0.0)           # the thresholds live in BankBot, not in Character

    def reply(self, text):
        r = self._all.respond(text, temperature=0)
        sim = r.confidence if r.text else 0.0
        if r.text and sim >= self.t_hi:
            return Answer(r.text, "recall", sim, None, None, r.matched)
        ax, mg = self._axis.predict(text)
        if mg >= self.margin and ax in self._by_axis:
            r2 = self._by_axis[ax].respond(text, temperature=0)
            if r2.text:
                return Answer(r2.text, "axis", sim, ax, round(mg, 2), r2.matched)
        return Answer(None, "silent", sim, ax, round(mg, 2), None)

    def info(self):
        s = bank_summary(self.dialogues)
        return dict(character=self.character, dialogues=s["dialogues"], per_axis=s["per_axis"], t_hi=self.t_hi, margin=self.margin)


DEMO = ["Lucia, I am terrified.", "You are really overreacting, Lucia.", "Could you ever forgive me, Lucia?", "My dog passed away yesterday.",
        "Nobody ever answers my calls any more.", "I failed my exam and I feel like a complete idiot.", "What is the capital of France?"]


def _show(a):
    how = {"recall": f"recall, similarity {a.similarity:.2f}, answering to {a.matched!r}", "axis": f"axis {a.axis}, margin {a.margin}, closest prompt {a.matched!r}",
           "silent": f"silent, best similarity {a.similarity:.2f}, axis guess {a.axis} (margin {a.margin})"}[a.mode]
    return f"{a.text if a.text is not None else '(she does not answer)'}\n   [{how}]"


def _opts(argv, names):
    o, rest, i = {}, [], 0
    while i < len(argv):
        if argv[i] in names and i + 1 < len(argv):
            o[argv[i][2:].replace("-", "_")] = argv[i + 1]; i += 2
        else:
            rest.append(argv[i]); i += 1
    return o, rest


def main(argv):
    o, rest = _opts(argv, ("--bank", "--character", "--t-hi", "--margin", "--say", "--make-seed", "--field"))
    if not argv or (not rest and not o.get("say") and not o.get("make_seed")):
        print(__doc__); return 0
    char, bank = o.get("character", "Lucia"), o.get("bank", DEFAULT_BANK)
    if not os.path.exists(bank):
        print(f"bank not found: {bank}"); return 2
    ds = load_bank(bank, char)
    if "--make-seed" in argv or o.get("make_seed"):
        out = o.get("make_seed")
        seed = axis_seed_from_bank(ds, o.get("field", "reply"), os.path.basename(bank))
        with open(out, "w", encoding="utf-8") as f:
            json.dump(seed, f, ensure_ascii=False, indent=1)
        print(f"{sum(len(v) for v in seed['phrases'].values())} lines -> {out}"); return 0
    bot = BankBot(ds, char, float(o.get("t_hi", 0.4)), float(o.get("margin", 8.0)))
    if "--info" in rest:
        print(json.dumps(bot.info(), ensure_ascii=False, indent=1))
    if o.get("say"):
        print(_show(bot.reply(o["say"])))
    if "--demo" in rest:
        for t in DEMO:
            print(f"\nYou:   {t}\n{char}: {_show(bot.reply(t))}")
    if "--chat" in rest:
        while True:
            try:
                t = input("\nYou: ").strip()
            except EOFError:
                break
            if not t:
                break
            print(f"{char}: {_show(bot.reply(t))}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
