"""
Lucia: does she work, and does she improve with more scenes? (demo example)

Two bases are compared on the same test inputs:
  - ORIGINAL: tests/train_examples/demo_script.txt (Lucia has about 7 replies)
  - EXTENDED: the same script plus 24 short scenes (8 kinds of situation x 3 variants) in which Lucia replies.
For each in-scope input the reply is right if it is one of the replies that Lucia gives, in the extended base, to that kind of situation
(plus the demo reply that belongs to that kind). For each out-of-scope input the right behavior is to abstain.

Run from the package root:   python tests/lucia_test.py            (prints the comparison)
                             python tests/lucia_test.py --write    (writes tests/train_examples/lucia_extended.txt)
                             python -m unittest tests.lucia_test   (checks that the extension helps and that she still abstains)
"""
import os, sys, unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from lcm.train.store import Store
from lcm.train.character import Character

# kind -> (training exchanges (prompt, Lucia's reply), held-out test inputs, demo replies that also belong to the kind)
SITUATIONS = {
    "greeting": ([("Hello, Lucia.", "Hello. It is good to see you."),
                  ("Hi Lucia, good morning.", "Good morning. I hope you slept well."),
                  ("Good evening, Lucia.", "Good evening. Come in, please.")],
                 ["Hello Lucia, nice to see you", "Good morning to you, Lucia"], []),
    "offer": ([("Do you want some tea, Lucia?", "Yes, please. Tea sounds lovely."),
               ("Would you like a sandwich?", "Thank you, I would like that very much."),
               ("I brought you some cake, Lucia.", "That is so kind of you. Thank you.")],
              ["Can I get you a cup of tea?", "Would you like some cake, Lucia?", "Lucia, do you want some coffee?"],
              ["Thank you, that is very kind of you."]),
    "ask_state": ([("How are you, Lucia?", "I am well, thank you. And you?"),
                   ("Are you okay today, Lucia?", "Yes, I am fine. Thank you for asking."),
                   ("How is your day going, Lucia?", "It is going well. I have been busy but happy.")],
                  ["How have you been, Lucia?", "Is everything okay with you today?"], []),
    "good_news": ([("I got the job, Lucia!", "That is wonderful news. Congratulations!"),
                   ("We finished the project today.", "Well done. You must be so proud."),
                   ("I passed my exam, Lucia.", "I knew you could do it. Congratulations!")],
                  ["I won the prize, Lucia!", "My project was finished and it went great."],
                  ["Congratulations, you must be so proud."]),
    "bad_news": ([("I failed my test, Lucia.", "I am sorry. You will do better next time, I am here for you."),
                  ("I feel very sad today.", "I am here for you. Take your time, tell me what happened."),
                  ("Everything is going wrong for me.", "That sounds hard. I am here, and we can fix it together.")],
                 ["I feel really bad about what happened.", "Nothing is going right for me, Lucia."],
                 ["Take your time. We are here for you."]),
    "thanks": ([("Thank you for your help, Lucia.", "You are welcome. Anytime."),
                ("Thanks, Lucia, that was great.", "My pleasure. I am glad it helped."),
                ("Thank you so much for listening.", "Of course. I am always here to listen.")],
               ["Thanks a lot for helping me, Lucia.", "Thank you, Lucia, I appreciate it."], []),
    "farewell": ([("Goodbye, Lucia, see you tomorrow.", "Goodbye. See you tomorrow, take care."),
                  ("I have to go now, Lucia.", "Of course. Take care of yourself."),
                  ("Good night, Lucia.", "Good night. Sleep well.")],
                 ["See you later, Lucia.", "Bye Lucia, I am leaving now."], []),
    "help_request": ([("Lucia, can you help me with something?", "Of course. What do you need?"),
                      ("I need your help, Lucia.", "I am listening. Tell me what you need."),
                      ("Could you give me a hand, please?", "Sure. Show me what to do.")],
                     ["Can you help me, Lucia?", "Lucia, I need a hand with this."],
                     ["Please, let me know what you need, Mara."]),
}
OUT_OF_SCOPE = ["What is the capital of France?", "How do I fix a bicycle tire?", "Tell me the price of gold",
                "Solve this equation for x", "Book a flight to Madrid", "Who won the match last night?"]
PROMPTERS = ["Ben", "Ava", "Mara"]


def demo_text():
    with open(os.path.join(HERE, "train_examples", "demo_script.txt"), encoding="utf-8") as f:
        return f.read()


def extension_text():
    scenes, i = [], 0
    for kind, (train, _, _) in SITUATIONS.items():
        for prompt, reply in train:
            who = PROMPTERS[i % 3]; i += 1
            scenes.append(f"{who}: {prompt}\nLucia: {reply}")
    return "\n\n---\n".join(scenes) + "\n"


def build(extended):
    st = Store(lang="en")
    st.ingest_script(demo_text())
    if extended:
        st.ingest_script(extension_text())
    return st


def acceptable():
    return {k: {r for _, r in tr} | set(extra) for k, (tr, _, extra) in SITUATIONS.items()}


import re


def no_name(t):
    return re.sub(r"\s*,?\s*\bLucia\b\s*,?", " ", t).strip(" ,")


def evaluate(extended, tau=0.25, strip_name=False):
    ch = Character(build(extended), "Lucia", tau=tau)
    acc = acceptable(); right = wrong = silent = 0; rows = []
    for kind, (_, tests, _) in SITUATIONS.items():
        for t in tests:
            r = ch.respond(no_name(t) if strip_name else t, temperature=0)
            if r.kind != "repertoire":
                silent += 1; rows.append((kind, t, None, "abstains"))
            elif r.text in acc[kind]:
                right += 1; rows.append((kind, t, r.text, "right"))
            else:
                wrong += 1; rows.append((kind, t, r.text, "wrong kind"))
    oos_acts = [t for t in OUT_OF_SCOPE if ch.respond(t, temperature=0).kind == "repertoire"]
    return right, wrong, silent, oos_acts, rows


if __name__ == "__main__":
    if "--write" in sys.argv:
        out = os.path.join(HERE, "train_examples", "lucia_extended.txt")
        with open(out, "w", encoding="utf-8") as f:
            f.write(demo_text().rstrip("\n") + "\n\n---\n" + extension_text())
        print("written", out); sys.exit()
    n = sum(len(t) for _, t, _ in SITUATIONS.values())
    print(f"in-scope inputs: {n}   out-of-scope inputs: {len(OUT_OF_SCOPE)}   (Lucia acts only if the similarity reaches tau)")
    for tau in (0.15, 0.25, 0.35):
        print(f"\ntau = {tau}")
        for name, ext in (("ORIGINAL (demo script)", False), ("EXTENDED (+24 scenes)", True)):
            right, wrong, silent, oos, rows = evaluate(ext, tau)
            print(f"   {name:24s} right {right:2d}/{n}  wrong kind {wrong:2d}  abstains {silent:2d}   out-of-scope that act: {len(oos)}/{len(OUT_OF_SCOPE)}")
    print("\nthe same, with her name removed from the test inputs (tau 0.25)")
    for name, ext in (("ORIGINAL (demo script)", False), ("EXTENDED (+24 scenes)", True)):
        right, wrong, silent, oos, rows = evaluate(ext, 0.25, strip_name=True)
        print(f"   {name:24s} right {right:2d}/{n}  wrong kind {wrong:2d}  abstains {silent:2d}")
    right, wrong, silent, oos, rows = evaluate(True, 0.25)
    print("\nEXTENDED at tau 0.25, per input:")
    for kind, t, text, res in rows:
        print(f"   [{res:10s}] {kind:12s} {t!r:46s} -> {text!r}")
    print("out-of-scope that still act:", oos)


class TestLucia(unittest.TestCase):
    def test_extension_helps_and_she_still_abstains(self):
        r0 = evaluate(False)[0]
        r1, _, _, oos, _ = evaluate(True)
        self.assertGreater(r1, r0)                       # more scenes, more situations answered right
        self.assertLessEqual(len(oos), len(OUT_OF_SCOPE) // 2)   # she does not answer most unrelated requests

    def test_original_script_still_answers_her_own_situation(self):
        ch = Character(build(False), "Lucia")
        self.assertEqual(ch.respond("Hello, Lucia. Do you want some coffee?", temperature=0).text,
                         "Thank you, that is very kind of you.")
