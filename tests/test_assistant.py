"""
Checks on the assistant simulation with the LCM training system as the engine (tests/assistant_lcm_test.py, tests/assistant_test.py).
Data (actions, requests, out-of-scope requests) come from assistant_test.py. The numbers of the simulation are not asserted here: they are
reported by the scripts. These tests check the mechanics: an action is learned from its phrases, the answer is deterministic, and with a
high threshold the assistant does not act on requests that are outside its scope.
"""
import os, sys, unittest
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from lcm.train.store import Store
from lcm.train.character import Character

_src = open(os.path.join(HERE, "assistant_test.py"), encoding="utf-8").read()
_ns = {}
exec(_src[_src.index("A = {"):_src.index("STOP=")], _ns)
A, OOS = _ns["A"], _ns["OOS"]


def assistant(k=4, tau=0.25):
    lines = []
    for act, (train, _) in A.items():
        for p in train[:k]:
            lines += [f"User: {p}", f"Assistant: {act}"]
    st = Store(lang="en")
    st.ingest_script("\n".join(lines))
    return Character(st, "Assistant", tau=tau)


class TestAssistantOnTheLCM(unittest.TestCase):
    def test_each_training_phrase_returns_its_own_action(self):
        ch = assistant()
        for act, (train, _) in A.items():
            for p in train:
                self.assertEqual(ch.respond(p, temperature=0).text, act, p)

    def test_answer_is_deterministic_at_temperature_zero(self):
        ch = assistant()
        req = A["bold"][1][0]
        self.assertEqual(ch.respond(req, temperature=0).text, ch.respond(req, temperature=0).text)

    def test_high_threshold_does_not_act_outside_scope(self):
        ch = assistant(tau=0.45)
        acts = [t for t in OOS if ch.respond(t, temperature=0).kind == "repertoire"]
        self.assertEqual(acts, [])


if __name__ == "__main__":
    unittest.main()
