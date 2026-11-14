"""
The same simulation as assistant_test.py, but with the LCM training system as the engine (lcm.train), without modifying it.

Lives in tests/ next to assistant_test.py. Run from the package root:  python tests/assistant_lcm_test.py
The data (actions, requests, topics, questions) is read from assistant_test.py so it is exactly the same.

A) Task assistant: each training example is a "User: <request>" turn followed by "Assistant: <action>".
   Store.ingest_script reads that conversation (format, lexicon, pairs) and Character(...).respond picks the action among
   those the assistant gave in similar situations; if none is similar enough (tau), it abstains.
B) Per-topic bases: each topic is one speaker (monologue) of a single Store; its layer (Store.uni_s) is its base.
   The router asks the base with the highest similarity; the router code is in this file, the counts come from the Store.
"""
import os, sys, math, collections

here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(here, ".."))      # package root (the folder that contains lcm/)
sys.path.insert(0, here)
src = open(os.path.join(here, "assistant_test.py"), encoding="utf-8").read()
ns = {}
exec(src[src.index("A = {"):src.index("STOP=")], ns)          # actions and out-of-scope requests
exec(src[src.index("T={"):src.index("docs={t:")], ns)           # topics, questions, two bases, off-topic
A, OOS, T, Q, BOTH, OOS2 = (ns[k] for k in ("A", "OOS", "T", "Q", "BOTH", "OOS2"))
import re
ns["re"] = re
exec(src[src.index("STOP="):src.index("def build")], ns)        # STOP and toks() of the baseline
pre = lambda text: " ".join(ns["toks"](text))

from lcm.train.store import Store
from lcm.train.character import Character
from lcm.train.lexicon import tokenize, norm, PUNCT

TAUS = (0.0, 0.15, 0.25, 0.35, 0.45)


def build_store(k, prep=lambda x: x):
    lines = []
    for act, (train, _) in A.items():
        for p in train[:k]:
            lines += [f"User: {prep(p)}", f"Assistant: {act}"]
    st = Store(lang="en")
    st.ingest_script("\n".join(lines))
    return st


print("=" * 86 + "\nA) TASK ASSISTANT with the LCM system (Store + Character): 12 actions, 48 requests, 20 out of scope\n" + "=" * 86)
print("(correct = right action; error = acted with another action; abstains = did nothing)")
detail = None
for k in (1, 2, 3, 4):
    st = build_store(k)
    ch = Character(st, "Assistant")
    print(f"\ntraining examples per action: {k}   (assistant pairs: {ch.n_pairs()})")
    print(f"   {'similarity threshold':>20s}{'correct':>10s}{'error':>8s}{'abstains':>13s}   | {'out of scope: acts by mistake':>36s}")
    for tau in TAUS:
        ch.tau = tau
        ok = err = 0; rows = []
        for a, v in A.items():
            for t in v[1]:
                r = ch.respond(t, temperature=0)
                rows.append((a, t, r))
                if r.kind == "repertoire":
                    ok += (r.text == a); err += (r.text != a)
        n = len(rows); ab = n - ok - err
        fa = sum(1 for t in OOS if ch.respond(t, temperature=0).kind == "repertoire")
        print(f"   {tau:>20.2f}{100*ok/n:9.0f}%{100*err/n:7.0f}%{100*ab/n:12.0f}%   | {100*fa/len(OOS):35.0f}%")
        if k == 4 and tau == 0.25:
            detail = (ch, rows)
ch, rows = detail
print("\nerrors with 4 examples and threshold 0.25:")
for a, t, r in rows:
    if r.kind == "repertoire" and r.text != a:
        print(f"   {t!r:55s} real={a:12s} eligio={r.text} (sim {r.confidence:.2f})")
print("out of scope requests that still act at threshold 0.25:", [(t, ch.respond(t, temperature=0).text) for t in OOS if ch.respond(t, temperature=0).kind == "repertoire"])

print("\n" + "=" * 86 + "\nA2) the same, with the LCM system, but preparing the text like the baseline (stop-word list and suffix stripping)\n" + "=" * 86)
print("(only the text going into Store and Character changes; the engine is the same)")
for k in (1, 2, 3, 4):
    st = build_store(k, pre); ch = Character(st, "Assistant")
    print(f"\ntraining examples per action: {k}")
    print(f"   {'similarity threshold':>20s}{'correct':>10s}{'error':>8s}{'abstains':>13s}   | {'out of scope: acts by mistake':>36s}")
    for tau in TAUS:
        ch.tau = tau; ok = err = n = 0
        for a, v in A.items():
            for t in v[1]:
                r = ch.respond(pre(t), temperature=0); n += 1
                if r.kind == "repertoire":
                    ok += (r.text == a); err += (r.text != a)
        fa = sum(1 for t in OOS if ch.respond(pre(t), temperature=0).kind == "repertoire")
        print(f"   {tau:>20.2f}{100*ok/n:9.0f}%{100*err/n:7.0f}%{100*(n-ok-err)/n:12.0f}%   | {100*fa/len(OOS):35.0f}%")

print("\n" + "=" * 86 + "\nB) PER-TOPIC BASES: each topic is a speaker of the Store; the router picks the base with the highest similarity\n" + "=" * 86)
st2 = Store(lang="en")
for topic, sents in T.items():
    st2.ingest_script("\n".join(sents), monologue=topic)
res = st2.lex.resolve


def toks2(text):
    return [res(norm(t)) for t in tokenize(text) if norm(t) not in PUNCT]


layers = {}
for topic in T:
    key = st2._speaker_key(topic)
    c = collections.Counter()
    for w, n in st2.uni_s[key].items():
        c[res(w)] += n
    layers[topic] = c
df = collections.Counter(w for c in layers.values() for w in c)
N = len(layers)
idf = lambda w: math.log(1 + N / (df.get(w, 0) + 0.5))
vec = lambda ws: {w: (1 + math.log(n)) * idf(w) for w, n in collections.Counter(ws).items() if w in df}
cents = {t: vec([w for w, n in c.items() for _ in range(n)]) for t, c in layers.items()}


def cos(a, b):
    na = math.sqrt(sum(x * x for x in a.values())); nb = math.sqrt(sum(x * x for x in b.values()))
    return sum(x * b.get(w, 0) for w, x in a.items()) / (na * nb) if na and nb else 0.0


def scores(q):
    v = vec(toks2(q))
    return sorted(((cos(v, c), t) for t, c in cents.items()), reverse=True)


n = ok = 0
for t, qs in Q.items():
    for q in qs:
        n += 1; ok += scores(q)[0][1] == t
print(f"una pregunta -> su base: {ok}/{n} = {100*ok/n:.0f}%")
for tau in (0.0, 0.1, 0.2):
    print(f"   off-topic questions that would be answered at threshold {tau:.1f}: {sum(1 for q in OOS2 if scores(q)[0][0] >= tau)}/{len(OOS2)}")
print("questions that need TWO bases: the ones above 0.15 are used")
for q, want in BOTH:
    s = scores(q); got = [t for sc, t in s if sc >= 0.15]
    print(f"   {q!r}\n      esperado {want}  elegido {got}  ({', '.join(f'{t} {sc:.2f}' for sc, t in s)})")
