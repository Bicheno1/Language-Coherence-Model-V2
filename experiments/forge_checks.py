"""Checks of what the lcm_forge README and the paper (Training of the LCM, v1.1, sections 3.5-3.7) claim, using the Lucia bank as data forge never saw.

    python experiments/forge_checks.py --forge /path/to/lcm_forge_project

A) the character table (año, mañana, sí/si, punctuation, characters outside the table)
B) additivity: 400 dialogues ingested in 4 batches == in one go
C) the axis classifier against the bank's own axis labels (the paper reports ~82% on its own blind batch)
D) speaker_profile('Lucia'): the bank has exactly 25% per axis
E) are the seed files of lcm_forge and of this package the same phrases? (content, and the `source` field)
F) paper 3.6 'characters as mixtures of axes': can the dominant axis of a PURE-axis character be recovered from n of her lines?
Needs lcm_forge on disk; it is not part of this package."""
import sys, os, json, time, random, collections
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FORGE = sys.argv[sys.argv.index("--forge") + 1] if "--forge" in sys.argv else os.path.join(ROOT, "..", "lcm_forge_project")
if not os.path.isdir(os.path.join(FORGE, "lcm_forge")):
    sys.exit(f"lcm_forge not found at {FORGE}  (use --forge PATH)")
sys.path.insert(0, FORGE); sys.path.insert(1, ROOT)
from lcm_forge.store import Store as FStore
from lcm_forge.charset import Charset
from lcm_forge.axes import AxisModel, AX
from lcm.train.bank import load_bank, bank_to_script

data = load_bank(os.path.join(ROOT, "data", "lucia", "lucia_bank.txt"))
PKG_SEED = os.path.join(ROOT, "lcm", "train", "data", "axis_seed.json"); FORGE_SEED = os.path.join(FORGE, "lcm_forge", "data", "axis_seed.json")

print("A) character table (paper 3.5):")
c = Charset()
for w, exp in (("año", [1, 27, 15]), ("mañana", [13, 1, 27, 1, 14, 1]), ("sí", [19, 30]), ("si", [19, 9])):
    got = list(c.encode(w)); print(f"   {w:7s} paper {exp}  package {got}  {'OK' if got == exp else 'DIFFERENT'}")
print("   Ñ == ñ:", list(c.encode("Ñ")) == list(c.encode("ñ")), "| . , ? ! ¿ ¡ … ->", [c.num(x) for x in ".,?!¿¡…"], "(paper: 51 52 53 54 59 60 65) | € ->", c.num("€"), "(paper: 200+)")

print("\nB) additivity, 400 dialogues, 4 batches vs 1:")
t0 = time.time(); a = FStore(lang="en"); a.ingest_script(bank_to_script(data)); ta = time.time() - t0
b = FStore(lang="en")
for k in range(0, 400, 100): b.ingest_script(bank_to_script(data[k:k + 100]))
da, db = a.to_dict(), b.to_dict(); diff = [k for k in da if da[k] != db[k]]
print(f"   ingest {ta:.2f}s | identical: {not diff} | differing fields: {diff}")

print("\nC) axis classifier vs the bank's labels on her 400 replies (paper: ~82% on its own blind batch; chance 25%):")
for name, path in (("lcm_forge seed", FORGE_SEED), ("this package's seed", PKG_SEED)):
    m = AxisModel.from_seed(path); ok = collections.Counter(); tot = collections.Counter()
    for x in data: p = m.predict(x["reply"])[0]; tot[x["ax"]] += 1; ok[x["ax"]] += p == x["ax"]
    print(f"   {name:22s} {sum(ok.values())}/400 = {100*sum(ok.values())/400:.0f}%   per axis {dict((q, f'{ok[q]}/{tot[q]}') for q in AX)}")

print("\nD) speaker_profile('Lucia'); truth in the bank is 25% per axis:", a.speaker_profile("Lucia").get("axis_mix"))

A_, B_ = json.load(open(FORGE_SEED, encoding="utf-8")), json.load(open(PKG_SEED, encoding="utf-8"))
print("\nE) seed files: same phrases per axis:", {k: set(A_["phrases"][k]) == set(B_["phrases"][k]) for k in AX}, "| source in lcm_forge:", A_["meta"].get("source"), "| source in this package:", B_["meta"].get("source"))

print("\nF) dominant axis of a pure-axis character from n of her lines (chance 25%):")
orig = AxisModel.from_seed(FORGE_SEED)
def dominant(model, lines): return collections.Counter(model.predict(l)[0] for l in lines).most_common(1)[0][0]
def trial(model, pool, n, draws, rng):
    ok = tot = 0
    for ax in AX:
        L = [d["reply"] for d in pool if d["ax"] == ax]
        for _ in range(draws): tot += 1; ok += dominant(model, rng.sample(L, min(n, len(L)))) == ax
    return ok / tot
rng = random.Random(0)
print("   n lines   original seed (no training on these lines)   seed trained on the bank (held-out lines, mean of 3 splits)")
for n in (1, 5, 10, 20):
    o = trial(orig, data, n, 200, rng); bt = []
    for s in range(3):
        r = random.Random(s); idx = list(range(len(data))); r.shuffle(idx); test = [data[i] for i in idx[:80]]; train = [data[i] for i in idx[80:]]
        m = AxisModel("en").fit({q: [d["reply"] for d in train if d["ax"] == q] for q in AX}); bt.append(trial(m, test, min(n, 20), 200, r))
    print(f"   {n:7d}   {100*o:5.0f}%                                      {100*sum(bt)/3:5.0f}%")
