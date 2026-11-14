"""Why does Lucia answer badly outside her bank? Data volume vs. calibration vs. reward. Run from anywhere: python experiments/lucia_diagnose.py
Six experiments (learning curves, ambiguity of the labels, confidence vs. correctness, hand-set knobs, a simulated reward). See docs/LUCIA.md, section 7."""
import sys, os, re, random, collections, math, statistics as S
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
BANK = os.path.join(ROOT, "data", "lucia", "lucia_bank.txt")
from lcm.train.store import Store
from lcm.train.character import Character
from lcm.train.reply import Retriever
from lcm.train.axes import AxisModel, AX
from lcm.train.lang import format_features
from lcm.train.bank import load_bank

data = load_bank(BANK)
reply_axes = collections.defaultdict(set)
for d in data: reply_axes[d["reply"].strip().lower()].add(d["ax"])
def script(items): return "\n\n---\n".join(f"{d['who']}: {d['prompt']}\nLucia: {d['reply']}" for d in items) + "\n"
def split(seed, k=80):
    r = random.Random(seed); x = data[:]; r.shuffle(x); return x[k:], x[:k]       # train pool, test
def strat_sub(items, frac, seed):
    r = random.Random(seed); out = []
    for a in AX:
        g = [d for d in items if d["ax"] == a]; r.shuffle(g); out += g[:max(1, round(len(g) * frac))]
    return out
def store_of(items):
    st = Store(lang="en"); st.ingest_script(script(items)); return st
FR = (0.125, 0.25, 0.5, 0.75, 1.0); SEEDS = range(6)
def mean(x): return sum(x) / len(x)

# ---------- classifiers (stdlib only) ----------
def toks(t, bigrams=False):
    w = re.findall(r"[a-z']+|[?!]", t.lower().replace("’", "'")); f = w + format_features(t, "en")
    if bigrams: f += [a + "_" + b for a, b in zip(w, w[1:])]
    return f
class NB:
    def __init__(self, bigrams=False): self.bg = bigrams
    def fit(self, items, key):
        self.c = {a: collections.Counter() for a in AX}; self.V = set()
        for d in items:
            t = toks(d[key], self.bg); self.c[d["ax"]].update(t); self.V.update(t)
        self.tot = {a: sum(self.c[a].values()) for a in AX}; return self
    def predict(self, text):
        t = toks(text, self.bg); V = len(self.V) + 1
        s = {a: sum(math.log((self.c[a][w] + .5) / (self.tot[a] + .5 * V)) for w in t) for a in AX}
        o = sorted(AX, key=s.get, reverse=True); return o[0], s[o[0]] - s[o[1]]
class KNN:
    def fit(self, items, key):
        self.docs = [(d["ax"], collections.Counter(toks(d[key]))) for d in items]
        df = collections.Counter(w for _, c in self.docs for w in c); N = len(self.docs)
        self.idf = lambda w: math.log(1 + N / (1 + df.get(w, 0)))
        self.v = [(a, self._v(c)) for a, c in self.docs]; return self
    def _v(self, c): return {w: (1 + math.log(n)) * self.idf(w) for w, n in c.items()}
    def predict(self, text):
        q = self._v(collections.Counter(toks(text))); nq = math.sqrt(sum(x * x for x in q.values())) or 1
        sc = []
        for a, v in self.v:
            nv = math.sqrt(sum(x * x for x in v.values())) or 1
            sc.append((sum(x * v.get(w, 0) for w, x in q.items()) / (nq * nv), a))
        sc.sort(reverse=True); vote = collections.Counter()
        for s, a in sc[:7]: vote[a] += s
        o = vote.most_common(); return o[0][0], o[0][1] - (o[1][1] if len(o) > 1 else 0)
class LR:   # softmax regression, bag of words, SGD
    def __init__(self, ep=40, lr=0.5, l2=1e-3): self.ep, self.lr, self.l2 = ep, lr, l2
    def fit(self, items, key):
        self.W = {a: collections.defaultdict(float) for a in AX}; rng = random.Random(0)
        X = [(collections.Counter(toks(d[key])), d["ax"]) for d in items]
        for e in range(self.ep):
            rng.shuffle(X)
            for c, y in X:
                z = {a: sum(self.W[a][w] * n for w, n in c.items()) for a in AX}; m = max(z.values())
                ex = {a: math.exp(z[a] - m) for a in AX}; s = sum(ex.values())
                for a in AX:
                    g = ex[a] / s - (1 if a == y else 0)
                    for w, n in c.items(): self.W[a][w] -= self.lr * (g * n + self.l2 * self.W[a][w]) / (1 + e)
        return self
    def predict(self, text):
        c = collections.Counter(toks(text)); z = {a: sum(self.W[a].get(w, 0) * n for w, n in c.items()) for a in AX}
        o = sorted(AX, key=z.get, reverse=True); return o[0], z[o[0]] - z[o[1]]
def acc(model, test, key="prompt"): return mean([model.predict(d[key])[0] == d["ax"] for d in test])

print("=" * 100); print("1) IS IT THE DATA?  learning curve, prompt -> Lucia's axis (accuracy, mean of 6 random 80/20 splits; chance 25%)")
print("   training pairs used:       " + "   ".join(f"{round(320*f):4d}" for f in FR))
for name, mk in (("NB (project's method)", lambda: NB()), ("NB + bigrams", lambda: NB(True)), ("kNN tf-idf", lambda: KNN()), ("softmax regression", lambda: LR())):
    row = []
    for f in FR:
        row.append(mean([acc(mk().fit(strat_sub(split(s)[0], f, s), "prompt"), split(s)[1]) for s in SEEDS]))
    print(f"   {name:24s}" + "   ".join(f"{100*x:3.0f}%" for x in row) + f"    gain last doubling (160->320): {100*(row[-1]-row[2]):+.0f} pts")

print("\n2) IS IT THE DATA?  learning curve of the project's REPLY retrieval (Character, tau=0 = always answers), same-axis rate")
row = []
for f in FR:
    r = []
    for s in SEEDS[:4]:
        tr, te = split(s); ch = Character(store_of(strat_sub(tr, f, s)), "Lucia", tau=0.0)
        r.append(mean([d["ax"] in reply_axes.get((ch.respond(d["prompt"], temperature=0).text or "").strip().lower(), set()) for d in te]))
    row.append(mean(r))
print("   training pairs used:       " + "   ".join(f"{round(320*f):4d}" for f in FR)); print("   Character same-axis:       " + "   ".join(f"{100*x:3.0f}%" for x in row))

print("\n3) IS THE LABEL ITSELF AMBIGUOUS?  (a ceiling no system can pass)")
by = collections.defaultdict(set)
for d in data: by[re.sub(r"[^a-z ]", "", d["prompt"].lower()).strip()].append(d["ax"]) if False else by[re.sub(r"[^a-z ]", "", d["prompt"].lower()).strip()].add(d["ax"])
amb = {p: a for p, a in by.items() if len(a) > 1}
print(f"   identical prompts that appear under 2+ axes: {len(amb)} -> examples: {[(p, sorted(a)) for p, a in list(amb.items())[:5]]}")
r = Retriever(store_of(data)); vs = [r._vec(d["prompt"]) for d in data]; cross = same = 0
for i, d in enumerate(data):
    best = max(((r._cos(vs[i], vs[j]), j) for j in range(len(data)) if j != i))
    if best[0] > 0.6: (same, cross)[0]; (cross := cross + 1) if data[best[1]]["ax"] != d["ax"] else (same := same + 1)
print(f"   prompts whose closest other prompt is very similar (cos>0.6): {same+cross}; of those the neighbor has a DIFFERENT axis in {cross} ({100*cross/max(1,same+cross):.0f}%)")

print("\n4) IS IT CALIBRATION?  does confidence predict correctness?  (project's classifier margin; 6 splits pooled = 480 held-out)")
pairs = []
for s in SEEDS:
    tr, te = split(s); m = AxisModel("en").fit({a: [d["prompt"] for d in tr if d["ax"] == a] for a in AX})
    for d in te: p, mg = m.predict(d["prompt"]); pairs.append((mg, p == d["ax"]))
pairs.sort(reverse=True); n = len(pairs)
print("   keep only the most confident X% -> accuracy:", "   ".join(f"top {int(q*100)}%: {100*mean([c for _, c in pairs[:int(n*q)]]):.0f}%" for q in (0.1, 0.25, 0.5, 0.75, 1.0)))
print("   Character reply: similarity (the quantity tau thresholds) vs same-axis correctness, 4 splits pooled (320):")
cp = []
for s in SEEDS[:4]:
    tr, te = split(s); ch = Character(store_of(tr), "Lucia", tau=0.0)
    for d in te:
        x = ch.respond(d["prompt"], temperature=0); cp.append((x.confidence, d["ax"] in reply_axes.get((x.text or "").strip().lower(), set())))
cp.sort(reverse=True); n = len(cp)
print("   ", "   ".join(f"top {int(q*100)}%: {100*mean([c for _, c in cp[:int(n*q)]]):.0f}%" for q in (0.1, 0.25, 0.5, 0.75, 1.0)))

print("\n5) IS IT THE HAND-SET KNOBS?  sweep tau / rel / temperature (same-axis rate among ANSWERED, and how many answered; 4 splits)")
def sweep(**kw):
    ans = ok = tot = 0
    for s in SEEDS[:4]:
        tr, te = split(s); ch = Character(store_of(tr), "Lucia", **{k: v for k, v in kw.items() if k != "temperature"})
        for d in te:
            x = ch.respond(d["prompt"], temperature=kw.get("temperature", 0), seed=1); tot += 1
            if x.kind == "repertoire": ans += 1; ok += d["ax"] in reply_axes.get((x.text or "").strip().lower(), set())
    return ans, tot, (ok / ans if ans else float("nan"))
for label, kws in (("tau 0.00", dict(tau=0.0)), ("tau 0.25 (default)", dict(tau=0.25)), ("tau 0.35", dict(tau=0.35)), ("tau 0.45", dict(tau=0.45)),
                   ("rel 0.5", dict(tau=0.0, rel=0.5)), ("rel 0.9", dict(tau=0.0, rel=0.9)), ("temperature 1.0 (sampling)", dict(tau=0.0, temperature=1.0)), ("temperature 2.0", dict(tau=0.0, temperature=2.0))):
    a, t, r_ = sweep(**kws); print(f"   {label:28s} answers {a}/{t}   same-axis among answered {100*r_:.0f}%")

print("\n6) WOULD A REWARD SIGNAL HELP?  simulated feedback loop (project has none): per-pair weights, +25% when a reply had the right axis, -20% otherwise")
res0, res1, loo = [], [], []
for s in SEEDS[:4]:
    tr, te = split(s); st = store_of(tr); ch = Character(st, "Lucia", tau=0.0); R = ch._r; P = st.pairs; V = R.vecs
    axes_of = [next(d["ax"] for d in tr if d["prompt"].strip() == p["pt"].strip() and d["reply"].strip() == p["rt"].strip()) for p in P]
    w = [1.0] * len(P)
    def pick(q, w, excl=None):
        sims = [(0.0 if i == excl else R._cos(q, V[i]), i) for i in range(len(P))]; best = max(x for x, _ in sims)
        g = collections.defaultdict(float)
        for x, i in sims:
            if x > 0 and x >= 0.7 * best: g[i] += x * w[i]
        return max(g, key=g.get) if g else None
    def evalw(w):
        out = []
        for d in te:
            j = pick(R._vec(d["prompt"]), w); out.append(j is not None and d["ax"] in reply_axes.get(P[j]["rt"].strip().lower(), set()))
        return mean(out)
    res0.append(evalw(w)); l = 0
    for ep in range(3):
        l = []
        for i, p in enumerate(P):
            j = pick(R._vec(p["pt"]), w, excl=i)
            if j is None: continue
            good = axes_of[j] == axes_of[i]; l.append(good); w[j] = min(5.0, w[j] * 1.25) if good else max(0.2, w[j] * 0.8)
    loo.append(mean(l)); res1.append(evalw(w))
print(f"   held-out same-axis BEFORE feedback: {100*mean(res0):.0f}%   AFTER 3 epochs of feedback: {100*mean(res1):.0f}%   (reward rate on training prompts, last epoch: {100*mean(loo):.0f}%)")
