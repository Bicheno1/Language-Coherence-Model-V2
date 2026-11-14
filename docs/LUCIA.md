# Lucia v2: a character that answers from her emotional bank (LCM v2.2)

Everything about how Lucia answers, how she was "trained", how she was measured, what does not work, and how to repeat every number.
All text of the character is English. Standard library only (Python 3.8+).

## Lucia v1 and Lucia v2

There are two Lucias in this package and I keep them apart.

**Lucia v1** is the first one: the character of the demo script, a small initial character. She has 10 pairs of her own, which I extended with 24 two-line scenes (34 pairs in all). I used her to check a simple question: does a character answer what she has lived and abstain on the rest?

**Lucia v2** is the incremented one: the bank of 400 dialogues that I wrote. I built her to see the limits of training with a somewhat larger base. This document is about v2.

| | **Lucia v1** (first version) | **Lucia v2** (this document) |
|---|---|---|
| Her base | **Short**: the demo script `tests/train_examples/demo_script.txt`, plus 24 scenes (8 kinds of situation x 3 variants) in `tests/train_examples/lucia_extended.txt` | **400 dialogues**, 4 emotional axes x 10 categories x 10 dialogues, exactly 100 per axis (`data/lucia/lucia_bank.txt`) |
| Where she lives | `tests/lucia_test.py` (demo script vs. extended scenes) | `lcm/train/bank.py`, `lcm/train/bankbot.py`, `python -m lcm lucia ...` |
| How she answers | `Character` over a `Store`: the closest situation she has lived (shared words), with the threshold `tau` | `BankBot`: recalls the closest prompt of her bank, or answers by axis when unsure, or stays silent |
| Labels | None: scenes are grouped by kind of situation | Every dialogue has one of the four axes (North, South, East, West), assigned by me |
| Character | An initial stand-in: polite, with lexical situations (greetings, offers, thanks) | The character I wrote: vulnerable, with outbursts |
| Evaluation | 17 held-out inputs and 6 out-of-scope requests (`tests/lucia_test.py`) | DEV/TEST protocol with fixed thresholds, 52 inputs (`tests/lucia_bank_eval.py`, section 5) |
| What it showed | She answers what she has lived, abstains on the rest, and improves with more scenes (14 of 17 right at `tau` 0.25) on situations defined by their words | The same idea with a larger base: she answers well when the input is close to her bank, but more data of the same kind does not widen her (section 7) |

I do not mix the two: the v1 scenes are lexical and do not match Lucia v2, so I do not put them into her training (see section 9).

## 0. At a glance

| Question | Answer (measured) |
|---|---|
| What is she? | A character who answers **with sentences already written in her bank** (400 dialogues, 4 emotional axes). She does not generate text. |
| Does she answer well? | **Yes, when the input is close to something in her bank**: 17 of 20 fresh close rewordings got her exact bank reply (TEST set). |
| And for new situations? | **Mostly she stays silent** (6 of 8 on TEST, 3 of 4 on DEV). The few answers are too few to rate (2 of 3 had the right axis). She does not improvise. |
| Unrelated requests? | Silent on 5 of 6 (TEST). One leak: a shared word ("works" -> "how the world works"). |
| Is more data the problem? | **Not more of the same kind.** Accuracy is flat from 40 to 320 training pairs (section 7). |
| Is the calibration or the reward the problem? | The hand-set knobs do nothing; a reward signal **does not exist** in the code (a simulated one gives about +4 points, inside noise); the confidence signal used before was the wrong one (section 7). |
| Main cause of bad answers | One line of text often does not decide the emotion: 71% of near-identical prompts belong to different axes. Context is missing, not volume. |
| Speed | Parse 4 ms, build 0.21 s, one reply about 1 ms. Nothing is stored between runs: she is rebuilt from the bank each time. |

## 1. What was added to the package

```
data/lucia/lucia_bank.txt                  the bank, in my own layout (axis headings, category titles, numbered dialogues)
lcm/train/bank.py                          reads a bank; bank_to_script; axis_seed_from_bank
lcm/train/bankbot.py                       BankBot (recall / axis / silent) and its command line
lcm/train/data/axis_seed_lucia.json        400 of her lines labeled by the bank's axis (made from the bank)
lcm/train/data/axis_seed*.json             provenance restored: meta.source = "first_version" (it had been lost in v2.1)
lcm/__main__.py                            command:  python -m lcm lucia ...
tests/test_bankbot.py                      17 unit tests
tests/lucia_bank_eval.py                   evaluation sets, threshold grid on DEV, result on TEST, 4 regression guards
tests/lucia_generic_scenes_eval.py         an earlier experiment with polite-assistant scenes (a stand-in, NOT the real Lucia); 3 tests
tests/train_examples/lucia_generic_en.txt  those scenes as a script
experiments/lucia_diagnose.py              data volume vs. calibration vs. reward (six experiments, about 10 minutes)
experiments/forge_checks.py                checks of lcm_forge and of the paper's claims on the Lucia bank
docs/LUCIA.md                              this file
```
Nothing from v2.1 was deleted. The four suites of the README give the same result before and after (62 + 14 + 41 tests).

## 2. The bank (data)

### 2.1 Layout
`data/lucia/lucia_bank.txt` keeps my layout. `parse_bank` also reads a simplified layout (`## NORTH`, `# 1 title`, `Name: text`); both give the same dialogues (tested).

```
NORTH (V) — We                                  axis heading: a line that starts with NORTH/SOUTH/EAST/WEST and has no colon
100 dialogues · Connection, future and hope     free text: ignored (descriptions and subtitles)
1. Miedo a perder a alguien                     category heading: "N. title", no colon in the title
1. Bernard: I'm leaving tomorrow, Lucia.        someone else speaks (the leading number is ignored)
Lucia: But... we just started spending time together.     the character answers  -> one dialogue
```
A dialogue is **two consecutive lines**: another person, then Lucia. The speaker named as the character is matched ignoring case and accents (`Lucia` = `Lucía`).
A category title containing a colon would be read as a spoken line; a dialogue before any axis heading is an error.

### 2.2 What the label means
Each dialogue carries the **axis of Lucia's reply**, assigned by me: N = V (we, connection, future), S = I (it/they: absence, fear, coldness),
E = Lv (I: anger, pride, defense of the self), W = Gv (you: empathy, calm, attention to the other). The label is not computed anywhere.
The bank has 4 axes x 10 categories x 10 dialogues = **400 dialogues, exactly 100 per axis**.

### 2.3 Provenance
- Lucia (the bank, the character and her training) is my work. `lucia_bank.txt` is my document converted to the package layout, with the headings translated into English; I still have to compare it with my master copy.
- The axis seeds `axis_seed.json` (English, 180 phrases) and `axis_seed_es.json` (Spanish, 100) are common phrases that I chose; I defined the axes and what each one means; `meta.source = first_version` in both files.
- The evaluation sets and tests (section 5) and the generic scenes (section 9) use common phrases, reviewed by me.

## 3. How she answers

```
input text
   |
   v
[1] word similarity (TF-IDF cosine) to every prompt in her bank          Character over a Store of the whole bank
   best similarity >= t_hi (0.4) ? ---- yes ---> RECALL: the reply she gave there
   | no
   v
[2] axis classifier (Naive Bayes on the prompts of the bank, by axis)    AxisModel
   margin >= margin (8.0) ? ---- yes ---> AXIS: the closest reply inside the predicted axis (a Store per axis)
   | no
   v
[3] SILENT: no reply (text is None)
```
- She **never says a sentence that is not in her bank** (tested). The only things chosen are *which* bank reply.
- It is **deterministic**: same input, same answer (temperature 0). There is no sampling in `BankBot`.
- `Answer(text, mode, similarity, axis, margin, matched)` says how she decided: `matched` is the bank prompt she answered to.
- Silence is deliberate. The measured alternative (always answering) gives the right axis only about 30% of the time on new situations (section 7).
- **Relation to the rest of v2**: `BankBot` uses `Store` and `Character` from `lcm/train` (the same code behind `python -m lcm character`) with `tau=0` and puts the thresholds in the bot.
  The old path (`lcm train ingest` + `lcm character`) still works and is unchanged.

### 3.1 The two thresholds
| name | default | meaning | how it was chosen |
|---|---|---|---|
| `t_hi` | 0.4 | similarity needed to recall a bank reply | grid on the DEV set (`tests/lucia_bank_eval.py`) |
| `margin` | 8.0 | classifier confidence needed to answer a *new* situation | same grid |

The score on DEV was `right answers - 2 x (wrong or invented answers)`. The grid (5 values of `t_hi` x 4 of `margin`) picked 0.4 and 8.0. These are **prototype values**: with 20 DEV items the choice is not sharp (several nearby cells score within 1–2 points).

## 4. Training: what it means here

There is **no gradient training and no hidden state**. "Training" is three things, all instant:
1. **Ingest**: the bank becomes a script (`bank_to_script`) and goes into a `Store` (counts of words, pairs prompt -> reply per speaker). One Store for the whole bank and one per axis.
2. **Fit the axis classifier**: Naive Bayes over the words (plus format features such as question mark, exclamation, capitals, negation, person) of the *prompts*, labeled by the dialogue's axis.
3. **Choose two thresholds** on a held-out DEV set (section 3.1).

### 4.1 Retrain after editing the bank
```
python -m lcm lucia --info                                  # dialogue count per axis: did the bank parse as you expect?
python -m lcm lucia --make-seed lcm/train/data/axis_seed_lucia.json     # optional: refresh the seed made from her lines
python tests/lucia_bank_eval.py                             # the grid on DEV and the result on TEST
python -m unittest tests.test_bankbot tests.lucia_bank_eval # 17 + 4 tests (they assert 400 dialogues, 100 per axis: change them if your bank differs)
```
To change the thresholds without editing the code: `python -m lcm lucia --t-hi 0.5 --margin 5 --say "..."`.

### 4.2 A different character or bank
`python -m lcm lucia --bank my_bank.txt --character Marina --say "..."`. The bank needs axis headings and two-line dialogues (section 2.1).
The evaluation sets in `tests/lucia_bank_eval.py` are written for Lucia's bank: write your own (close rewordings, new situations, unrelated requests) and keep the DEV/TEST split.

### 4.3 What actually improves her, in order of measured effect
1. **More dialogues that are *near* what people will say** (rewordings and variants of the same situation): they raise recall directly.
2. **Multi-turn dialogues with a scene tag** (who speaks to whom, what just happened): the only change that can move the ceiling (section 7.3).
3. **More single-exchange dialogues of new situations**: does *not* help (flat learning curve, section 7.1).

## 5. Evaluation

### 5.1 Protocol
- Three kinds of input, written **after** the bank and none of them in it (tested): **light** = a close rewording of one bank line (right = her exact bank reply); **free** = a new situation with only its axis known (right = a reply of that axis);
  **oos** = unrelated requests (right = no answer).
- DEV: 20 light + 4 free + 6 oos (used to choose the thresholds). TEST: 20 light + 8 free + 6 oos (used once, after choosing). The inspection of failures below was done **after** the thresholds were fixed; nothing was tuned afterwards.
- Noise: 20 items per group means about +-10 points. DEV and TEST differ by 20 points on light inputs (13/20 vs 18/20 with exact-or-same-axis), which shows it.

### 5.2 Results (shipped defaults: `t_hi` 0.4, `margin` 8.0)
| set | close rewordings (light) | new situations (free) | unrelated (oos) |
|---|---|---|---|
| DEV  | 12/20 exact + 1 same axis = 13/20; 3 wrong axis, 4 silent | 1 right axis, 0 wrong, 3 silent | 0 of 6 answered |
| TEST | **17/20 exact bank reply**, 1 same axis other reply, 1 wrong axis, 1 silent | 1 right axis, 1 wrong axis, **6 silent** (chance of right axis: 25%) | 5 silent, 1 answered |

Pooled light, DEV + TEST: **29/40 exact bank reply (72%)**, 31/40 exact-or-same-axis (78%).

### 5.3 A real conversation (`python -m lcm lucia --demo`)
```
You:   Lucia, I am terrified.                           Lucia: I understand. Would holding my hand help?        [recall, similarity 0.51, answering to "I'm terrified."]
You:   You are really overreacting, Lucia.              Lucia: DON'T TELL ME HOW TO FEEL!                       [recall, similarity 0.58]
You:   Could you ever forgive me, Lucia?                Lucia: I can try. But I need some time.                 [recall, similarity 0.50]
You:   My dog passed away yesterday.                    Lucia: Oh, I'm so sorry. I know how much you loved him. [recall, similarity 0.64]
You:   Nobody ever answers my calls any more.           Lucia: (she does not answer)                            [silent, best similarity 0.26, axis guess W, margin 7.17]
You:   I failed my exam and I feel like a complete idiot.  Lucia: You're not a burden for needing someone.       [axis W, margin 10.75, closest prompt "I feel like a burden."]
You:   What is the capital of France?                   Lucia: (she does not answer)                            [silent]
```
The sixth is the weak kind of answer: right axis (empathy toward you), but not a reply to what was said.

### 5.4 What failed, and why (TEST and DEV, after the thresholds were fixed)
| input | she said | why |
|---|---|---|
| "I have to go away tomorrow, Lucia." (DEV) | "Alright. I'm leaving." | the words *go away* match "Go away!" from the South axis; the intended source was "I'm leaving tomorrow" (North) |
| "I do not believe we can repair this." | "You don't have to do it alone." | closest in words to "I don't think I can do this."; same axis (North), so counted as right-by-axis, not exact |
| "You are completely wrong, Lucia." (DEV) | "I DON'T KNOW! I JUST CAN'T STOP CRYING!" | the word *wrong* matches "Lucia, what's wrong?!" (South): wrong axis |
| "I cannot stop crying, Lucia." (DEV) | "I CAN'T! I'm trying, but I can't!" | **ambiguity inside the bank**: "stop crying" is a North dialogue ("Lucia, please stop crying.") and a West one ("I can't stop crying."). Wrong axis against the West source |
| "Are you okay, Lucia?" (DEV) | "NO! IT'S NOT OKAY!" | matches "It's okay, it's okay." (South); same axis as the source, so right-by-axis, not exact |
| "Please calm down, Lucia." | "I CAN'T! I CAN'T!" | **ambiguity inside the bank**: "calm down" has two dialogues (East: "I'M CALM! I'M JUST ANGRY!"; South: "I CAN'T! I CAN'T!"). Scored wrong, defensible |
| "Did you wait here the whole afternoon?" | silent | bank has "You waited all afternoon?". Likely because inflections (*wait / waited*) are never merged by design (not verified) |
| "Lucia, why are you in tears?" (DEV) | silent | no shared words with "Why are you crying?" apart from *why are you* |
| "Will you still be here when this is over?" | "NO! YOU DON'T GET TO DECIDE WHEN I'M FINISHED!" | wrong axis, from the word *over* ("This conversation is over") |
| "Explain how photosynthesis works" | "Then the world is terribly unfair." | the word *works* ("That's just how the world works") |
| "I'm not sure there's anything left for us." | "There has to be something! There just has to!" | right axis and a fitting answer (counted as right) |

Every failure is a **word-overlap** failure or a **label ambiguity**. None is a missing-data failure.

### 5.5 Regression guards (`python -m unittest tests.lucia_bank_eval`)
Floors a little below what was measured: at least 15/20 exact on light TEST, at most 2 wrong on free TEST, at most 2 answered on oos TEST. The code is deterministic, so they move only if the code, the bank or the thresholds move.

## 6. Using her

```
python -m lcm lucia --say "Could you ever forgive me, Lucia?"
python -m lcm lucia --demo      python -m lcm lucia --chat      python -m lcm lucia --info
python -m lcm lucia --make-seed out.json [--field reply|prompt]
options: --bank PATH  --character NAME  --t-hi X  --margin X
```
```python
from lcm.train.bank import load_bank
from lcm.train.bankbot import BankBot, DEFAULT_BANK
bot = BankBot(load_bank(DEFAULT_BANK), "Lucia")
a = bot.reply("Could you ever forgive me, Lucia?")
a.text, a.mode, a.similarity, a.matched     # ('I can try. But I need some time.', 'recall', 0.5, 'Can you forgive me?')
```
`bot.t_hi` and `bot.margin` can be changed on the object (the evaluation grid does this).

## 7. Why does she answer badly outside her bank? Data vs. calibration vs. reward

Protocol (`experiments/lucia_diagnose.py`): random 80/20 splits of the 400 dialogues, 4 to 6 repeats, training sets stratified by axis. Metric: **same-axis rate** = the reply belongs to the gold axis
(a proxy: it does not judge wording). Chance is 25%. Sanity check: 316 of 320 training prompts retrieve their own reply, so the measurement code is not what makes held-out numbers low.

| Suspect | Verdict |
|---|---|
| More data (of the same kind) | **Not the limiter** (7.1) |
| Reward system | **Does not exist in the code**; a simulated one: +4 points, inside noise (7.5) |
| Hand-set knobs (`tau`, `rel`, `temperature`) | **Not the limiter** (7.4) |
| The confidence *signal* | **Partly**: reply similarity says nothing about emotional fit; the axis classifier's margin does (7.3) |
| Task design | **Main cause** (7.2) |

### 7.1 Learning curve, prompt -> axis (mean of 6 splits)
| training pairs | 40 | 80 | 160 | 240 | 320 |
|---|---|---|---|---|---|
| Naive Bayes (the project's) | 50% | 47% | 54% | 54% | 53% |
| Naive Bayes + bigrams | 49% | 49% | 51% | 53% | 53% |
| kNN (TF-IDF cosine) | 48% | 42% | 44% | 43% | 44% |
| Softmax regression | 49% | 46% | 47% | 47% | 47% |
| Reply retrieval by words (same-axis) | 34% | 31% | 27% | 29% | 30% |

Flat. Four different learners reach the same ceiling, so the limit is in the signal, not in one learner. Each dialogue of the bank is a one-off situation and its categories are thematic, not lexical: a new input rarely shares words with the right category.

### 7.2 Is the label ambiguous?
3 identical prompts appear under 2 axes ("Are you crying?" N and S; "He chose someone else." E and S; "What happened to you?" E and S). 82 prompts have a very similar neighbor (cosine > 0.6) and in **58 of them (71%) the neighbor has a different axis**.
The same words, said in another relationship or scene, get another emotional answer. One line cannot determine the axis.

### 7.3 Does confidence predict correctness?
Keep only the most confident X%:
| signal | top 10% | top 25% | top 50% | top 75% | all |
|---|---|---|---|---|---|
| axis classifier margin -> accuracy | 79% | 74% | 69% | 61% | 53% |
| reply similarity (what `tau` thresholds) -> same-axis | 31% | 26% | 28% | 30% | 31% |

The margin is informative; the similarity is flat. That is why `BankBot` uses the margin for new situations and similarity only to decide *recall*.

### 7.4 Hand-set knobs (4 splits)
| setting | answered | same-axis among answered |
|---|---|---|
| tau 0.00 / 0.25 (default) / 0.35 / 0.45 | 319 / 294 / 228 / 136 of 320 | 31% / 31% / 31% / 26% |
| rel 0.5 / 0.9 | 319 / 319 | 31% / 31% |
| temperature 1.0 / 2.0 | 319 / 319 | 32% / 32% |

### 7.5 Would a reward help? (simulation; the code has none)
Per-pair weights, +25% when the chosen reply had the right axis, -20% otherwise, 3 passes (leave-one-out): held-out same-axis **30% -> 34%**. Reward rate on training prompts at the last pass: 46%.
The test is optimistic (the reward used the gold axis). A weight per pair only re-ranks pairs that retrieval already finds, and retrieval is the weak part. A reward would have to act on the classifier and come from real human feedback.

## 8. Checks of lcm_forge and of the paper (`experiments/forge_checks.py`)
The paper is "Training of the Language Coherence Model (LCM)", v1.1. Data: the Lucia bank, which `lcm_forge` never saw.

| claim | result |
|---|---|
| 26 tests of lcm_forge pass | **yes** (26/26) |
| character numbers: año, mañana, sí/si, punctuation 51–54, 59, 60, 65, extra characters from 200 | **all equal** |
| additivity: batches == all at once | **yes** on 400 dialogues (identical, 0.10 s) |
| axis classifier ~82% on the paper's blind batch | **46%** on Lucia's 400 replies with the bank's labels (per axis N 33, S 37, E 61, W 55 of 100). Different style and labeling criterion, so not like for like; but it is the test with phrases from outside the seed that the paper says is missing |
| `speaker_profile('Lucia')` | V 14%, I 16%, Lv 46%, Gv 25%; the bank is exactly 25% each: East is overstated almost 2x |
| characters as mixtures of axes: dominant axis of a pure-axis character from n lines | original seed: 50 / 54 / 59 / 61% for n = 1 / 5 / 10 / 20 (it stops improving: a systematic bias). Seed trained on the bank (held-out lines): 69 / 86 / 93 / 100%. The second is optimistic: same style and same writer (me) |
| paper section 6.7 examples ("hola compañero", "hola camarada") | **not reproducible**: they are not in the zip |
| Spanish 33/40 on a blind batch | **not reproducible**: the batch is not in the zip |
| seed files of forge and of v2.1 | same phrases; v2.1 had lost `source: first_version` (**restored in v2.2**) |
| typed relations with spaCy | not tested (spaCy not installed) |

## 9. Lucia v1 and the polite-assistant scenes (an earlier experiment, a stand-in, not the real Lucia v2)
An earlier experiment used 14 situations x 6 scenes (+ 8 situations x 3) of a polite assistant (`tests/train_examples/lucia_generic_en.txt`, `tests/lucia_generic_scenes_eval.py`).
At `tau 0.25`, right answers on fresh inputs (test B, 28): **9 -> 17 -> 20**; on the set used to diagnose round 1 (test A, 42): 17 -> 34 -> 37. Cost: unrelated requests that still got an answer: 2 -> 4 -> 5 of 12.
The test `test_she_still_abstains_on_most_unrelated_requests` is kept as an **expected failure** on purpose. Those scenes are lexical (greetings, offers, thanks), which is where word retrieval works. They do not match Lucia v2 (vulnerable, outbursts): I do not mix them into her training.

## 10. Known limitations
- **One exchange per dialogue.** No memory of the conversation. This is the main limit (7.2).
- **Word overlap decides recall.** Single shared words cause wrong or invented-looking answers (*go away*, *works*, *over*); inflections are not merged by design.
- **New situations:** she is silent 6 of 8 times on TEST (3 of 4 on DEV). She answered only 3 times in total (2 with the right axis): too few to estimate. The axis classifier alone is right 53% of the time overall and 69% when only its more confident half is used (7.3).
- **Labels:** the bank assigns one axis per dialogue; some prompts are legitimately two (7.2).
- **Evaluation size:** 52 common sentences, reviewed by me; +-10 points per group; thresholds are prototype values.
- **Conversion:** I converted the bank from my document to the package layout; I still have to verify it against my master copy.
- **Not tested here:** the 22 tests that need baked WordNet (still SKIP), spaCy relations, any Spanish behavior of Lucia (everything is English by request).
- **Privacy/permissions:** a bank made from other people's texts needs their permission. `ingest --no-text` disables replies.

## 11. Recommended next steps (in order)
1. **Context, not volume**: write 3–6 turn dialogues with a scene tag (relationship, what just happened) and feed the previous turn to the classifier. Measure with the same DEV/TEST protocol.
2. **Several axes per prompt**: label ambiguous prompts with a distribution, not one forced label.
3. **Ask when unsure** instead of silence: below the margin, a clarifying line in her own voice, taken from the bank.
4. **A human read of 50–100 answers**: the same-axis metric does not judge tone or continuity.
5. **If a reward is wanted**: thumbs up/down per reply plus the axis the person expected, applied to the classifier (not to pair weights), measured on held-out dialogues.
6. A stronger classifier is worth trying, but the flat curves suggest the ambiguity of one-line inputs would still limit it.

## 12. Reproduce everything
| what | command | expected |
|---|---|---|
| unit tests of the bank and the bot | `python -m unittest tests.test_bankbot` | 17 OK |
| evaluation and guards | `python tests/lucia_bank_eval.py` ; `python -m unittest tests.lucia_bank_eval` | TEST 17 / 1 / 1 / 1 ; 1 / 1 / 6 ; 5 silent / 1 answered ; 4 OK |
| generic scenes | `python -m unittest tests.lucia_generic_scenes_eval` | 3 tests, OK (expected failures=1) |
| README suites | `python tests/test_tables.py` ; `python tests/test_arithmetic.py` ; `python -m unittest tests.test_train tests.test_v2_integration` | 62/62 ; 14/14 ; 41 OK |
| diagnosis (section 7) | `python experiments/lucia_diagnose.py` | the tables of 7.1–7.5 (about 10 minutes) |
| lcm_forge checks (section 8) | `python experiments/forge_checks.py --forge PATH/lcm_forge_project` | the table of section 8 |

## 13. Notes on the figures
- Prompt -> axis accuracy is quoted as the mean of 6 random splits (53%). A single 80-item split can read 60% or more.
- The pooled light result is quoted two ways: exact bank reply (29/40) and exact-or-same-axis (31/40).
- Everything shipped with the character is in English.
