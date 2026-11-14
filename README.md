# LCM v2 — Language Coherence Model

The Language Coherence Model (LCM) is a research prototype of a language module for the Cognitive Coherence Model (CCM). Words, replies and speakers are placed on four axes — **North** (V, *we*), **South** (I, *it* / *they*), **East** (Lv, *I*) and **West** (Gv, *you*) — and a reply is a path through that plane, guided and measured by its geometry. There is no neural network: every number can be read and corrected.

**Version 2 changes the base of the model.** Instead of a statistical map of humankind, which needs data and computing that this project does not have, the base is the **pattern of an individual**: the system learns from conversations how one person or character speaks and answers, and it answers only from what that individual has said, abstaining when the situation is new. It runs with **the Python standard library only**.

> **Status: research prototype (v2.2.0-dev).** It does not produce open free text. The hypothesis behind it is tested in part and not confirmed. The grammatical analysis is English only. Read [Known limitations](#known-limitations) before relying on it.

## What is in v2

- **No runtime dependencies.** The analyzer that used spaCy is replaced by a deterministic grammar (`lcm/lite_nlp.py`), the inflection library by rules and irregular-form tables (`lcm/inflect.py`), and the WordNet calls at runtime by data baked once into `data/wordnet_baked.json`.
- **A training module** (`lcm/train/`): reads conversations (scripts, chat exports, dialogues), keeps a common layer and one layer per speaker, and estimates the four axis values of every word.
- **Reply as a character** (`lcm/train/character.py`): she answers from her own repertoire and says nothing when nothing resembles the input.
- **Lucía** (`docs/LUCIA.md`): Lucía v1 is the first, small initial character (the demo script plus 24 scenes); Lucía v2 is the incremented one, a bank of 400 emotional dialogues (100 per axis) that I wrote to see the limits of training with a somewhat larger base.
- **`overlay` and `compare`**: write what was learned from text into the tables, and check learned axes against the hand-made ones.
- **Arithmetic by relations** (`docs/ARITHMETIC.md`): operator precedence is derived from how operations relate to each other, not written in the code.
- **Examples of use** in `tests/`: an assistant simulation (request → action, with abstention) and a Lucía comparison. Both use common phrases.

## Install and first run

Python 3.8 or later. Nothing to install to run it. From the repository folder:

```
python -m lcm stats                          # the tables: 9,263 nodes
python -m lcm reply "what is 12 plus 30"     # 12 plus 30 is 42.
python -m lcm recompose "tell me a story about a cat" --trace --seed 7
```

Stories and generic-species replies read WordNet data that is baked once. Do this one time:

```
pip install nltk
python tools/setup_nltk.py
python tools/bake_wordnet.py                 # writes data/wordnet_baked.json (about a minute)
```

`nltk` and `openpyxl` are needed only to rebuild data files.

## Train it on a conversation

```
python -m lcm train detect tests/train_examples/demo_script.txt       # is it a conversation? which format and language?
python -m lcm train ingest tests/train_examples/demo_script.txt --db forge_db.json
python -m lcm train profile Lucia --db forge_db.json                  # the character in numbers
python -m lcm character Lucia "Hello Lucia" --db forge_db.json --seed 3
python -m lcm overlay --db forge_db.json --out overlay_related.json   # learned relations -> the LCM tables
python -m lcm compare --db forge_db.json --min-n 5                    # learned axes vs. the tables
```

`python -m lcm train --help` lists the training commands (`detect`, `ingest`, `profile`, `word`, `reply`, `tables`, `alias`).

| Text | Example | What you get |
|---|---|---|
| Script with names | `Lucia: Hello, are you okay today?` | pairs between turns of different speakers |
| Chat export | `12/03/2024, 10:16 - Lucía: Hola, ¿estás bien?` | pairs |
| Dialogue with dashes | `—¿Vienes mañana al pueblo?` | pairs between two speakers |
| Prose | a paragraph | rejected as a conversation; with `--monologue NAME` it is the text of one person, with no pairs. A paragraph is one utterance, so for the axes put **one sentence per line** |

One base has one language (English or Spanish). A line without a speaker name continues the previous turn only if that turn has no final punctuation or the line is indented; otherwise it is ignored. `ingest --no-text` keeps counts only, which disables replies as a character.

**The axes** are learned by a classifier trained on labeled phrases (`lcm/train/data/axis_seed.json`, 180 English phrases; `axis_seed_es.json`, 100 Spanish; `axis_seed_lucia.json`, 100 per axis, made from Lucía’s bank). Each word accumulates the axis of the utterances where it appears and is normalized by lift; a center is given only with five or more occurrences. To train with your own phrases, replace the seed file or create the store from Python with `Store(axis_seed=path)`.

## Lucía

There are two Lucías and I keep them apart. **Lucía v1** is the first: the character of the demo script, extended with 24 scenes (`tests/lucia_test.py`). **Lucía v2** is the incremented one, which I wrote to see the limits of training with a somewhat larger base; everything below is about v2.

Lucía v2 answers from a bank of 400 dialogues, four axes of 100. She recalls the closest prompt of her bank, or answers by axis when she is unsure of the prompt, or stays silent. She never says a sentence that is not in her bank.

```
python -m lcm lucia --info                                   # what is loaded: 400 dialogues, 100 per axis
python -m lcm lucia --say "Could you ever forgive me, Lucia?"   # one answer, and how she decided
python -m lcm lucia --demo                                   # a short scripted conversation
python -m lcm lucia --chat                                   # talk to her (empty line to quit)
python -m lcm lucia --make-seed lcm/train/data/axis_seed_lucia.json
python -m lcm lucia --bank my_bank.txt --character Marina --say "..."   # another character, another bank
```

Options: `--bank PATH`, `--character NAME`, `--t-hi X` (similarity needed to recall, default 0.4) and `--margin X` (classifier margin needed to answer by axis, default 8.0). Her bank is `data/lucia/lucia_bank.txt`; she does not need the tables.

Measured on held-out inputs written after the bank: 17 of 20 close rewordings got her exact bank reply, and she stayed silent on 6 of 8 new situations. More data of the same kind did not help; the limit is that one line often does not decide the emotion. The protocol, the failures and the next steps are in [`docs/LUCIA.md`](docs/LUCIA.md).

## Tests

```
python tests/test_tables.py
python tests/test_arithmetic.py
python -m unittest tests.test_train tests.test_v2_integration tests.test_assistant tests.lucia_test
python -m unittest tests.test_bankbot tests.lucia_bank_eval tests.lucia_generic_scenes_eval      # Lucía's bank
python tests/lucia_bank_eval.py [--detail]   # thresholds chosen on a DEV set, result on a separate TEST set
python tests/assistant_test.py               # assistant simulation, simple method (does not use the LCM)
python tests/assistant_lcm_test.py           # the same, with the LCM training system as the engine
python tests/lucia_test.py                   # Lucía, demo script vs. 24 extra initial scenes
```

With Python 3.12.3, WordNet baked and the v2.2.0-dev package: `test_tables.py` 62 of 62, `test_arithmetic.py` 14 of 14, the training, integration, assistant and Lucía-demo tests 46 of 46, and the Lucía bank tests 24 (one expected failure, on purpose). Before baking WordNet, the tests that need it print `SKIP`. One table test uses `lemminflect` as a reference (`pip install lemminflect`); the runtime does not.

## Repository layout

```
lcm/                 the model: tables, geometry, walk, parse, narrate, reply, arithmetic, lite_nlp, inflect, wn_baked
lcm/train/           the training module: script, charset, lang, lexicon, analyzer, axes, store, reply, character, export, cli, and bank + bankbot (Lucía)
data/                LCM_tables.json (9,263 nodes), LCM_arithmetic.json, connector punctuation, Excel sheets for review, lucia/ (her bank)
tools/               bake_wordnet.py, build_tables.py and the data builders (build time only)
tests/               unit tests, assistant and Lucía examples, tests/train_examples/ (demo scripts)
docs/                ARITHMETIC.md, LUCIA.md, README_v0.5.md
experiments/         the measurements of the architecture paper on the Brown corpus
ai_corpus/, source_data/   experimental pipeline and source databases; the model does not depend on them
```

## Papers

- Aguilar Apolinario, P. A. (2026). *Cognitive Coherence Model (CCM), v1.*
- Aguilar Apolinario, P. A. (2026). *Language Coherence Model (LCM): Architecture Paper, v1.*
- Aguilar Apolinario, P. A. (2026). *Training of the Language Coherence Model (LCM), v1.1.*
- Aguilar Apolinario, P. A. (2026). *Language Coherence Model (LCM): Architecture Paper, v2.*

## Known limitations

- **`lite_nlp` equals spaCy only on what the tests encode** (about 63 distinct inputs). Free-form sentences are analyzed in a degraded way. The 98% of roles reported with spaCy has not been measured for v2.
- **The responder answers some forms only:** opinion, taste, state, story requests and arithmetic. Greetings, statements, exclamations and other questions get no reply. Questions without a question mark, and questions preceded by a greeting, are not recognized.
- A story theme that is not a node of the tables is ignored, and the protagonist is drawn from the seeds.
- **Training has been run only on demo scripts.** No real script or chat has been ingested. Typed relations exist only for English.
- **The axis classifier is weak outside its seed:** about 82–85% on the blind batch of common phrases, and 46% on Lucía’s 400 replies with the bank’s labels.
- Lucía has one exchange per dialogue and no memory of the conversation; she is silent on most new situations.
- The profile is unreversed: the state engine, the decision module and the reasoning module are outside this prototype, and theme selection and grammatical constructions are placeholders.

## Data, privacy and provenance

- The axis seeds are common phrases that I chose; I defined the axes and what each one means (`meta.source = first_version`); `axis_seed_lucia.json` comes from my own bank (`author_bank`). The evaluation sets and the example tests use common phrases, reviewed by me.
- Scripts and chats are texts of other people: use them only with permission. A character trained on a real person’s text reproduces how that person writes and needs that person’s consent.
- `data/LCM_tables.json` was built with WordNet and SentiWordNet. See `NOTICE`.

## License

[PolyForm Noncommercial License 1.0.0](LICENSE). Commercial use requires a separate license; see `NOTICE` for the contact.

## Citation

If you use this work, cite the architecture paper: Aguilar Apolinario, P. A. (2026). *Language Coherence Model (LCM): Architecture Paper, v2.* Independent research paper.

## Author

Pool Andres Aguilar Apolinario (Akimsa3) — Independent Researcher, Lima, Perú.
