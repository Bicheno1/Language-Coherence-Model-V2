"""Reading scripts with speaker names ('Lucia: Hello, are you okay?') and extracting input/reply pairs."""
import re
from dataclasses import dataclass
from typing import List, Optional

_SPEAKER = re.compile(r"^\s*([^\W\d_][^:\n]{0,39}?)\s*:\s*(\S.*)$")
_DIRECTION = re.compile(r"\([^)]*\)|\[[^\]]*\]")
_BREAK = re.compile(r"^\s*(-{3,}|={3,}|\*{3,}|SCENE\b.*|INT\.\s.*|EXT\.\s.*)\s*$", re.I)


@dataclass
class Turn:
    speaker: str
    text: str
    scene: int
    index: int


@dataclass
class Pair:
    prompt: Turn          # what the other person said
    reply: Turn           # what the character answers
    context: Optional[Turn]  # turn before the prompt (same scene), for the tone


def _clean(s: str, strip_directions: bool) -> str:
    if strip_directions:
        s = _DIRECTION.sub(" ", s)
    return re.sub(r"\s+", " ", s).strip()


def parse_script(text: str, strip_directions: bool = True, unlabeled: str = "auto") -> List[Turn]:
    """A line 'Name: text' opens a turn. A line without a name continues the previous turn.
        Two consecutive blank lines, '---', 'SCENE ...', 'INT.'/'EXT.' open a new scene."""
    turns: List[Turn] = []
    scene, pending_break, blanks = 0, False, 0
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            blanks += 1
            if blanks >= 2:
                pending_break = True
            continue
        blanks = 0
        if _BREAK.match(line):
            pending_break = True
            continue
        m = _SPEAKER.match(line)
        if m and not m.group(2).startswith("//"):
            if pending_break and turns:
                scene += 1
            pending_break = False
            body = _clean(m.group(2), strip_directions)
            if body:
                turns.append(Turn(re.sub(r"\s+", " ", m.group(1)).strip(), body, scene, len(turns)))
        elif turns and not pending_break:
            body = _clean(line, strip_directions)
            cont = unlabeled == "append" or (unlabeled == "auto" and (raw[:1].isspace() or not re.search(r"[.!?…\"”»]$", turns[-1].text)))
            if body and cont:
                turns[-1].text += " " + body
    return turns


def make_pairs(turns: List[Turn]) -> List[Pair]:
    """Each change of speaker within a scene gives a pair (what the other said -> the reply)."""
    out = []
    for i in range(len(turns) - 1):
        a, b = turns[i], turns[i + 1]
        if a.scene == b.scene and a.speaker.casefold() != b.speaker.casefold():
            ctx = turns[i - 1] if i > 0 and turns[i - 1].scene == a.scene else None
            out.append(Pair(a, b, ctx))
    return out


# ======================= recognizing that the text is a conversation =======================
class NotDialogueError(ValueError):
    def __init__(self, report):
        self.report = report
        super().__init__("The text does not look like a conversation: " + report["reason"] + "  (options: --format named|chat|dash, or --monologue NAME for text by a single person)")


_TS = r"\[?\d{1,2}[/.\-]\d{1,2}[/.\-]\d{2,4},?\s+\d{1,2}:\d{2}(?::\d{2})?(?:\s?[APap]\.?\s?[Mm]\.?)?\]?"
_CHAT = re.compile(r"^" + _TS + r"\s*[-–]?\s*([^:\n]{1,40}?):\s*(\S.*)$")
_TS_START = re.compile(r"^" + _TS)
_DASH = re.compile(r"^\s*[—–\-]\s*(\S.*)$")


def detect_format(text: str) -> dict:
    """Looks at the lines and decides: named ('Name: text'), chat (WhatsApp: '12/03/23, 10:15 - Name: text'),
        dash (literary dialogue with a dash and no names) or plain (running text). Returns the format, the confidence and the reason."""
    lines = [l.strip() for l in text.splitlines() if l.strip() and not _BREAK.match(l.strip())]
    n = len(lines)
    if n == 0:
        return dict(format="plain", confidence=0.0, speakers=[], alternation=0.0, reason="the text is empty")
    chat = [m.group(1).strip() for l in lines if (m := _CHAT.match(l))]
    dash = [l for l in lines if _DASH.match(l) and not _TS_START.match(l)]
    labels = [m.group(1).strip() for l in lines if not _TS_START.match(l) and (m := _SPEAKER.match(l)) and not m.group(2).startswith("//")]
    freq = {}
    for x in labels: freq[x.casefold()] = freq.get(x.casefold(), 0) + 1
    # if almost every line has a label, all of them are speakers; otherwise a label that appears only once is not one (Note:, Source:...)
    real = labels if len(labels) / n >= 0.8 else [x for x in labels if freq[x.casefold()] >= 2]
    def alt(seq): return sum(1 for a, b in zip(seq, seq[1:]) if a.casefold() != b.casefold()) / max(len(seq) - 1, 1)
    spk = sorted({x.casefold() for x in real})
    if len(chat) / n >= 0.5 and len(set(c.casefold() for c in chat)) >= 2:
        return dict(format="chat", confidence=round(len(chat) / n, 2), speakers=sorted(set(chat)), alternation=round(alt(chat), 2), reason="lines with date and time, WhatsApp style")
    if len(dash) / n >= 0.5:
        return dict(format="dash", confidence=round(len(dash) / n, 2), speakers=["A", "B"], alternation=1.0, reason="lines with a dialogue dash and no names (speakers alternate A/B)")
    if real and len(real) / n >= 0.5 and len(spk) >= 2 and alt(real) >= 0.3:
        return dict(format="named", confidence=round(len(real) / n, 2), speakers=sorted(set(real)), alternation=round(alt(real), 2), reason="lines 'Name: text' with at least 2 alternating speakers")
    why = (f"only {len(real)} of {n} lines carry a speaker name" if len(real) / n < 0.5 else
           "there is a single speaker" if len(spk) < 2 else "the names hardly alternate")
    return dict(format="plain", confidence=round(len(real) / n, 2), speakers=sorted(set(real)), alternation=round(alt(real), 2) if real else 0.0, reason=why)


def parse_chat(text: str, strip_directions: bool = True) -> List[Turn]:
    turns: List[Turn] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line: continue
        m = _CHAT.match(line)
        if m:
            body = _clean(m.group(2), strip_directions)
            if body: turns.append(Turn(re.sub(r"\s+", " ", m.group(1)).strip(), body, 0, len(turns)))
        elif _TS_START.match(line):
            continue                                               # system message (encryption, group changes...)
        elif turns:
            turns[-1].text += " " + _clean(line, strip_directions)  # multi-line message
    return turns


def parse_dash(text: str, strip_directions: bool = True) -> List[Turn]:
    """Dialogue with a dash and no names: each dashed line is a turn; speakers alternate A/B (their names are unknown)."""
    turns: List[Turn] = []; scene = 0; blanks = 0
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            blanks += 1
            if blanks >= 2 and turns: scene += 1 if turns[-1].scene == scene else 0
            continue
        blanks = 0
        m = _DASH.match(line)
        if m:
            body = _clean(m.group(1), strip_directions)
            if body: turns.append(Turn("A" if len(turns) % 2 == 0 else "B", body, scene, len(turns)))
    return turns


def parse_monologue(text: str, name: str) -> List[Turn]:
    """Text by ONE person: each non-empty line is a turn of theirs. There are no pairs (nobody answers), but it feeds their personal layer."""
    out, scene, blanks = [], 0, 0
    for raw in text.splitlines():
        line = _clean(raw, True)
        if not line:
            blanks += 1
            if blanks >= 2: scene += 1
            continue
        blanks = 0; out.append(Turn(name, line, scene, len(out)))
    return out


def parse_auto(text: str, fmt: str = "auto", monologue: Optional[str] = None, unlabeled: str = "auto"):
    """Returns (turns, report). If the text does not look like a conversation it raises NotDialogueError with the reason."""
    if monologue:
        return parse_monologue(text, monologue), dict(format="monologue", confidence=1.0, speakers=[monologue], alternation=0.0, reason="text by a single person (forced)")
    rep = detect_format(text)
    use = rep["format"] if fmt == "auto" else fmt
    if use == "plain":
        raise NotDialogueError(rep)
    parser = {"named": parse_script, "chat": parse_chat, "dash": parse_dash}[use]
    rep = dict(rep, format=use, forced=fmt != "auto")
    turns = parser(text, unlabeled=unlabeled) if use == "named" else parser(text)
    if not turns:
        raise NotDialogueError(dict(rep, reason=f"the format '{use}' found no turns"))
    return turns, rep
