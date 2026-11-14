"""Table of numbers per character. Every letter, including n with tilde (ñ), every accent and every punctuation mark has a FIXED number.
Capital letters have no number of their own: case is a separate feature (case_of). Words are sequences of these numbers (their 'cipher'),
and the typo search runs on those ciphers, not on text.
  1-26  a-z      27 ñ      28-33 á é í ó ú ü      40-49 digits 0-9
  50 space  51 .  52 ,  53 ?  54 !  55 ;  56 :  57 '  58 -  59 ¿  60 ¡  61 "  62 «  63 »  64 —  65 …  66 (  67 )  68 –
  200+  any other character: it gets the next free number and is saved in the database."""
import unicodedata

_ACCENT = {"á": "a", "é": "e", "í": "i", "ó": "o", "ú": "u", "ü": "u"}   # ñ is not folded: año and ano are different words


class Charset:
    def __init__(self, dynamic=None):
        self.table = {}
        for i, c in enumerate("abcdefghijklmnopqrstuvwxyz", 1): self.table[c] = i
        for i, c in enumerate("ñáéíóúü", 27): self.table[c] = i
        for i, c in enumerate("0123456789", 40): self.table[c] = i
        for i, c in enumerate(" .,?!;:'-¿¡\"«»—…()–", 50): self.table[c] = i
        self.next = 200
        for c, n in (dynamic or {}).items():
            self.table[c] = n; self.next = max(self.next, n + 1)

    def num(self, ch):
        ch = ch.lower()
        if ch not in self.table:
            self.table[ch] = self.next; self.next += 1
        return self.table[ch]

    def encode(self, s):
        return tuple(self.num(c) for c in unicodedata.normalize("NFC", s))

    def decode(self, codes):
        inv = {v: k for k, v in self.table.items()}
        return "".join(inv.get(c, "?") for c in codes)

    def dynamic(self):
        return {c: n for c, n in self.table.items() if n >= 200}

    @staticmethod
    def fold(word):
        """Strip accents (not ñ): tells whether two words differ ONLY in an accent (si/sí, tu/tú)."""
        return "".join(_ACCENT.get(c, c) for c in unicodedata.normalize("NFC", word.lower()))

    @staticmethod
    def case_of(tok):
        if tok.isupper() and len(tok) > 1: return "UPPER"
        if tok[:1].isupper() and tok[1:].islower(): return "Title"
        if tok.islower(): return "lower"
        return "mixed"
