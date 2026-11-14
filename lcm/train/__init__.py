"""lcm.train: the training module of the LCM v2. It reads conversations (scripts with speaker names) and builds the databases that feed the LCM:
the common layer (average of language), one layer per character, the axis values of every word and the repertoire of replies."""
__version__ = "0.1"
from .script import parse_script, make_pairs
from .lexicon import Lexicon
from .store import Store
