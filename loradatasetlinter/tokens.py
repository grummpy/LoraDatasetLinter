"""CLIP token estimate that does not download a tokenizer.

SD1.x CLIP truncates a caption at 75 tokens (77 with the special tokens).
This estimate splits words and punctuation, then grows a long word the way a
small BPE piece would. It is an estimate, not the official CLIP byte-pair count.
"""

from __future__ import annotations

import re

# ``\w`` includes underscores.  CLIP's byte-pair tokenizer treats an underscore
# as punctuation, so keeping it inside a word can substantially undercount tags
# such as ``red_hair``.  Keep apostrophes inside words, but emit underscores as
# their own approximate piece.
_PIECE = re.compile(r"[^\W_]+(?:'[^\W_]+)?|_|[^\s\w]", re.UNICODE)


def estimate_clip_tokens(text: str) -> int:
    if not text or not text.strip():
        return 0
    total = 0
    for piece in _PIECE.findall(text):
        if piece.isalnum() or "'" in piece:
            total += max(1, (len(piece) + 3) // 4) if len(piece) > 8 else 1
        else:
            total += 1
    return total
