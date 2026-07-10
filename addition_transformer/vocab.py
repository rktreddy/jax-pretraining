"""Character vocabulary for addition: digit 0=9, space, +, = and PAD"""

PAD_TOKEN = "<pad>"
CHARS = list[str]("0123456789 +=")
VOCAB = [PAD_TOKEN] + CHARS
CHAR_TO_ID = {c: i for i, c in enumerate[str](VOCAB)}
ID_TO_CHAR = {i: c for c, i in CHAR_TO_ID.items()}

PAD_ID = CHAR_TO_ID[PAD_TOKEN]
VOCAB_SIZE = len(VOCAB)

# Longest example: "999 + 999 = 1998" (16 chars). Pad to fixed length
MAX_SEQ_LEN = 20

# Longest sum: 999 + 999 = 1998 -> 4 answer digits
MAX_ANSWER_DIGITS = 4

def encode(text: str, max_len: int = MAX_SEQ_LEN) -> list[int]:
    """Encode a string to fixed-length token ids (right-padded with PAD)."""
    ids = [CHAR_TO_ID[c] for c in text]
    if len(ids) > max_len:
        raise ValueError(f"Sequence too long ({len(ids)} > {max_len}): {text!r}")
    return ids + [PAD_ID] * (max_len - len(ids))


def decode(ids: list[int] | tuple[int, ...]) -> str:
    """Decode token ids to string. stripping trailing PAD."""
    chars = []
    for i in ids:
        idx = int(i)
        if idx == PAD_ID:
            break
        chars.append(ID_TO_CHAR[idx])
    return "".join(chars)
