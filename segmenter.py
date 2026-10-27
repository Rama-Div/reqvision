"""ReqVision - split a story-style SRS (no punctuation) into one requirement per sentence.

Text that already has sentence terminators / new lines is left untouched.  Only a chunk that
contains two or more modal verbs (shall / must / can / ...) and no separator is split, by
finding where the next subject ("the system", "admin", ...) starts before each modal verb.
"""
from __future__ import annotations

import re

# Modal phrases as token tuples (longest first so "shall not" wins over "shall").
_MODALS = sorted(
    [
        ("shall", "not"), ("must", "not"), ("should", "not"), ("will", "not"),
        ("may", "not"), ("can", "not"), ("cannot",), ("can't",), ("won't",),
        ("mustn't",), ("shouldn't",),
        ("must", "be", "able", "to"),
        ("is", "able", "to"), ("are", "able", "to"),
        ("is", "allowed", "to"), ("are", "allowed", "to"),
        ("has", "to"), ("have", "to"), ("needs", "to"), ("need", "to"),
        ("shall",), ("must",), ("should",), ("will",), ("can",), ("could",),
        ("may",), ("might",),
    ],
    key=len, reverse=True,
)

_CONNECTORS = {
    "and", "or", "but", "then", "also", "next", "finally", "afterwards",
    "additionally", "moreover", "furthermore", "besides", "meanwhile", "so", "plus",
}
_NO_SUBJECT = _CONNECTORS | {"to", "that", "which", "who", "not", "be", "is", "are", "if", "when"}
_DETERMINERS = {"the", "a", "an", "each", "every", "all", "any", "this", "these",
                "those", "our", "their", "its", "some", "no"}
_COND = {"if", "when", "whenever", "once", "unless"}

# A sentence terminator followed by whitespace, or a new line. "99.9%" is not a terminator.
_SEP = re.compile(r"((?<=[.!?;؟؛])\s+|\n+)")


def _clean(tok: str) -> str:
    return re.sub(r"^[^\w']+|[^\w']+$", "", tok.lower())


def _find_modals(clean: list[str]) -> list[tuple[int, int]]:
    found: list[tuple[int, int]] = []
    i, n = 0, len(clean)
    while i < n:
        for m in _MODALS:
            if tuple(clean[i:i + len(m)]) == m:
                found.append((i, i + len(m)))
                i += len(m)
                break
        else:
            i += 1
    return found


def _subject_start(clean: list[str], s: int, lo: int) -> int | None:
    """Index where the subject of the modal at `s` begins, or None if it shares the previous
    subject ("... can login and can logout"). `lo` = first token after the previous modal;
    at least one object token must stay with the previous sentence."""
    idx = s - 1
    if idx < lo + 1 or clean[idx] in _NO_SUBJECT:
        return None
    for k in range(1, 4):
        j = s - k
        if j < lo + 1:
            break
        if clean[j] in _DETERMINERS:
            return j
    return idx  # bare subject: "admin can ...", "users can ..."


def _move_for_condition(toks: list[str], clean: list[str], start: int, lo: int) -> int:
    """'... and if X the system shall ...' -> the condition belongs to the NEXT requirement
    when it follows a connector or a comma; otherwise it stays with the previous one."""
    for c in range(lo + 1, start):
        if clean[c] in _COND:
            prev = toks[c - 1]
            if _clean(prev) in _CONNECTORS or prev.endswith((",", ";")):
                return c
            break
    return start


def _split_chunk(chunk: str) -> list[str]:
    toks = chunk.split()
    clean = [_clean(t) for t in toks]
    modals = _find_modals(clean)
    if len(modals) < 2:
        return [chunk.strip()]
    cuts: list[int] = []
    prev_end = modals[0][1]
    for s, e in modals[1:]:
        start = _subject_start(clean, s, prev_end)
        if start is not None:
            cuts.append(_move_for_condition(toks, clean, start, prev_end))
        prev_end = e
    if not cuts:
        return [chunk.strip()]
    bounds = [0] + sorted(set(cuts)) + [len(toks)]
    out: list[str] = []
    for a, b in zip(bounds, bounds[1:]):
        seg = toks[a:b]
        while seg:  # drop dangling connectors at the end ("... login and" | "... then")
            if len(seg) >= 2 and (_clean(seg[-2]), _clean(seg[-1])) == ("after", "that"):
                seg = seg[:-2]
            elif _clean(seg[-1]) in _CONNECTORS:
                seg.pop()
            else:
                break
        if a > 0:
            while seg and _clean(seg[0]) in _CONNECTORS:
                seg.pop(0)
        if seg:
            out.append(" ".join(seg).rstrip(",;:"))
    return out or [chunk.strip()]


def split_story(text: str) -> str:
    """Return `text` with one requirement per sentence. Well-punctuated text is unchanged."""
    parts = _SEP.split(text)
    for i in range(0, len(parts), 2):
        pieces = _split_chunk(parts[i])
        if len(pieces) > 1:
            parts[i] = ". ".join(pieces)
    return "".join(parts)
