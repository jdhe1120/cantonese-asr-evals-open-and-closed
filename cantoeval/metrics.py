"""Character error rate and per-clip diffs."""
import difflib

from .normalize import normalize


def _levenshtein(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def score(reference: str, hypothesis: str, equiv: bool = False, numerals: bool = False, trad: bool = False) -> dict:
    ref, hyp = normalize(reference, equiv, numerals, trad), normalize(hypothesis, equiv, numerals, trad)
    edits = _levenshtein(ref, hyp)
    return {"edits": edits, "refLen": len(ref), "cer": edits / len(ref) if ref else 0.0}


def diff(reference: str, hypothesis: str, equiv: bool = False, numerals: bool = False,
         trad: bool = False) -> list[list[str]]:
    """Segments of [kind, refText, hypText]; kind is eq | sub | del | ins."""
    ref, hyp = normalize(reference, equiv, numerals, trad), normalize(hypothesis, equiv, numerals, trad)
    kinds = {"equal": "eq", "replace": "sub", "delete": "del", "insert": "ins"}
    return [[kinds[op], ref[i1:i2], hyp[j1:j2]]
            for op, i1, i2, j1, j2 in difflib.SequenceMatcher(None, ref, hyp, autojunk=False).get_opcodes()]


def summarize(scores: list[dict]) -> dict:
    """Mean per-utterance CER and corpus CER over a list of score() dicts."""
    if not scores:
        return {"meanCer": None, "corpusCer": None, "n": 0}
    total_ref = sum(s["refLen"] for s in scores)
    return {
        "meanCer": sum(s["cer"] for s in scores) / len(scores),
        "corpusCer": sum(s["edits"] for s in scores) / total_ref if total_ref else 0.0,
        "n": len(scores),
    }
