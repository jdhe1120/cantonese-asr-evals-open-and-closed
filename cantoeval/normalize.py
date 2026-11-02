"""Text normalisation for scoring.

NFKC first: PDF text layers use Kangxi radical code points (⼆ ⼗ ⽉) that
render identically to the normal graphs; NFKC folds them. Then optional
interchangeable-graph folding (rules/bench_conventions.toml), then OpenCC t2s,
then drop punctuation, symbols and whitespace.

The traditional view (trad=True) skips t2s, so a simplified character counts as an error; instead Hong Kong and
Taiwan variant spellings of the same traditional character (裏/裡, 着/著, 衞/衛) are folded to one form, so neither
standard is penalised. It is only scored on clips whose references are traditional (runner.traditional_clips).
"""
import re
import tomllib
import unicodedata
from functools import lru_cache
from pathlib import Path

import opencc

RULES = Path(__file__).resolve().parent.parent / "rules" / "bench_conventions.toml"


@lru_cache(maxsize=1)
def _t2s() -> opencc.OpenCC:
    return opencc.OpenCC("t2s")


@lru_cache(maxsize=1)
def _equiv_table() -> dict[int, str]:
    with RULES.open("rb") as f:
        classes = tomllib.load(f)["equivalent"]
    table: dict[int, str] = {}
    for c in classes:
        graphs = c["graphs"]
        # The fold runs before t2s, and some text is already simplified (SenseVoice
        # writes 吗/啰), so each graph's simplified form joins its class too.
        graphs += "".join(dict.fromkeys(
            s for s in map(_t2s().convert, graphs) if len(s) == 1 and s not in graphs))
        for g in graphs[1:]:
            table[ord(g)] = graphs[0]
    return table


@lru_cache(maxsize=1)
def _variants() -> dict[int, str]:
    """Single-character HK and Taiwan variants -> OpenCC's standard traditional form (its *VariantsRev tables)."""
    table: dict[int, str] = {}
    for name in ("HKVariantsRev.txt", "TWVariantsRev.txt"):
        for line in (Path(opencc.__file__).parent / "dictionary" / name).read_text(encoding="utf-8").splitlines():
            if "\t" in line and not line.startswith("#"):
                k, v = line.split("\t")
                if len(k) == 1 and len(v.split()[0]) == 1 and k != v.split()[0]:
                    table[ord(k)] = v.split()[0]
    return table


# Standard in Hong Kong although OpenCC's simplified->traditional table converts them (晒 is the Cantonese 晒 particle,
# not 曬); traditional-writing models and the references use them.
HK_ALSO = frozenset("晒吓群床峰雇")


@lru_cache(maxsize=1)
def simplified_only() -> frozenset[str]:
    """Characters that are never traditional: OpenCC maps them to another form, no simplified character maps to them
    (们 这 说; not 台 只 干 后, which OpenCC may convert but are valid traditional too), and Hong Kong doesn't use them."""
    traditional, simplified = set(), set()
    for line in (Path(opencc.__file__).parent / "dictionary" / "STCharacters.txt").read_text(encoding="utf-8").splitlines():
        if "\t" in line and not line.startswith("#"):
            k, v = line.split("\t")
            traditional.update(v.split())
            if k not in v.split():
                simplified.add(k)
    hk = set()  # Hong Kong standard forms that OpenCC also lists as simplified (揾 温 秘 悦 税 户...)
    for line in (Path(opencc.__file__).parent / "dictionary" / "HKVariants.txt").read_text(encoding="utf-8").splitlines():
        if "\t" in line and not line.startswith("#"):
            hk.update(x for x in line.split("\t")[1].split() if len(x) == 1)
    return frozenset(simplified - traditional - hk - HK_ALSO)


def is_simplified(text: str) -> bool:
    """True when the text contains a simplified-only character."""
    return any(ch in simplified_only() for ch in text)


# CJK Radicals Supplement look-alikes (U+2E80-2EF3). Unlike the Kangxi radicals (U+2F00-2FD5), NFKC leaves
# these alone, and the PDFs' text layers use them, so a reference could contain ⻝ where a model writes 食.
# Only the ones found in the references so far; `python -m cantoeval check` fails if another turns up.
_RADICALS = str.maketrans({"⻝": "食", "⻑": "長", "⺠": "民", "⻘": "青", "⻣": "骨"})


def stray_radicals(text: str) -> set[str]:
    """Radical-block characters left after normalisation (should always be empty)."""
    text = unicodedata.normalize("NFKC", text).translate(_RADICALS)
    return {c for c in text if 0x2E80 <= ord(c) <= 0x2FDF}


_DIGITS = "零一二三四五六七八九"
_UNITS = ["", "十", "百", "千"]


def _group(n: int) -> str:
    """0 < n < 10000 as Chinese: 12 -> 十二, 105 -> 一百零五, 2016 -> 二千零一十六."""
    out, zero = "", False
    for i, ch in enumerate(f"{n:04d}"):
        d, unit = int(ch), _UNITS[3 - i]
        if d == 0:
            zero = bool(out)
        else:
            if zero:
                out += "零"
            zero = False
            out += ("" if (d == 1 and unit == "十" and not out) else _DIGITS[d]) + unit
    return out


def _cardinal(n: int) -> str:
    if n == 0:
        return "零"
    if n >= 10**8:
        return "".join(_DIGITS[int(c)] for c in str(n))  # too big to read as a quantity
    hi, lo = divmod(n, 10000)
    out = (_group(hi) + "萬") if hi else ""
    if lo:
        out += ("零" if hi and lo < 1000 else "") + _group(lo)
    return out


def _number(m: re.Match) -> str:
    whole, frac, tail = m.group("int"), m.group("frac"), m.group("tail")
    spoken_digits = lambda d: "".join(_DIGITS[int(c)] for c in d)  # noqa: E731
    # Years, long runs (phone/ID numbers) and leading zeros are read digit by digit.
    if tail == "年" or len(whole) >= 7 or (len(whole) > 1 and whole[0] == "0"):
        body = spoken_digits(whole)
    else:
        body = _cardinal(int(whole))
    return body + ("點" + spoken_digits(frac) if frac else "") + (tail or "")


def numerals_to_chinese(text: str) -> str:
    """Arabic digits -> Chinese numerals (2016年 -> 二零一六年, 12月 -> 十二月).
    Not applied by the benchmark scorers this follows; see the numerals toggle in the UI."""
    return re.sub(r"(?P<int>\d+)(?:\.(?P<frac>\d+))?(?P<tail>年)?", _number, text)


def normalize(text: str, equiv: bool = False, numerals: bool = False, trad: bool = False) -> str:
    text = unicodedata.normalize("NFKC", text).translate(_RADICALS)
    if numerals:
        text = numerals_to_chinese(text)
    if equiv:
        text = text.translate(_equiv_table())
    text = text.translate(_variants()) if trad else _t2s().convert(text)
    return "".join(ch for ch in text if unicodedata.category(ch)[0] not in "PSZC")
