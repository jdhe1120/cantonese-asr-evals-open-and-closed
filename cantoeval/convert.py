"""Simplified -> Hong Kong traditional for models that write simplified (SenseVoice, stock Qwen), as cantocaptions-ai
converts Qwen output: OpenCC s2t, then its HK variant table with Cantonese fixes (rules/opencc_hk_cantocaptions.txt;
longest match first, so 只係 stays 只係 while 一只 becomes 一隻)."""
import functools
import re
from pathlib import Path

import opencc

TABLE = Path(__file__).resolve().parent.parent / "rules" / "opencc_hk_cantocaptions.txt"


@functools.cache
def _table() -> tuple[re.Pattern, dict[str, str]]:
    m = {}
    for line in TABLE.read_text(encoding="utf-8").splitlines():
        if "\t" in line and not line.startswith("#"):
            k, v = line.split("\t")
            m[k] = v.split()[0]
    return re.compile("|".join(map(re.escape, sorted(m, key=len, reverse=True)))), m


@functools.cache
def _s2t() -> opencc.OpenCC:
    return opencc.OpenCC("s2t")


def to_hk_traditional(text: str) -> str:
    pat, m = _table()
    return pat.sub(lambda x: m[x.group(0)], _s2t().convert(text))
