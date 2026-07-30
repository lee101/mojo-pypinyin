"""Chinese to pinyin conversion accelerated by Mojo."""

from .api import (
    PinyinNotFoundException,
    Style,
    lazy_pinyin,
    load_phrases_dict,
    load_single_dict,
    pinyin,
    slug,
)

__version__ = "0.1.0"

NORMAL = STYLE_NORMAL = Style.NORMAL
TONE = STYLE_TONE = Style.TONE
TONE2 = STYLE_TONE2 = Style.TONE2
TONE3 = STYLE_TONE3 = Style.TONE3
INITIALS = STYLE_INITIALS = Style.INITIALS
FIRST_LETTER = STYLE_FIRST_LETTER = Style.FIRST_LETTER
FINALS = STYLE_FINALS = Style.FINALS
FINALS_TONE = STYLE_FINALS_TONE = Style.FINALS_TONE
FINALS_TONE2 = STYLE_FINALS_TONE2 = Style.FINALS_TONE2
FINALS_TONE3 = STYLE_FINALS_TONE3 = Style.FINALS_TONE3
BOPOMOFO = STYLE_BOPOMOFO = Style.BOPOMOFO
BOPOMOFO_FIRST = STYLE_BOPOMOFO_FIRST = Style.BOPOMOFO_FIRST
CYRILLIC = STYLE_CYRILLIC = Style.CYRILLIC
CYRILLIC_FIRST = STYLE_CYRILLIC_FIRST = Style.CYRILLIC_FIRST

__all__ = [
    "pinyin",
    "lazy_pinyin",
    "slug",
    "load_single_dict",
    "load_phrases_dict",
    "PinyinNotFoundException",
    "Style",
    "NORMAL",
    "TONE",
    "TONE2",
    "TONE3",
    "INITIALS",
    "FIRST_LETTER",
    "FINALS",
    "FINALS_TONE",
    "FINALS_TONE2",
    "FINALS_TONE3",
]
