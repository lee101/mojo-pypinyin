"""pypinyin-compatible public conversion API."""

from __future__ import annotations

from enum import IntEnum, unique
from functools import lru_cache
from typing import Iterable

import numpy as np

from . import _data as data
from ._lib import lib


@unique
class Style(IntEnum):
    NORMAL = 0
    TONE = 1
    TONE2 = 2
    TONE3 = 8
    INITIALS = 3
    FIRST_LETTER = 4
    FINALS = 5
    FINALS_TONE = 6
    FINALS_TONE2 = 7
    FINALS_TONE3 = 9
    BOPOMOFO = 10
    BOPOMOFO_FIRST = 11
    CYRILLIC = 12
    CYRILLIC_FIRST = 13
    WADEGILES = 14
    GWOYEU = 15
    BRAILLE_MAINLAND = 16
    BRAILLE_MAINLAND_TONE = 17


class PinyinNotFoundException(Exception):
    pass


_CUSTOM_SINGLE: dict[int, tuple[str, ...]] = {}
_CUSTOM_PHRASES: dict[str, tuple[tuple[str, ...], ...]] = {}


def _style_row(style: int, strict: bool, v_to_u: bool, neutral_five: bool) -> int:
    try:
        position = data.STYLE_POSITION[int(style)]
    except KeyError as exc:
        raise NotImplementedError(
            f"Style {int(style)} is not covered by mojopypinyin"
        ) from exc
    flags = int(bool(v_to_u)) * 2 + int(bool(neutral_five))
    return flags * 20 + int(bool(strict)) * 10 + position


@lru_cache(maxsize=80)
def _rendered_syllables(
    style: int, strict: bool, v_to_u: bool, neutral_five: bool
) -> tuple[str, ...]:
    row = data.STYLE_TOKENS[_style_row(style, strict, v_to_u, neutral_five)]
    return tuple(data.TOKENS[int(token)] for token in row)


@lru_cache(maxsize=80)
def _rendered_first_entries(
    style: int, strict: bool, v_to_u: bool, neutral_five: bool
) -> tuple[str, ...]:
    syllables = _rendered_syllables(style, strict, v_to_u, neutral_five)
    return tuple(syllables[entry[0]] for entry in data.ENTRIES)


def _kernel(text: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    codepoints = np.frombuffer(text.encode("utf-32-le"), dtype="<u4")
    count = codepoints.size
    entries = np.empty(count, dtype=np.int32)
    spans = np.empty(count, dtype=np.int32)
    if count:
        if codepoints.dtype != np.dtype(np.uint32) or not codepoints.flags.c_contiguous:
            raise RuntimeError("UTF-32 input buffer has an incompatible native layout")
        lib().mpy_convert(
            int(codepoints.ctypes.data),
            int(count),
            data.addr(data.DENSE),
            data.DENSE_START,
            int(data.DENSE.size),
            data.addr(data.EXTRA_CODES),
            data.addr(data.EXTRA_ENTRIES),
            int(data.EXTRA_CODES.size),
            data.addr(data.NODE_FIRST),
            data.addr(data.NODE_COUNT),
            data.addr(data.NODE_TERMINAL),
            data.addr(data.EDGE_CODES),
            data.addr(data.EDGE_NEXT),
            data.addr(data.PHRASE_OFFSETS),
            data.addr(data.PHRASE_ENTRIES),
            int(entries.ctypes.data),
            int(spans.ctypes.data),
        )
    return codepoints, entries, spans


def _is_han(codepoint: int) -> bool:
    return (
        codepoint == 0x3007
        or 0xE815 <= codepoint <= 0xE864
        or codepoint == 0xFA18
        or 0x3400 <= codepoint <= 0x4DBF
        or 0x4E00 <= codepoint <= 0x9FFF
        or 0xF900 <= codepoint <= 0xFAFF
        or 0x20000 <= codepoint <= 0x2A6DF
        or 0x2A703 <= codepoint <= 0x2B73F
        or 0x2B740 <= codepoint <= 0x2B81D
        or 0x2B825 <= codepoint <= 0x2BF6E
        or 0x2C029 <= codepoint <= 0x2CE93
        or codepoint == 0x2D016
        or 0x2D11B <= codepoint <= 0x2EBD9
        or 0x2F80A <= codepoint <= 0x2FA1F
        or 0x30000 <= codepoint <= 0x32389
    )


_TONE_MARKS = {
    "ā": ("a", "1"), "á": ("a", "2"), "ǎ": ("a", "3"), "à": ("a", "4"),
    "ē": ("e", "1"), "é": ("e", "2"), "ě": ("e", "3"), "è": ("e", "4"),
    "ī": ("i", "1"), "í": ("i", "2"), "ǐ": ("i", "3"), "ì": ("i", "4"),
    "ō": ("o", "1"), "ó": ("o", "2"), "ǒ": ("o", "3"), "ò": ("o", "4"),
    "ū": ("u", "1"), "ú": ("u", "2"), "ǔ": ("u", "3"), "ù": ("u", "4"),
    "ǖ": ("ü", "1"), "ǘ": ("ü", "2"), "ǚ": ("ü", "3"), "ǜ": ("ü", "4"),
    "ḿ": ("m", "2"), "ń": ("n", "2"), "ň": ("n", "3"), "ǹ": ("n", "4"),
}
_INITIALS = (
    "zh", "ch", "sh", "b", "p", "m", "f", "d", "t", "n", "l",
    "g", "k", "h", "j", "q", "x", "r", "z", "c", "s",
)


def _basic_style(
    syllable: str,
    style: int,
    strict: bool,
    v_to_u: bool,
    neutral_five: bool,
) -> str:
    normal_chars = []
    tone = ""
    mark_index = -1
    for char in syllable:
        base, digit = _TONE_MARKS.get(char, (char, ""))
        if digit:
            tone = digit
            mark_index = len(normal_chars)
        normal_chars.append(base)
    normal_u = "".join(normal_chars)
    normal = normal_u if v_to_u else normal_u.replace("ü", "v")
    initial = next((value for value in _INITIALS if normal.startswith(value)), "")
    if not strict and not initial and normal[:1] in ("y", "w"):
        initial = normal[:1]
    final_start = len(initial)
    if style == Style.TONE:
        return syllable
    if style == Style.NORMAL:
        return normal
    if style == Style.FIRST_LETTER:
        return normal[:1]
    if style == Style.INITIALS:
        return initial
    if style in (Style.TONE3, Style.FINALS_TONE3):
        value = normal[final_start:] if style == Style.FINALS_TONE3 else normal
        return value + (tone or ("5" if neutral_five and value else ""))
    if style in (Style.TONE2, Style.FINALS_TONE2):
        value = normal[final_start:] if style == Style.FINALS_TONE2 else normal
        index = max(0, mark_index - final_start)
        digit = tone or ("5" if neutral_five and value else "")
        return value[: index + 1] + digit + value[index + 1:]
    if style == Style.FINALS:
        return normal[final_start:]
    if style == Style.FINALS_TONE:
        return syllable[final_start:]
    raise NotImplementedError(f"Style {int(style)} is not covered by mojopypinyin")


def _render_original(
    original: str,
    style: int,
    strict: bool,
    v_to_u: bool,
    neutral_five: bool,
) -> str:
    syllable_id = data.SYLLABLE_ID.get(original)
    if syllable_id is not None:
        return _rendered_syllables(
            style, strict, v_to_u, neutral_five
        )[syllable_id]
    return _basic_style(original, style, strict, v_to_u, neutral_five)


def _handle_error(chars: str, errors, heteronym: bool) -> list[list[str]]:
    if callable(errors):
        value = errors(chars)
    elif errors == "default":
        value = chars
    elif errors == "ignore":
        return []
    elif errors == "exception":
        raise PinyinNotFoundException(chars)
    elif errors == "replace":
        value = "".join(f"{ord(char):x}" for char in chars)
    else:
        return []

    if not value:
        return []
    if isinstance(value, str):
        return [[value]]
    if value and isinstance(value[0], (list, tuple)):
        return [list(item if heteronym else item[:1]) for item in value]
    return [[item] for item in value]


def _built_units(text: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return _kernel(text)


def _convert_text(
    text: str,
    style: int,
    heteronym: bool,
    errors,
    strict: bool,
    v_to_u: bool,
    neutral_five: bool,
) -> list[list[str]]:
    if not text and callable(errors):
        return _handle_error("", errors, heteronym)
    if _CUSTOM_PHRASES and any(phrase in text for phrase in _CUSTOM_PHRASES):
        return _convert_text_with_custom_phrases(
            text, style, heteronym, errors, strict, v_to_u, neutral_five
        )
    codepoints, entry_ids, spans = _built_units(text)
    rendered_first = _rendered_first_entries(
        style, strict, v_to_u, neutral_five
    )
    rendered_syllables = None
    if heteronym:
        rendered_syllables = _rendered_syllables(
            style, strict, v_to_u, neutral_five
        )

    result: list[list[str]] = []
    index = 0
    while index < len(text):
        entry_id = int(entry_ids[index])
        custom = _CUSTOM_SINGLE.get(int(codepoints[index]))
        if custom is not None and int(spans[index]) == 1:
            originals = custom if heteronym else custom[:1]
            result.append([
                _render_original(value, style, strict, v_to_u, neutral_five)
                for value in originals
            ])
            index += 1
            continue
        if entry_id >= 0:
            if heteronym:
                values = [
                    rendered_syllables[syllable]
                    for syllable in data.ENTRIES[entry_id]
                ]
                result.append(list(dict.fromkeys(values)))
            else:
                value = rendered_first[entry_id]
                result.append([value])
            index += 1
            continue

        stop = index + 1
        if not _is_han(int(codepoints[index])):
            while (
                stop < len(text)
                and int(entry_ids[stop]) < 0
                and not _is_han(int(codepoints[stop]))
            ):
                stop += 1
        result.extend(_handle_error(text[index:stop], errors, heteronym))
        index = stop
    return result


def _convert_text_with_custom_phrases(
    text: str,
    style: int,
    heteronym: bool,
    errors,
    strict: bool,
    v_to_u: bool,
    neutral_five: bool,
) -> list[list[str]]:
    rendered_first = _rendered_first_entries(
        style, strict, v_to_u, neutral_five
    )
    rendered_syllables = _rendered_syllables(
        style, strict, v_to_u, neutral_five
    )
    result: list[list[str]] = []
    index = 0
    while index < len(text):
        remainder = text[index:]
        codepoints, entry_ids, spans = _kernel(remainder)
        entry_id = int(entry_ids[0])
        built_span = int(spans[0]) if entry_id >= 0 else 1
        matches = [
            (len(phrase), values)
            for phrase, values in _CUSTOM_PHRASES.items()
            if text.startswith(phrase, index)
        ]
        if matches:
            custom_span, custom_values = max(matches, key=lambda item: item[0])
            if custom_span >= built_span:
                for entry in custom_values:
                    originals = entry if heteronym else entry[:1]
                    result.append([
                        _render_original(
                            value, style, strict, v_to_u, neutral_five
                        )
                        for value in originals
                    ])
                index += custom_span
                continue

        if entry_id >= 0:
            for offset in range(built_span):
                current_id = int(entry_ids[offset])
                custom = (
                    _CUSTOM_SINGLE.get(int(codepoints[offset]))
                    if built_span == 1
                    else None
                )
                if custom is not None:
                    originals = custom if heteronym else custom[:1]
                    result.append([
                        _render_original(
                            value, style, strict, v_to_u, neutral_five
                        )
                        for value in originals
                    ])
                elif heteronym:
                    result.append(list(dict.fromkeys(
                        rendered_syllables[syllable]
                        for syllable in data.ENTRIES[current_id]
                    )))
                else:
                    result.append([rendered_first[current_id]])
            index += built_span
            continue

        stop = 1
        if not _is_han(int(codepoints[0])):
            while (
                stop < len(remainder)
                and int(entry_ids[stop]) < 0
                and not _is_han(int(codepoints[stop]))
            ):
                stop += 1
        result.extend(_handle_error(remainder[:stop], errors, heteronym))
        index += stop
    return result


def _normalize_input(hans: str | Iterable[str]) -> list[str]:
    if isinstance(hans, bytes):
        raise AssertionError("must be unicode string or [unicode, ...] list")
    if isinstance(hans, str):
        return [hans]
    values = list(hans)
    if any(not isinstance(value, str) for value in values):
        raise AssertionError("must be unicode string or [unicode, ...] list")
    return values


def pinyin(
    hans,
    style=Style.TONE,
    heteronym=False,
    errors="default",
    strict=True,
    v_to_u=False,
    neutral_tone_with_five=False,
):
    result = []
    for text in _normalize_input(hans):
        result.extend(
            _convert_text(
                text,
                int(style),
                bool(heteronym),
                errors,
                bool(strict),
                bool(v_to_u),
                bool(neutral_tone_with_five),
            )
        )
    return result


def lazy_pinyin(
    hans,
    style=Style.NORMAL,
    errors="default",
    strict=True,
    v_to_u=False,
    neutral_tone_with_five=False,
    tone_sandhi=False,
):
    if tone_sandhi:
        raise NotImplementedError("tone_sandhi=True is not covered")
    return [
        item
        for group in pinyin(
            hans,
            style=style,
            heteronym=False,
            errors=errors,
            strict=strict,
            v_to_u=v_to_u,
            neutral_tone_with_five=neutral_tone_with_five,
        )
        for item in group
    ]


def slug(
    hans,
    style=Style.NORMAL,
    heteronym=False,
    separator="-",
    errors="default",
    strict=True,
):
    return separator.join(
        item
        for group in pinyin(
            hans,
            style=style,
            heteronym=heteronym,
            errors=errors,
            strict=strict,
        )
        for item in group
    )


def load_single_dict(pinyin_dict, style="default"):
    for codepoint, value in pinyin_dict.items():
        readings = str(value).split(",")
        if style == "tone2":
            readings = [_tone2_to_tone(value) for value in readings]
        _CUSTOM_SINGLE[int(codepoint)] = tuple(readings)


def load_phrases_dict(phrases_dict, style="default"):
    for phrase, values in phrases_dict.items():
        converted = []
        for readings in values:
            readings = list(readings)
            if style == "tone2":
                readings = [_tone2_to_tone(value) for value in readings]
            converted.append(tuple(readings))
        _CUSTOM_PHRASES[str(phrase)] = tuple(converted)


def _tone2_to_tone(value: str) -> str:
    digits = {"1": "āēīōūǖ", "2": "áéíóúǘ", "3": "ǎěǐǒǔǚ", "4": "àèìòùǜ"}
    vowels = "aeiouv"
    chars = list(value)
    for index, char in enumerate(chars):
        if char in digits and index:
            base = chars[index - 1]
            if base in vowels:
                chars[index - 1] = digits[char][vowels.index(base)]
                del chars[index]
                break
    return "".join(chars)
