from __future__ import annotations

import inspect
import random

import pytest
import pypinyin as upstream

import mojopypinyin as mojo
from mojopypinyin import _data
from mojopypinyin._data import META


CORE_STYLES = [
    mojo.Style.NORMAL,
    mojo.Style.TONE,
    mojo.Style.TONE2,
    mojo.Style.TONE3,
    mojo.Style.INITIALS,
    mojo.Style.FIRST_LETTER,
    mojo.Style.FINALS,
    mojo.Style.FINALS_TONE,
    mojo.Style.FINALS_TONE2,
    mojo.Style.FINALS_TONE3,
]


def test_upstream_dataset_is_fully_vendored():
    assert META["stats"] == {
        "characters": 41923,
        "phrases": 47111,
        "syllables": 1559,
        "entries": 8660,
        "trie_nodes": META["stats"]["trie_nodes"],
    }
    assert META["stats"]["trie_nodes"] > 80_000


def test_ffi_tables_are_native_contiguous_and_read_only():
    for array in _data._KERNEL_ARRAYS.values():
        assert array.dtype.isnative
        assert array.dtype == "int32"
        assert array.ndim == 1
        assert array.flags.c_contiguous
        assert array.flags.aligned
        assert not array.flags.writeable
        assert array.ctypes.data


def test_ffi_validation_rejects_wrong_dtype_and_cross_table_lengths():
    arrays = dict(_data._KERNEL_ARRAYS)
    arrays["dense"] = arrays["dense"].astype("int64")
    with pytest.raises(RuntimeError, match="native int32"):
        _data._validate_kernel_data(arrays)

    arrays = dict(_data._KERNEL_ARRAYS)
    arrays["extra_entries"] = arrays["extra_entries"][:-1]
    with pytest.raises(RuntimeError, match="key/value lengths"):
        _data._validate_kernel_data(arrays)

    arrays = dict(_data._KERNEL_ARRAYS)
    arrays["phrase_offsets"] = arrays["phrase_offsets"].copy()
    arrays["phrase_offsets"][1] += 1
    with pytest.raises(RuntimeError, match="monotonic|trie depth"):
        _data._validate_kernel_data(arrays)


def test_public_signatures_match_upstream():
    for name in ("pinyin", "lazy_pinyin", "slug", "load_single_dict", "load_phrases_dict"):
        assert inspect.signature(getattr(mojo, name)) == inspect.signature(
            getattr(upstream, name)
        )


@pytest.mark.parametrize("style", CORE_STYLES)
@pytest.mark.parametrize("strict", [False, True])
def test_all_core_styles_match(style, strict):
    text = "你好，重庆银行音乐中心的女儿略有不同。"
    assert mojo.pinyin(text, style=style, strict=strict) == upstream.pinyin(
        text, style=style, strict=strict
    )


@pytest.mark.parametrize(
    "text",
    [
        "你好，我是中国人，我爱我的祖国",
        "一不小心走进重庆银行",
        "音乐和快乐并不冲突",
        "朝辞白帝彩云间，千里江陵一日还。",
        "衣裳战略儿化音",
    ],
)
def test_phrase_disambiguation_matches(text):
    assert mojo.pinyin(text) == upstream.pinyin(text)
    assert mojo.lazy_pinyin(text) == upstream.lazy_pinyin(text)


def test_heteronyms_match_and_are_deduplicated():
    text = "中心重庆还行"
    assert mojo.pinyin(text, heteronym=True) == upstream.pinyin(
        text, heteronym=True
    )


@pytest.mark.parametrize("errors", ["default", "ignore", "replace"])
def test_error_modes_match(errors):
    text = "中国 abc\U0010ffff 未知"
    assert mojo.pinyin(text, errors=errors) == upstream.pinyin(text, errors=errors)


def test_callable_error_handler_matches():
    callback = lambda value: [f"<{value}>"]
    text = "中国 abc"
    assert mojo.pinyin(text, errors=callback) == upstream.pinyin(
        text, errors=callback
    )
    assert mojo.pinyin("", errors=callback) == upstream.pinyin("", errors=callback)


def test_exception_error_mode():
    with pytest.raises(mojo.PinyinNotFoundException):
        mojo.pinyin("中国!", errors="exception")


def test_list_input_and_slug_match():
    words = ["重庆", "银行", "音乐"]
    assert mojo.pinyin(words) == upstream.pinyin(words)
    assert mojo.slug(words, separator=" ") == upstream.slug(words, separator=" ")


@pytest.mark.parametrize("style", [mojo.Style.NORMAL, mojo.Style.TONE2, mojo.Style.TONE3])
@pytest.mark.parametrize("v_to_u", [False, True])
@pytest.mark.parametrize("neutral", [False, True])
def test_v_and_neutral_tone_options_match(style, v_to_u, neutral):
    text = "战略衣裳好了"
    assert mojo.lazy_pinyin(
        text,
        style=style,
        v_to_u=v_to_u,
        neutral_tone_with_five=neutral,
    ) == upstream.lazy_pinyin(
        text,
        style=style,
        v_to_u=v_to_u,
        neutral_tone_with_five=neutral,
    )


def test_random_character_parity():
    rng = random.Random(0)
    codepoints = rng.sample(list(upstream.constants.PINYIN_DICT), 1000)
    text = "".join(map(chr, codepoints))
    for style in CORE_STYLES:
        assert mojo.lazy_pinyin(text, style=style) == upstream.lazy_pinyin(
            text, style=style
        )


def test_random_phrase_parity():
    rng = random.Random(1)
    phrases = rng.sample(list(upstream.constants.PHRASES_DICT), 300)
    for phrase in phrases:
        assert mojo.pinyin(phrase) == upstream.pinyin(phrase)


def test_empty_and_bytes_behavior():
    assert mojo.pinyin("") == upstream.pinyin("")
    assert mojo.lazy_pinyin([]) == upstream.lazy_pinyin([])
    with pytest.raises(AssertionError):
        mojo.pinyin("中国".encode())


def test_custom_dictionaries():
    single = {ord("桔"): "jú"}
    phrase = {"桔子水": [["jú"], ["zǐ"], ["shuǐ"]]}
    mojo.load_single_dict(single)
    mojo.load_phrases_dict(phrase)
    assert mojo.pinyin("桔") == [["jú"]]
    assert mojo.lazy_pinyin("桔子水") == ["ju", "zi", "shui"]
    assert mojo.lazy_pinyin("喝桔子水很好") == ["he", "ju", "zi", "shui", "hen", "hao"]

    missing_codepoint = 0x3402
    mojo.load_single_dict({missing_codepoint: "cè"})
    assert mojo.pinyin(chr(missing_codepoint)) == [["cè"]]


def test_unsupported_styles_and_tone_sandhi_are_explicit():
    with pytest.raises(NotImplementedError):
        mojo.pinyin("中国", style=mojo.Style.BOPOMOFO)
    with pytest.raises(NotImplementedError):
        mojo.lazy_pinyin("你好", tone_sandhi=True)
