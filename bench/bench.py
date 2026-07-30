"""Benchmark mojopypinyin against pypinyin on identical text."""

from __future__ import annotations

import math
import os
import platform
import sys
import time

sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "python"
    ),
)

import mojopypinyin as mojo  # noqa: E402
import pypinyin as upstream  # noqa: E402


def timeit(fn, repeat=5):
    best = math.inf
    for _ in range(repeat):
        start = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - start)
    return best


def main():
    sentence = "你好，我是中国人，我在重庆银行工作，也喜欢音乐和旅行。"
    text = sentence * 4000
    cases = [
        (
            "lazy_pinyin NORMAL",
            lambda: mojo.lazy_pinyin(text),
            lambda: upstream.lazy_pinyin(text),
        ),
        (
            "lazy_pinyin TONE3",
            lambda: mojo.lazy_pinyin(text, style=mojo.Style.TONE3),
            lambda: upstream.lazy_pinyin(text, style=upstream.Style.TONE3),
        ),
        (
            "pinyin TONE",
            lambda: mojo.pinyin(text),
            lambda: upstream.pinyin(text),
        ),
    ]

    mojo.lazy_pinyin("中国")
    print(
        f"Machine: {platform.processor() or platform.machine()}, "
        f"Python {platform.python_version()}, {len(text):,} characters"
    )
    print()
    print("| case | mojopypinyin | pypinyin 0.55.0 | speedup |")
    print("| --- | ---: | ---: | ---: |")
    for name, ours, theirs in cases:
        ours_result = ours()
        theirs_result = theirs()
        assert ours_result == theirs_result
        ours_time = timeit(ours)
        theirs_time = timeit(theirs)
        print(
            f"| {name} | {ours_time * 1000:.2f} ms | "
            f"{theirs_time * 1000:.2f} ms | {theirs_time / ours_time:.2f}x |"
        )


if __name__ == "__main__":
    main()
