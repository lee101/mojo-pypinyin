# mojo-pypinyin

Chinese-to-pinyin conversion with dictionary lookup and phrase segmentation in
[Mojo](https://www.modular.com/mojo).

`mojopypinyin` mirrors pypinyin's core function names and signatures. It ships a
compact generated form of the complete pypinyin 0.55.0 character and phrase
dictionaries, so conversion does not import or call pypinyin at runtime.
pypinyin is present in the development environment only for parity tests.

## Covered subset

- `pinyin`, `lazy_pinyin`, and `slug`
- The complete character and phrase dictionaries shipped by pypinyin 0.55.0
- Phrase-aware pronunciation, heteronyms, string and list input
- `default`, `ignore`, `replace`, `exception`, and callable error handling
- `strict`, `v_to_u`, and `neutral_tone_with_five`
- `load_single_dict` and `load_phrases_dict`, including embedded custom phrases
- `Style.NORMAL`, `TONE`, `TONE2`, `TONE3`, `INITIALS`, `FIRST_LETTER`,
  `FINALS`, `FINALS_TONE`, `FINALS_TONE2`, and `FINALS_TONE3`

The Bopomofo, Cyrillic, Wade-Giles, Gwoyeu Romatzyh, and braille styles are not
implemented. `tone_sandhi=True`, converter subclasses, segmentation extension
objects, and pypinyin's command-line interface are also outside the current
scope. Unsupported styles and tone sandhi raise `NotImplementedError`. The
current release is a Linux x86_64 source distribution; it does not provide a
PyPI wheel or a standalone command.

## Install and use

```bash
pixi install
pixi run build
pixi run test
```

Run Python through pixi so that the repository's `python/` directory is on
`PYTHONPATH`:

```bash
pixi run python - <<'PY'
import mojopypinyin as pypinyin
from mojopypinyin import Style

print(pypinyin.pinyin("重庆银行"))
print(pypinyin.lazy_pinyin("中国人", style=Style.TONE3))
print(pypinyin.slug("你好，世界", separator=" "))
PY
```

Output:

```text
[['chóng'], ['qìng'], ['yín'], ['háng']]
['zhong1', 'guo2', 'ren2']
ni hao ， shi jie
```

For the covered API, changing `import pypinyin` to
`import mojopypinyin as pypinyin` is otherwise sufficient.

## Benchmarks

Measured with `pixi run bench` on this x86_64 machine using Python 3.13.14 and
the same 108,000-character input for both libraries:

| case | mojopypinyin | pypinyin 0.55.0 | speedup |
| --- | ---: | ---: | ---: |
| `lazy_pinyin`, NORMAL | 100.87 ms | 915.65 ms | 9.08x |
| `lazy_pinyin`, TONE3 | 110.06 ms | 1047.17 ms | 9.51x |
| `pinyin`, TONE | 98.25 ms | 626.04 ms | 6.37x |

These are best-of-five wall-clock measurements after loading the shared
library. The benchmark first asserts that both implementations return exactly
the same result. Run `pixi run bench` to reproduce it; the pixi task takes a
machine-wide lock to avoid overlap with other benchmark jobs.

No GPU path is provided. The conversion kernel is a branch-heavy trie and
dictionary lookup with low arithmetic intensity, so host/device transfers and
launch overhead would dominate its few integer operations per table load.

## How it works

At build time, one Mojo compilation unit becomes
`dist/libmojo-pypinyin.so`. Python passes contiguous NumPy buffers to that
library through `ctypes`. In accordance with Mojo's C ABI constraints, every
buffer crosses the boundary as an integer address and is reconstructed as an
`UnsafePointer[..., AnyOrigin[mut=True]]` inside the exported function. Python
owns every allocation.

Input strings are encoded once as contiguous UTF-32 codepoints. The Mojo kernel
walks a compact phrase trie using binary-searched edge arrays,
chooses the same longest-prefix phrases as pypinyin, and writes integer reading
IDs to a caller-owned output buffer. Common CJK character records use a dense
lookup table; extension-plane records use a sorted binary-search table.

Python turns those IDs into the requested nested-list shape. Style conversion
does not repeatedly manipulate Unicode strings: generated tables map source
syllables to exact pypinyin results for every covered style, strictness,
`v_to_u`, and neutral-tone combination.

The generated dictionaries and style results are derived from pypinyin 0.55.0
under its MIT license; see `THIRD_PARTY_NOTICES.md`.

MIT.
