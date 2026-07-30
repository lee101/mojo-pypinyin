"""Generate the compact, runtime-independent tables shipped with mojopypinyin."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pypinyin
from pypinyin.constants import PHRASES_DICT, PINYIN_DICT
from pypinyin.converter import UltimateConverter


ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "python" / "mojopypinyin" / "data"
STYLE_ORDER = [0, 1, 2, 8, 3, 4, 5, 6, 7, 9]
DENSE_START = 0x3000
DENSE_STOP = 0xA000


def main() -> None:
    syllables = set()
    entries = set()
    single_entries = {}
    for codepoint, value in PINYIN_DICT.items():
        entry = tuple(value.split(","))
        single_entries[int(codepoint)] = entry
        entries.add(entry)
        syllables.update(entry)

    phrase_entries = {}
    for phrase, values in PHRASES_DICT.items():
        converted = tuple(tuple(readings) for readings in values)
        phrase_entries[phrase] = converted
        entries.update(converted)
        for entry in converted:
            syllables.update(entry)

    syllables = sorted(syllables)
    syllable_id = {value: index for index, value in enumerate(syllables)}
    entries = sorted(entries)
    entry_id = {value: index for index, value in enumerate(entries)}

    dense = np.full(DENSE_STOP - DENSE_START, -1, dtype=np.int32)
    extra_codes = []
    extra_entries = []
    for codepoint, entry in sorted(single_entries.items()):
        eid = entry_id[entry]
        if DENSE_START <= codepoint < DENSE_STOP:
            dense[codepoint - DENSE_START] = eid
        else:
            extra_codes.append(codepoint)
            extra_entries.append(eid)

    children: list[dict[int, int]] = [{}]
    terminals = [-1]
    phrase_offsets = [0]
    phrase_values = []
    for phrase_id, phrase in enumerate(sorted(phrase_entries)):
        node = 0
        for char in phrase:
            codepoint = ord(char)
            child = children[node].get(codepoint)
            if child is None:
                child = len(children)
                children[node][codepoint] = child
                children.append({})
                terminals.append(-1)
            node = child
        terminals[node] = phrase_id
        phrase_values.extend(entry_id[entry] for entry in phrase_entries[phrase])
        phrase_offsets.append(len(phrase_values))

    node_first = []
    node_count = []
    edge_codes = []
    edge_next = []
    for edges in children:
        node_first.append(len(edge_codes))
        ordered = sorted(edges.items())
        node_count.append(len(ordered))
        for codepoint, target in ordered:
            edge_codes.append(codepoint)
            edge_next.append(target)

    rendered: list[list[str]] = []
    all_tokens = set()
    for v_to_u in (False, True):
        for neutral_five in (False, True):
            converter = UltimateConverter(
                v_to_u=v_to_u,
                neutral_tone_with_five=neutral_five,
            )
            for strict in (False, True):
                for style in STYLE_ORDER:
                    values = []
                    for syllable in syllables:
                        try:
                            value = converter.convert_style(
                                "", syllable, style=style, strict=strict
                            )
                        except TypeError:
                            value = UltimateConverter(
                                v_to_u=v_to_u
                            ).convert_style(
                                "", syllable, style=style, strict=strict
                            )
                        values.append(value)
                    rendered.append(values)
                    all_tokens.update(values)

    tokens = sorted(all_tokens)
    token_id = {value: index for index, value in enumerate(tokens)}
    style_tokens = np.asarray(
        [[token_id[value] for value in row] for row in rendered],
        dtype=np.int32,
    )

    metadata = {
        "upstream": f"pypinyin {pypinyin.__version__}",
        "dense_start": DENSE_START,
        "dense_stop": DENSE_STOP,
        "style_order": STYLE_ORDER,
        "syllables": syllables,
        "entries": [[syllable_id[value] for value in entry] for entry in entries],
        "tokens": tokens,
        "stats": {
            "characters": len(single_entries),
            "phrases": len(phrase_entries),
            "syllables": len(syllables),
            "entries": len(entries),
            "trie_nodes": len(children),
        },
    }

    DEST.mkdir(parents=True, exist_ok=True)
    (DEST / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )
    np.savez_compressed(
        DEST / "arrays.npz",
        dense=dense,
        extra_codes=np.asarray(extra_codes, dtype=np.int32),
        extra_entries=np.asarray(extra_entries, dtype=np.int32),
        node_first=np.asarray(node_first, dtype=np.int32),
        node_count=np.asarray(node_count, dtype=np.int32),
        node_terminal=np.asarray(terminals, dtype=np.int32),
        edge_codes=np.asarray(edge_codes, dtype=np.int32),
        edge_next=np.asarray(edge_next, dtype=np.int32),
        phrase_offsets=np.asarray(phrase_offsets, dtype=np.int32),
        phrase_entries=np.asarray(phrase_values, dtype=np.int32),
        style_tokens=style_tokens,
    )
    print(json.dumps(metadata["stats"], sort_keys=True))


if __name__ == "__main__":
    main()
