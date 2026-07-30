"""Read-only conversion tables generated from pypinyin 0.55.0."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


DATA_DIR = Path(__file__).with_name("data")
META = json.loads((DATA_DIR / "metadata.json").read_text(encoding="utf-8"))
_archive = np.load(DATA_DIR / "arrays.npz", allow_pickle=False)

DENSE = _archive["dense"]
EXTRA_CODES = _archive["extra_codes"]
EXTRA_ENTRIES = _archive["extra_entries"]
NODE_FIRST = _archive["node_first"]
NODE_COUNT = _archive["node_count"]
NODE_TERMINAL = _archive["node_terminal"]
EDGE_CODES = _archive["edge_codes"]
EDGE_NEXT = _archive["edge_next"]
PHRASE_OFFSETS = _archive["phrase_offsets"]
PHRASE_ENTRIES = _archive["phrase_entries"]
STYLE_TOKENS = _archive["style_tokens"]

DENSE_START = int(META["dense_start"])
STYLE_ORDER = tuple(META["style_order"])
STYLE_POSITION = {style: index for index, style in enumerate(STYLE_ORDER)}
SYLLABLES = tuple(META["syllables"])
SYLLABLE_ID = {value: index for index, value in enumerate(SYLLABLES)}
ENTRIES = tuple(tuple(value) for value in META["entries"])
TOKENS = tuple(META["tokens"])

_KERNEL_ARRAYS = {
    "dense": DENSE,
    "extra_codes": EXTRA_CODES,
    "extra_entries": EXTRA_ENTRIES,
    "node_first": NODE_FIRST,
    "node_count": NODE_COUNT,
    "node_terminal": NODE_TERMINAL,
    "edge_codes": EDGE_CODES,
    "edge_next": EDGE_NEXT,
    "phrase_offsets": PHRASE_OFFSETS,
    "phrase_entries": PHRASE_ENTRIES,
}


def _validate_kernel_data(arrays: dict[str, np.ndarray] = _KERNEL_ARRAYS) -> None:
    """Reject archive layouts that would be unsafe for the pointer-only C ABI."""
    for name, array in arrays.items():
        if array.dtype != np.dtype(np.int32):
            raise RuntimeError(f"{name} must have native int32 dtype, got {array.dtype}")
        if array.ndim != 1 or not array.flags.c_contiguous or not array.flags.aligned:
            raise RuntimeError(f"{name} must be an aligned, contiguous 1-D array")
        if array.size == 0 or int(array.ctypes.data) == 0:
            raise RuntimeError(f"{name} must have non-empty storage")

    dense = arrays["dense"]
    extra_codes = arrays["extra_codes"]
    extra_entries = arrays["extra_entries"]
    node_first = arrays["node_first"]
    node_count = arrays["node_count"]
    node_terminal = arrays["node_terminal"]
    edge_codes = arrays["edge_codes"]
    edge_next = arrays["edge_next"]
    phrase_offsets = arrays["phrase_offsets"]
    phrase_entries = arrays["phrase_entries"]

    if extra_codes.size != extra_entries.size:
        raise RuntimeError("extra lookup key/value lengths differ")
    if not (node_first.size == node_count.size == node_terminal.size):
        raise RuntimeError("trie node table lengths differ")
    if edge_codes.size != edge_next.size:
        raise RuntimeError("trie edge table lengths differ")
    terminal_ids = node_terminal[node_terminal >= 0]
    if phrase_offsets.size != terminal_ids.size + 1:
        raise RuntimeError("phrase offset count does not match trie terminals")
    if (
        terminal_ids.size
        and not np.array_equal(np.sort(terminal_ids), np.arange(terminal_ids.size))
    ):
        raise RuntimeError("trie terminal ids are not a complete phrase-id range")
    if int(phrase_offsets[0]) != 0 or int(phrase_offsets[-1]) != phrase_entries.size:
        raise RuntimeError("phrase offsets do not cover the phrase entry table")
    if np.any(phrase_offsets[1:] < phrase_offsets[:-1]):
        raise RuntimeError("phrase offsets are not monotonic")
    if np.any(node_first < 0) or np.any(node_count < 0):
        raise RuntimeError("trie node ranges must be non-negative")
    if np.any(node_first.astype(np.int64) + node_count > edge_codes.size):
        raise RuntimeError("trie node range exceeds the edge tables")
    if np.any(edge_next < 0) or np.any(edge_next >= node_first.size):
        raise RuntimeError("trie edge target is outside the node tables")
    if extra_codes.size > 1 and np.any(extra_codes[1:] <= extra_codes[:-1]):
        raise RuntimeError("extra lookup keys must be strictly increasing")

    depths = np.full(node_first.size, -1, dtype=np.int64)
    depths[0] = 0
    stack = [0]
    while stack:
        node = stack.pop()
        first = int(node_first[node])
        count = int(node_count[node])
        codes = edge_codes[first : first + count]
        if codes.size > 1 and np.any(codes[1:] <= codes[:-1]):
            raise RuntimeError("trie child keys must be strictly increasing")
        for child in edge_next[first : first + count]:
            child_index = int(child)
            if depths[child_index] >= 0:
                raise RuntimeError("trie nodes must have exactly one acyclic parent")
            depths[child_index] = depths[node] + 1
            stack.append(child_index)
    if np.any(depths < 0):
        raise RuntimeError("trie contains unreachable nodes")
    for node in np.flatnonzero(node_terminal >= 0):
        phrase_id = int(node_terminal[node])
        stored_length = int(phrase_offsets[phrase_id + 1] - phrase_offsets[phrase_id])
        if stored_length != int(depths[node]):
            raise RuntimeError("phrase output length differs from its trie depth")

    entry_count = len(ENTRIES)
    for name, array in (
        ("dense", dense),
        ("extra_entries", extra_entries),
        ("phrase_entries", phrase_entries),
    ):
        if np.any(array < -1) or np.any(array >= entry_count):
            raise RuntimeError(f"{name} contains an invalid entry id")


_validate_kernel_data()
for _array in _KERNEL_ARRAYS.values():
    _array.flags.writeable = False


def addr(array: np.ndarray) -> int:
    if (
        array.dtype != np.dtype(np.int32)
        or array.ndim != 1
        or not array.flags.c_contiguous
        or not array.flags.aligned
        or array.size == 0
    ):
        raise RuntimeError("unsafe array passed to the Mojo C ABI")
    return int(array.ctypes.data)
