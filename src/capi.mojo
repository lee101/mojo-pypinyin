comptime U32Ptr = UnsafePointer[UInt32, AnyOrigin[mut=True]]
comptime I32Ptr = UnsafePointer[Int32, AnyOrigin[mut=True]]


def binary_lookup(keys: I32Ptr, values: I32Ptr, count: Int, key: Int) -> Int:
    var low = 0
    var high = count
    while low < high:
        var middle = (low + high) // 2
        var candidate = Int(keys.load(middle))
        if candidate < key:
            low = middle + 1
        else:
            high = middle
    if low < count and Int(keys.load(low)) == key:
        return Int(values.load(low))
    return -1


def trie_child(
    node: Int,
    codepoint: Int,
    node_first: I32Ptr,
    node_count: I32Ptr,
    edge_codes: I32Ptr,
    edge_next: I32Ptr,
) -> Int:
    var first = Int(node_first.load(node))
    var low = first
    var high = first + Int(node_count.load(node))
    while low < high:
        var middle = (low + high) // 2
        var candidate = Int(edge_codes.load(middle))
        if candidate < codepoint:
            low = middle + 1
        else:
            high = middle
    if low < first + Int(node_count.load(node)) and Int(edge_codes.load(low)) == codepoint:
        return Int(edge_next.load(low))
    return -1


@export("mpy_convert")
def mpy_convert(
    codepoint_addr: Int,
    n: Int,
    dense_addr: Int,
    dense_start: Int,
    dense_count: Int,
    extra_code_addr: Int,
    extra_entry_addr: Int,
    extra_count: Int,
    node_first_addr: Int,
    node_count_addr: Int,
    node_terminal_addr: Int,
    edge_code_addr: Int,
    edge_next_addr: Int,
    phrase_offset_addr: Int,
    phrase_entry_addr: Int,
    result_addr: Int,
    span_addr: Int,
) abi("C"):
    var codepoints = U32Ptr(unsafe_from_address=codepoint_addr)
    var dense = I32Ptr(unsafe_from_address=dense_addr)
    var extra_codes = I32Ptr(unsafe_from_address=extra_code_addr)
    var extra_entries = I32Ptr(unsafe_from_address=extra_entry_addr)
    var node_first = I32Ptr(unsafe_from_address=node_first_addr)
    var node_count = I32Ptr(unsafe_from_address=node_count_addr)
    var node_terminal = I32Ptr(unsafe_from_address=node_terminal_addr)
    var edge_codes = I32Ptr(unsafe_from_address=edge_code_addr)
    var edge_next = I32Ptr(unsafe_from_address=edge_next_addr)
    var phrase_offsets = I32Ptr(unsafe_from_address=phrase_offset_addr)
    var phrase_entries = I32Ptr(unsafe_from_address=phrase_entry_addr)
    var result = I32Ptr(unsafe_from_address=result_addr)
    var spans = I32Ptr(unsafe_from_address=span_addr)

    var i = 0
    while i < n:
        var node = 0
        var cursor = i
        var terminal = -1
        while cursor < n:
            var child = trie_child(
                node,
                Int(codepoints.load(cursor)),
                node_first,
                node_count,
                edge_codes,
                edge_next,
            )
            if child < 0:
                break
            node = child
            cursor += 1
            var candidate = Int(node_terminal.load(node))
            if candidate >= 0:
                terminal = candidate

        if terminal >= 0:
            var start = Int(phrase_offsets.load(terminal))
            var stop = Int(phrase_offsets.load(terminal + 1))
            var length = stop - start
            spans.store(i, Int32(length))
            var k = 0
            while k < length:
                result.store(i + k, phrase_entries.load(start + k))
                if k > 0:
                    spans.store(i + k, Int32(0))
                k += 1
            i += length
        else:
            var codepoint = Int(codepoints.load(i))
            var entry = -1
            if codepoint >= dense_start and codepoint < dense_start + dense_count:
                entry = Int(dense.load(codepoint - dense_start))
            else:
                entry = binary_lookup(
                    extra_codes, extra_entries, extra_count, codepoint
                )
            result.store(i, Int32(entry))
            spans.store(i, Int32(1))
            i += 1
