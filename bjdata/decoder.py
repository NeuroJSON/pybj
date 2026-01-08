# Copyright (c) 2020-2025 Qianqian Fang <q.fang at neu.edu>. All rights reserved.
# Copyright (c) 2016-2019 Iotic Labs Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://github.com/NeuroJSON/pybj/blob/master/LICENSE
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


"""BJData (Draft 4) and UBJSON decoder with SOA support"""

from io import BytesIO
from struct import Struct, pack, error as StructError
from decimal import Decimal, DecimalException
from functools import reduce

from .compat import raise_from, intern_unicode
from .markers import (
    TYPE_NONE,
    TYPE_NULL,
    TYPE_NOOP,
    TYPE_BOOL_TRUE,
    TYPE_BOOL_FALSE,
    TYPE_BYTE,
    TYPE_INT8,
    TYPE_UINT8,
    TYPE_INT16,
    TYPE_INT32,
    TYPE_INT64,
    TYPE_FLOAT32,
    TYPE_FLOAT64,
    TYPE_HIGH_PREC,
    TYPE_CHAR,
    TYPE_UINT16,
    TYPE_UINT32,
    TYPE_UINT64,
    TYPE_FLOAT16,
    TYPE_STRING,
    OBJECT_START,
    OBJECT_END,
    ARRAY_START,
    ARRAY_END,
    CONTAINER_TYPE,
    CONTAINER_COUNT,
)
from numpy import (
    array as ndarray,
    dtype as npdtype,
    frombuffer as buffer2numpy,
    half as halfprec,
    zeros as npzeros,
    empty as npempty,
)
from array import array as typedarray

__TYPES = frozenset(
    (
        TYPE_NULL,
        TYPE_BOOL_TRUE,
        TYPE_BOOL_FALSE,
        TYPE_BYTE,
        TYPE_INT8,
        TYPE_UINT8,
        TYPE_INT16,
        TYPE_INT32,
        TYPE_INT64,
        TYPE_FLOAT32,
        TYPE_FLOAT64,
        TYPE_UINT16,
        TYPE_UINT32,
        TYPE_UINT64,
        TYPE_FLOAT16,
        TYPE_HIGH_PREC,
        TYPE_CHAR,
        TYPE_STRING,
        ARRAY_START,
        OBJECT_START,
    )
)
__TYPES_NO_DATA = frozenset((TYPE_NULL, TYPE_BOOL_FALSE, TYPE_BOOL_TRUE))
__TYPES_INT = frozenset(
    (
        TYPE_BYTE,
        TYPE_INT8,
        TYPE_UINT8,
        TYPE_INT16,
        TYPE_INT32,
        TYPE_INT64,
        TYPE_UINT16,
        TYPE_UINT32,
        TYPE_UINT64,
    )
)
__TYPES_FIXLEN = frozenset(
    (
        TYPE_BYTE,
        TYPE_INT8,
        TYPE_UINT8,
        TYPE_INT16,
        TYPE_INT32,
        TYPE_INT64,
        TYPE_UINT16,
        TYPE_UINT32,
        TYPE_UINT64,
        TYPE_FLOAT16,
        TYPE_FLOAT32,
        TYPE_FLOAT64,
        TYPE_CHAR,
    )
)

__SMALL_INTS_DECODED = [
    {pack(">b", i): i for i in range(-128, 128)},
    {pack("<b", i): i for i in range(-128, 128)},
]
__SMALL_UINTS_DECODED = [
    {pack(">B", i): i for i in range(256)},
    {pack("<B", i): i for i in range(256)},
]
__UNPACK_INT16 = [Struct(">h").unpack, Struct("<h").unpack]
__UNPACK_INT32 = [Struct(">i").unpack, Struct("<i").unpack]
__UNPACK_INT64 = [Struct(">q").unpack, Struct("<q").unpack]
__UNPACK_UINT16 = [Struct(">H").unpack, Struct("<H").unpack]
__UNPACK_UINT32 = [Struct(">I").unpack, Struct("<I").unpack]
__UNPACK_UINT64 = [Struct(">Q").unpack, Struct("<Q").unpack]
__UNPACK_FLOAT16 = [Struct(">h").unpack, Struct("<h").unpack]
__UNPACK_FLOAT32 = [Struct(">f").unpack, Struct("<f").unpack]
__UNPACK_FLOAT64 = [Struct(">d").unpack, Struct("<d").unpack]

__DTYPE_MAP = {
    TYPE_BYTE: "B",
    TYPE_INT8: "b",
    TYPE_UINT8: "B",
    TYPE_INT16: "h",
    TYPE_UINT16: "H",
    TYPE_INT32: "i",
    TYPE_UINT32: "I",
    TYPE_INT64: "q",
    TYPE_UINT64: "Q",
    TYPE_FLOAT16: "h",
    TYPE_FLOAT32: "f",
    TYPE_FLOAT64: "d",
    TYPE_CHAR: "c",
}

__DTYPELEN_MAP = {
    TYPE_BYTE: 1,
    TYPE_INT8: 1,
    TYPE_UINT8: 1,
    TYPE_INT16: 2,
    TYPE_UINT16: 2,
    TYPE_INT32: 4,
    TYPE_UINT32: 4,
    TYPE_INT64: 8,
    TYPE_UINT64: 8,
    TYPE_FLOAT16: 2,
    TYPE_FLOAT32: 4,
    TYPE_FLOAT64: 8,
    TYPE_CHAR: 1,
}

# Numpy dtype strings for SOA
__NUMPY_DTYPE_MAP = {
    TYPE_BYTE: "u1",
    TYPE_INT8: "i1",
    TYPE_UINT8: "u1",
    TYPE_INT16: "i2",
    TYPE_UINT16: "u2",
    TYPE_INT32: "i4",
    TYPE_UINT32: "u4",
    TYPE_INT64: "i8",
    TYPE_UINT64: "u8",
    TYPE_FLOAT16: "f2",
    TYPE_FLOAT32: "f4",
    TYPE_FLOAT64: "f8",
    TYPE_BOOL_TRUE: "?",
    TYPE_BOOL_FALSE: "?",
}


class DecoderException(ValueError):
    """Raised when decoding of a UBJSON stream fails."""

    def __init__(self, message, position=None):
        if position is not None:
            super(DecoderException, self).__init__(
                "%s (at byte %d)" % (message, position), position
            )
        else:
            super(DecoderException, self).__init__(str(message), None)

    @property
    def position(self):
        """Position in stream where decoding failed. Can be None in case where decoding from string of when file-like
        object does not support tell().
        """
        return self.args[1]  # pylint: disable=unsubscriptable-object


# pylint: disable=unused-argument
def __decode_high_prec(fp_read, marker, le=1):
    length = __decode_int_non_negative(fp_read, fp_read(1), le)
    raw = fp_read(length)
    if len(raw) < length:
        raise DecoderException("High prec. too short")
    try:
        return Decimal(raw.decode("utf-8"))
    except UnicodeError as ex:
        raise_from(DecoderException("Failed to decode decimal string"), ex)
    except DecimalException as ex:
        raise_from(DecoderException("Failed to decode decimal"), ex)


def __decode_int_non_negative(fp_read, marker, le=1):
    if marker not in __TYPES_INT:
        raise DecoderException("Integer marker expected")
    value = __METHOD_MAP[marker](fp_read, marker, le)
    if value < 0:
        raise DecoderException("Negative count/length unexpected")
    return value


def __decode_byte(fp_read, marker, le=1):
    try:
        return __SMALL_UINTS_DECODED[le][fp_read(1)]
    except KeyError as ex:
        raise_from(DecoderException("Failed to unpack byte"), ex)


def __decode_int8(fp_read, marker, le=1):
    try:
        return __SMALL_INTS_DECODED[le][fp_read(1)]
    except KeyError as ex:
        raise_from(DecoderException("Failed to unpack int8"), ex)


def __decode_uint8(fp_read, marker, le=1):
    try:
        return __SMALL_UINTS_DECODED[le][fp_read(1)]
    except KeyError as ex:
        raise_from(DecoderException("Failed to unpack uint8"), ex)


def __decode_int16(fp_read, marker, le=1):
    try:
        return __UNPACK_INT16[le](fp_read(2))[0]
    except StructError as ex:
        raise_from(DecoderException("Failed to unpack int16"), ex)


def __decode_int32(fp_read, marker, le=1):
    try:
        return __UNPACK_INT32[le](fp_read(4))[0]
    except StructError as ex:
        raise_from(DecoderException("Failed to unpack int32"), ex)


def __decode_int64(fp_read, marker, le=1):
    try:
        return __UNPACK_INT64[le](fp_read(8))[0]
    except StructError as ex:
        raise_from(DecoderException("Failed to unpack int64"), ex)


def __decode_uint16(fp_read, marker, le=1):
    try:
        return __UNPACK_UINT16[le](fp_read(2))[0]
    except StructError as ex:
        raise_from(DecoderException("Failed to unpack uint16"), ex)


def __decode_uint32(fp_read, marker, le=1):
    try:
        return __UNPACK_UINT32[le](fp_read(4))[0]
    except StructError as ex:
        raise_from(DecoderException("Failed to unpack uint32"), ex)


def __decode_uint64(fp_read, marker, le=1):
    try:
        return __UNPACK_UINT64[le](fp_read(8))[0]
    except StructError as ex:
        raise_from(DecoderException("Failed to unpack uint64"), ex)


def __decode_float16(fp_read, marker, le=1):
    try:
        return __UNPACK_FLOAT16[le](fp_read(2))[0]
    except StructError as ex:
        raise_from(DecoderException("Failed to unpack float16"), ex)


def __decode_float32(fp_read, marker, le=1):
    try:
        return __UNPACK_FLOAT32[le](fp_read(4))[0]
    except StructError as ex:
        raise_from(DecoderException("Failed to unpack float32"), ex)


def __decode_float64(fp_read, marker, le=1):
    try:
        return __UNPACK_FLOAT64[le](fp_read(8))[0]
    except StructError as ex:
        raise_from(DecoderException("Failed to unpack float64"), ex)


def __decode_char(fp_read, marker, le=1):
    raw = fp_read(1)
    if not raw:
        raise DecoderException("Char missing")
    try:
        return raw.decode("utf-8")
    except UnicodeError as ex:
        raise_from(DecoderException("Failed to decode char"), ex)


def __decode_string(fp_read, marker, le=1):
    # current marker is string identifier, so read next byte which identifies integer type
    length = __decode_int_non_negative(fp_read, fp_read(1), le)
    raw = fp_read(length)
    if len(raw) < length:
        raise DecoderException("String too short")
    try:
        return raw.decode("utf-8")
    except UnicodeError as ex:
        raise_from(DecoderException("Failed to decode string"), ex)


# same as string, except there is no 'S' marker
def __decode_object_key(fp_read, marker, intern_object_keys, le=1):
    length = __decode_int_non_negative(fp_read, marker, le)
    raw = fp_read(length)
    if len(raw) < length:
        raise DecoderException("String too short")
    try:
        return (
            intern_unicode(raw.decode("utf-8"))
            if intern_object_keys
            else raw.decode("utf-8")
        )
    except UnicodeError as ex:
        raise_from(DecoderException("Failed to decode object key"), ex)


__METHOD_MAP = {
    TYPE_NULL: (lambda _, __, ___: None),
    TYPE_BOOL_TRUE: (lambda _, __, ___: True),
    TYPE_BOOL_FALSE: (lambda _, __, ___: False),
    TYPE_BYTE: __decode_byte,
    TYPE_INT8: __decode_int8,
    TYPE_UINT8: __decode_uint8,
    TYPE_INT16: __decode_int16,
    TYPE_UINT16: __decode_uint16,
    TYPE_INT32: __decode_int32,
    TYPE_UINT32: __decode_uint32,
    TYPE_INT64: __decode_int64,
    TYPE_UINT64: __decode_uint64,
    TYPE_FLOAT16: __decode_float16,
    TYPE_FLOAT32: __decode_float32,
    TYPE_FLOAT64: __decode_float64,
    TYPE_HIGH_PREC: __decode_high_prec,
    TYPE_CHAR: __decode_char,
    TYPE_STRING: __decode_string,
}


def prodlist(mylist):
    result = 1
    for x in mylist:
        result = result * x
    return result


def __decode_soa_schema(fp_read, intern_object_keys, le):
    """Decode SOA schema: {field1:type1, field2:type2, ...}

    Supports:
        - Fixed-length numeric types
        - Boolean (T marker)
        - Null (Z marker)
        - Fixed-length string: S<int><length>
        - Dictionary-based string: [$S#<n><str1><str2>...
        - Offset-table-based string: [$<int-type>]
        - Fixed-size array: [<type><type>...] (repeated markers)
    """
    schema = []
    marker = fp_read(1)

    while marker != OBJECT_END:
        if marker == TYPE_NOOP:
            marker = fp_read(1)
            continue

        field_name = __decode_object_key(fp_read, marker, intern_object_keys, le)
        type_marker = fp_read(1)

        if type_marker in __TYPES_FIXLEN:
            schema.append(
                {
                    "name": field_name,
                    "type": "numeric",
                    "marker": type_marker,
                    "bytes": __DTYPELEN_MAP[type_marker],
                }
            )
        elif type_marker in (TYPE_BOOL_TRUE, TYPE_BOOL_FALSE):
            schema.append(
                {"name": field_name, "type": "bool", "marker": type_marker, "bytes": 1}
            )
        elif type_marker == TYPE_NULL:
            schema.append({"name": field_name, "type": "null", "bytes": 0})
        elif type_marker == TYPE_STRING:
            length_marker = fp_read(1)
            if length_marker not in __TYPES_INT:
                raise DecoderException("SOA schema only supports fixed-length types")
            length = __decode_int_non_negative(fp_read, length_marker, le)
            schema.append(
                {
                    "name": field_name,
                    "type": "string",
                    "encoding": "fixed",
                    "bytes": length,
                }
            )
        elif type_marker == OBJECT_START:
            nested = __decode_soa_schema(fp_read, intern_object_keys, le)
            schema.append(
                {
                    "name": field_name,
                    "type": "nested",
                    "schema": nested,
                    "bytes": sum(f["bytes"] for f in nested),
                }
            )
        elif type_marker == ARRAY_START:
            schema.append(__decode_array_or_string_schema(fp_read, field_name, le))
        else:
            raise DecoderException("Unsupported SOA schema type")

        marker = fp_read(1)

    return schema


def __decode_array_or_string_schema(fp_read, field_name, le):
    """Decode array schema: fixed array [TTT], dict [$S#...], or offset [$U]."""
    next_marker = fp_read(1)

    if next_marker == CONTAINER_TYPE:
        inner = fp_read(1)
        if inner == TYPE_STRING:
            if fp_read(1) != CONTAINER_COUNT:
                raise DecoderException("Expected # in dict-string schema")
            dict_count = __decode_int_non_negative(fp_read, fp_read(1), le)
            dictionary = []
            for _ in range(dict_count):
                str_len = __decode_int_non_negative(fp_read, fp_read(1), le)
                dictionary.append(fp_read(str_len).decode("utf-8"))
            idx_bytes = 1 if dict_count <= 255 else (2 if dict_count <= 65535 else 4)
            idx_marker = (
                TYPE_UINT8
                if idx_bytes == 1
                else (TYPE_UINT16 if idx_bytes == 2 else TYPE_UINT32)
            )
            return {
                "name": field_name,
                "type": "string",
                "encoding": "dict",
                "bytes": idx_bytes,
                "dict": dictionary,
                "index_marker": idx_marker,
            }
        elif inner in __TYPES_INT:
            if fp_read(1) != ARRAY_END:
                raise DecoderException("Expected ] in offset-string schema")
            return {
                "name": field_name,
                "type": "string",
                "encoding": "offset",
                "bytes": __DTYPELEN_MAP[inner],
                "index_marker": inner,
            }
        else:
            raise DecoderException("Unsupported array schema type")
    elif next_marker in __TYPES_FIXLEN or next_marker in (
        TYPE_BOOL_TRUE,
        TYPE_BOOL_FALSE,
    ):
        elem_marker = next_marker
        elem_count = 1
        m = fp_read(1)
        while m != ARRAY_END:
            if m != elem_marker:
                raise DecoderException("Mixed types in fixed array not supported")
            elem_count += 1
            m = fp_read(1)
        is_bool = elem_marker in (TYPE_BOOL_TRUE, TYPE_BOOL_FALSE)
        return {
            "name": field_name,
            "type": "array",
            "marker": elem_marker,
            "count": elem_count,
            "elem_type": "bool" if is_bool else "numeric",
            "bytes": elem_count * (1 if is_bool else __DTYPELEN_MAP[elem_marker]),
        }
    else:
        raise DecoderException("Unsupported array schema type")


def __decode_soa_field_value(field, raw, record_index, le):
    """Decode a single field value from raw bytes."""
    ftype = field["type"]

    if ftype == "numeric":
        return buffer2numpy(raw, dtype=npdtype(__NUMPY_DTYPE_MAP[field["marker"]]))[0]
    elif ftype == "bool":
        return raw[0:1] == TYPE_BOOL_TRUE
    elif ftype == "null":
        return None
    elif ftype == "array":
        if field["elem_type"] == "bool":
            return ndarray(
                [raw[i : i + 1] == TYPE_BOOL_TRUE for i in range(field["count"])]
            )
        return buffer2numpy(raw, dtype=npdtype(__NUMPY_DTYPE_MAP[field["marker"]]))
    elif ftype == "nested":
        result = {}
        offset = 0
        for f in field["schema"]:
            result[f["name"]] = __decode_soa_field_value(
                f, raw[offset : offset + f["bytes"]], record_index, le
            )
            offset += f["bytes"]
        return result
    elif ftype == "string":
        enc = field["encoding"]
        if enc == "fixed":
            return raw.rstrip(b"\x00").decode("utf-8")
        elif enc == "dict":
            idx = buffer2numpy(
                raw, dtype=npdtype(__NUMPY_DTYPE_MAP[field["index_marker"]])
            )[0].item()
            return field["dict"][idx]
        elif enc == "offset":
            idx = buffer2numpy(
                raw, dtype=npdtype(__NUMPY_DTYPE_MAP[field["index_marker"]])
            )[0].item()
            return field["string_buffer"][
                field["offsets"][idx] : field["offsets"][idx + 1]
            ]
    return None


def __schema_to_dtype(schema):
    """Build numpy dtype from SOA schema."""
    dtype_list = []
    for f in schema:
        fname = f["name"]
        ftype = f["type"]
        if ftype == "numeric":
            dtype_list.append((fname, __NUMPY_DTYPE_MAP[f["marker"]]))
        elif ftype == "bool":
            dtype_list.append((fname, "?"))
        elif ftype == "null":
            dtype_list.append((fname, "O"))
        elif ftype == "array":
            dt = "?" if f["elem_type"] == "bool" else __NUMPY_DTYPE_MAP[f["marker"]]
            dtype_list.append((fname, dt, (f["count"],)))
        elif ftype == "nested":
            dtype_list.append((fname, __schema_to_dtype(f["schema"])))
        elif ftype == "string" and f["encoding"] == "fixed":
            dtype_list.append((fname, "U" + str(f["bytes"])))
    return npdtype(dtype_list)


def __decode_soa(fp_read, schema, is_row_major, intern_object_keys, le):
    """Decode SOA payload into numpy structured array."""
    if fp_read(1) != CONTAINER_COUNT:
        raise DecoderException("Expected # after SOA schema")
    marker = fp_read(1)
    if marker == ARRAY_START:
        dims = []
        m = fp_read(1)
        while m != ARRAY_END:
            dims.append(__METHOD_MAP[m](fp_read, m, le))
            m = fp_read(1)
        count = prodlist(dims)
    else:
        count = __decode_int_non_negative(fp_read, marker, le)
        dims = [count]
    record_bytes = sum(f["bytes"] for f in schema)
    payload = fp_read(record_bytes * count)
    __read_offset_tables(fp_read, schema, count, le)

    # Convert variable strings through dict intermediate, then to numpy
    if __has_variable_strings(schema):
        records = __decode_soa_to_dicts(
            payload, schema, count, record_bytes, is_row_major, le
        )
        return __dicts_to_numpy(records, schema, dims)

    result = npempty(count, dtype=__schema_to_dtype(schema))
    __fill_numpy_result(result, payload, schema, count, record_bytes, is_row_major, le)
    if len(dims) > 1:
        result = result.reshape(dims)
    return result


def __read_offset_tables(fp_read, schema, count, le):
    """Recursively read offset tables for offset-based string fields."""
    for f in schema:
        if f.get("encoding") == "offset":
            f["offsets"] = []
            for _ in range(count + 1):
                f["offsets"].append(
                    __METHOD_MAP[f["index_marker"]](fp_read, f["index_marker"], le)
                )
            buf_len = f["offsets"][-1] if f["offsets"] else 0
            f["string_buffer"] = fp_read(buf_len).decode("utf-8") if buf_len else ""
        elif f.get("type") == "nested":
            __read_offset_tables(fp_read, f["schema"], count, le)


def __has_variable_strings(schema):
    """Check if schema has dict/offset strings."""
    for f in schema:
        if f.get("type") == "string" and f.get("encoding") in ("dict", "offset"):
            return True
        if f.get("type") == "nested" and __has_variable_strings(f["schema"]):
            return True
    return False


def __fill_numpy_result(result, payload, schema, count, record_bytes, is_row_major, le):
    """Fill numpy structured array from payload."""
    if is_row_major:
        for i in range(count):
            base = i * record_bytes
            offset = 0
            for f in schema:
                __set_field_value(
                    result,
                    f["name"],
                    f,
                    i,
                    payload[base + offset : base + offset + f["bytes"]],
                    le,
                )
                offset += f["bytes"]
    else:
        col_offset = 0
        for f in schema:
            for i in range(count):
                offset = col_offset + i * f["bytes"]
                __set_field_value(
                    result, f["name"], f, i, payload[offset : offset + f["bytes"]], le
                )
            col_offset += f["bytes"] * count


def __set_field_value(result, name, field, index, raw, le):
    """Set a field value in numpy result, handling nested structs."""
    if field["type"] == "nested":
        offset = 0
        for f in field["schema"]:
            __set_field_value(
                result[name], f["name"], f, index, raw[offset : offset + f["bytes"]], le
            )
            offset += f["bytes"]
    else:
        result[name][index] = __decode_soa_field_value(field, raw, index, le)


def __decode_soa_to_dicts(payload, schema, count, record_bytes, is_row_major, le):
    """Decode SOA to list of dicts (for variable-length strings)."""
    records = []
    for i in range(count):
        record = {}
        if is_row_major:
            base = i * record_bytes
            offset = 0
            for f in schema:
                record[f["name"]] = __decode_soa_field_value(
                    f, payload[base + offset : base + offset + f["bytes"]], i, le
                )
                offset += f["bytes"]
        else:
            col_offset = 0
            for f in schema:
                offset = col_offset + i * f["bytes"]
                record[f["name"]] = __decode_soa_field_value(
                    f, payload[offset : offset + f["bytes"]], i, le
                )
                col_offset += f["bytes"] * count
        records.append(record)
    return records


def __get_max_string_len(records, field_name):
    """Get maximum string length for a field across all records."""
    max_len = 1
    for r in records:
        val = r.get(field_name, "")
        if isinstance(val, str):
            max_len = max(max_len, len(val))
        elif isinstance(val, dict):
            for k in val:
                nested_max = __get_max_string_len_nested(val, k)
                max_len = max(max_len, nested_max)
    return max_len


def __get_max_string_len_nested(record, field_name):
    """Get maximum string length for nested field."""
    val = record.get(field_name, "")
    if isinstance(val, str):
        return len(val)
    return 1


# --- NEW FUNCTION ---
def __schema_to_dtype_with_strings(schema, records):
    """Build numpy dtype from SOA schema, using actual string lengths from data."""
    dtype_list = []
    for f in schema:
        fname = f["name"]
        ftype = f["type"]
        if ftype == "numeric":
            dtype_list.append((fname, __NUMPY_DTYPE_MAP[f["marker"]]))
        elif ftype == "bool":
            dtype_list.append((fname, "?"))
        elif ftype == "null":
            dtype_list.append((fname, "O"))
        elif ftype == "array":
            dt = "?" if f["elem_type"] == "bool" else __NUMPY_DTYPE_MAP[f["marker"]]
            dtype_list.append((fname, dt, (f["count"],)))
        elif ftype == "nested":
            nested_records = [r.get(fname, {}) for r in records]
            nested_dt = __schema_to_dtype_with_strings(f["schema"], nested_records)
            dtype_list.append((fname, nested_dt))
        elif ftype == "string":
            max_len = max(1, max(len(r.get(fname, "")) for r in records))
            dtype_list.append((fname, f"U{max_len}"))
    return npdtype(dtype_list)


# --- NEW FUNCTION ---
def __dicts_to_numpy(records, schema, dims):
    """Convert list of dicts to numpy structured array."""
    count = len(records)
    if count == 0:
        return npempty(0, dtype=__schema_to_dtype(schema))

    struct_dtype = __schema_to_dtype_with_strings(schema, records)
    result = npempty(count, dtype=struct_dtype)

    for i, record in enumerate(records):
        __set_record_value(result, record, schema, i)

    if len(dims) > 1:
        result = result.reshape(dims)

    return result


# --- NEW FUNCTION ---
def __set_record_value(result, record, schema, index):
    """Set a record's values in numpy result."""
    for f in schema:
        fname = f["name"]
        val = record.get(fname)
        if f["type"] == "nested" and isinstance(val, dict):
            __set_record_value(result[fname], val, f["schema"], index)
        else:
            result[fname][index] = val


def __get_container_params(
    fp_read,
    in_mapping,
    no_bytes,
    uint8_bytes,
    object_hook,
    object_pairs_hook,
    intern_object_keys,
    islittle,
):
    marker = fp_read(1)
    dims = []
    if marker == CONTAINER_TYPE:
        marker = fp_read(1)

        # Check for SOA: ${ indicates schema object
        if marker == OBJECT_START:
            # This is SOA format - decode schema
            schema = __decode_soa_schema(fp_read, intern_object_keys, islittle)
            return marker, True, -1, schema, [], True  # -1 count signals SOA

        if marker not in __TYPES:
            raise DecoderException("Invalid container type")
        type_ = marker
        marker = fp_read(1)
    else:
        type_ = TYPE_NONE
    if marker == CONTAINER_COUNT:
        marker = fp_read(1)
        if marker == ARRAY_START:
            dims = __decode_array(
                fp_read,
                no_bytes,
                uint8_bytes,
                object_hook,
                object_pairs_hook,
                intern_object_keys,
                islittle,
            )
            count = prodlist(dims)
        else:
            count = __decode_int_non_negative(fp_read, marker, islittle)
        counting = True

        # special cases (no data (None or bool) / bytes array) will be handled in calling functions
        if not (
            type_ in __TYPES_NO_DATA
            or (
                type_ == (TYPE_UINT8 if uint8_bytes else TYPE_BYTE)
                and not in_mapping
                and not no_bytes
            )
        ):
            # Reading ahead is just to capture type, which will not exist if type is fixed
            marker = fp_read(1) if (in_mapping or type_ == TYPE_NONE) else type_

    elif type_ == TYPE_NONE:
        # set to one to indicate that not finished yet
        count = 1
        counting = False
    else:
        raise DecoderException("Container type without count")
    return marker, counting, count, type_, dims, False


def __decode_object(
    fp_read,
    no_bytes,
    uint8_bytes,
    object_hook,
    object_pairs_hook,  # pylint: disable=too-many-branches
    intern_object_keys,
    islittle,
):
    result = __get_container_params(
        fp_read,
        True,
        no_bytes,
        uint8_bytes,
        object_hook,
        object_pairs_hook,
        intern_object_keys,
        islittle,
    )

    # Check if this is SOA format
    if len(result) == 6 and result[5]:  # is_soa flag
        schema = result[3]
        return __decode_soa(fp_read, schema, False, intern_object_keys, islittle)

    marker, counting, count, type_, dims, _ = result
    has_pairs_hook = object_pairs_hook is not None
    obj = [] if has_pairs_hook else {}

    le = islittle

    # special case - no data (None or bool)
    if type_ in __TYPES_NO_DATA:
        value = __METHOD_MAP[type_](fp_read, type_, le)
        if has_pairs_hook:
            for _ in range(count):
                obj.append(
                    (
                        __decode_object_key(
                            fp_read, fp_read(1), intern_object_keys, le
                        ),
                        value,
                    )
                )
            return object_pairs_hook(obj)

        for _ in range(count):
            obj[
                __decode_object_key(fp_read, fp_read(1), intern_object_keys, le)
            ] = value
        return object_hook(obj)

    while count > 0 and (counting or marker != OBJECT_END):
        if marker == TYPE_NOOP:
            marker = fp_read(1)
            continue

        # decode key for object
        key = __decode_object_key(fp_read, marker, intern_object_keys, le)
        marker = fp_read(1) if type_ == TYPE_NONE else type_

        # decode value
        try:
            value = __METHOD_MAP[marker](fp_read, marker, islittle)
        except KeyError:
            handled = False
        else:
            handled = True

        # handle outside above except (on KeyError) so do not have unfriendly "exception within except" backtrace
        if not handled:
            if marker == ARRAY_START:
                value = __decode_array(
                    fp_read,
                    no_bytes,
                    uint8_bytes,
                    object_hook,
                    object_pairs_hook,
                    intern_object_keys,
                    islittle,
                )
            elif marker == OBJECT_START:
                value = __decode_object(
                    fp_read,
                    no_bytes,
                    uint8_bytes,
                    object_hook,
                    object_pairs_hook,
                    intern_object_keys,
                    islittle,
                )
            else:
                raise DecoderException("Invalid marker within object")

        if has_pairs_hook:
            obj.append((key, value))
        else:
            obj[key] = value
        if counting:
            count -= 1
        if count > 0:
            marker = fp_read(1)

    return object_pairs_hook(obj) if has_pairs_hook else object_hook(obj)


def __decode_array(
    fp_read,
    no_bytes,
    uint8_bytes,
    object_hook,
    object_pairs_hook,
    intern_object_keys,
    islittle,
):
    result = __get_container_params(
        fp_read,
        False,
        no_bytes,
        uint8_bytes,
        object_hook,
        object_pairs_hook,
        intern_object_keys,
        islittle,
    )

    # Check if this is SOA format (row-major)
    if len(result) == 6 and result[5]:  # is_soa flag
        schema = result[3]
        return __decode_soa(fp_read, schema, True, intern_object_keys, islittle)

    marker, counting, count, type_, dims, _ = result

    # special case - no data (None or bool)
    if type_ in __TYPES_NO_DATA:
        return [__METHOD_MAP[type_](fp_read, type_, islittle)] * count

    # special case - bytes array
    if (
        type_ == (TYPE_UINT8 if uint8_bytes else TYPE_BYTE)
        and not no_bytes
        and len(dims) == 0
    ):
        container = fp_read(count)
        if len(container) < count:
            raise DecoderException("Container bytes array too short")
        return container

    if type_ in __TYPES_FIXLEN and count > 0:
        if hasattr(count, "dtype"):
            container = fp_read(count.item() * __DTYPELEN_MAP[type_])
        else:
            container = fp_read(count * __DTYPELEN_MAP[type_])
        if len(container) < count * __DTYPELEN_MAP[type_]:
            raise DecoderException("Container bytes array too short")

        # container=typedarray(__DTYPE_MAP[type_], container)
        if len(dims) > 0:
            container = buffer2numpy(container, dtype=npdtype(__DTYPE_MAP[type_]))
            container = container.reshape(dims)
        else:
            container = buffer2numpy(container, dtype=npdtype(__DTYPE_MAP[type_]))
        return container

    container = list()
    while count > 0 and (counting or marker != ARRAY_END):
        if marker == TYPE_NOOP:
            marker = fp_read(1)
            continue

        # decode value
        try:
            value = __METHOD_MAP[marker](fp_read, marker, islittle)
        except KeyError:
            handled = False
        else:
            handled = True

        # handle outside above except (on KeyError) so do not have unfriendly "exception within except" backtrace
        if not handled:
            if marker == ARRAY_START:
                value = __decode_array(
                    fp_read,
                    no_bytes,
                    uint8_bytes,
                    object_hook,
                    object_pairs_hook,
                    intern_object_keys,
                    islittle,
                )
            elif marker == OBJECT_START:
                value = __decode_object(
                    fp_read,
                    no_bytes,
                    uint8_bytes,
                    object_hook,
                    object_pairs_hook,
                    intern_object_keys,
                    islittle,
                )
            else:
                raise DecoderException("Invalid marker within array")

        container.append(value)
        if counting:
            count -= 1
        if count and type_ == TYPE_NONE:
            marker = fp_read(1)

    if len(dims) > 0:
        container = list(
            reduce(
                lambda x, y: map(list, zip(*y * (x,))),
                (iter(container),) + tuple(dims[:0:-1]),
            )
        )
        container = ndarray(container, dtype=npdtype(__DTYPE_MAP[type_]))

    return container


def __object_hook_noop(obj):
    return obj


def load(
    fp,
    no_bytes=False,
    uint8_bytes=False,
    object_hook=None,
    object_pairs_hook=None,
    intern_object_keys=False,
    islittle=True,
):
    """Decodes and returns BJData/UBJSON from the given file-like object

    Args:
        fp: read([size])-able object
        no_bytes (bool): If set, typed UBJSON arrays (byte) will not be
                         converted to a bytes instance and instead treated like
                         any other array (i.e. result in a list).
        uint8_bytes (bool): If set, typed UBJSON arrays (uint8) will be
                         converted to a bytes instance instead of being
                         treated as an array (for UBJSON & BJData Draft 4).
                         Ignored if no_bytes is set.
        object_hook (callable): Called with the result of any object literal
                                decoded (instead of dict).
        object_pairs_hook (callable): Called with the result of any object
                                      literal decoded with an ordered list of
                                      pairs (instead of dict). Takes precedence
                                      over object_hook.
        intern_object_keys (bool): If set, object keys are interned which can
                                   provide a memory saving when many repeated
                                   keys are used. NOTE: This is not supported
                                   in Python2 (since interning does not apply
                                   to unicode) and wil be ignored.
        islittle (1 or 0): default is 1 for little-endian for all numerics (for
                            BJData Draft 4), change to 0 to use big-endian
                            (for UBJSON & BJData Draft 1)

    Returns:
        Decoded object

    Raises:
        DecoderException: If an encoding failure occured.

    BJData/UBJSON types are mapped to Python types as follows.  Numbers in
    brackets denote Python version.

        +----------------------------------+---------------+
        | BJData/UBJSON                    | Python        |
        +==================================+===============+
        | object                           | dict          |
        +----------------------------------+---------------+
        | array                            | list          |
        +----------------------------------+---------------+
        | string                           | (3) str       |
        |                                  | (2) unicode   |
        +----------------------------------+---------------+
        | uint8, int8, int16, int32, int64 | (3) int       |
        | byte                             | (2) int, long |
        +----------------------------------+---------------+
        | float32, float64                 | float         |
        +----------------------------------+---------------+
        | high_precision                   | Decimal       |
        +----------------------------------+---------------+
        | array (typed, byte)              | (3) bytes     |
        |                                  | (2) str       |
        +----------------------------------+---------------+
        | true                             | True          |
        +----------------------------------+---------------+
        | false                            | False         |
        +----------------------------------+---------------+
        | null                             | None          |
        +----------------------------------+---------------+

    SOA (Structure of Arrays) format is automatically detected and decoded
    to a list of dicts. String fields with fixed, dict, or offset encoding
    are all supported.
    """
    if object_pairs_hook is None and object_hook is None:
        object_hook = __object_hook_noop

    if not callable(fp.read):
        raise TypeError("fp.read not callable")
    fp_read = fp.read

    newobj = []

    while True:
        marker = fp_read(1)
        if len(marker) == 0:
            break
        try:
            try:
                return __METHOD_MAP[marker](fp_read, marker, islittle)
            except KeyError:
                pass
            if marker == ARRAY_START:
                newobj.append(
                    __decode_array(
                        fp_read,
                        bool(no_bytes),
                        bool(uint8_bytes),
                        object_hook,
                        object_pairs_hook,
                        intern_object_keys,
                        islittle,
                    )
                )
            if marker == OBJECT_START:
                newobj.append(
                    __decode_object(
                        fp_read,
                        bool(no_bytes),
                        bool(uint8_bytes),
                        object_hook,
                        object_pairs_hook,
                        intern_object_keys,
                        islittle,
                    )
                )
            raise DecoderException("Invalid marker")
        except DecoderException as ex:
            if len(newobj) > 0:
                pass
            else:
                raise_from(
                    DecoderException(
                        ex.args[0],
                        position=(fp.tell() if hasattr(fp, "tell") else None),
                    ),
                    ex,
                )
    if len(newobj) == 1:
        newobj = newobj[0]
    elif len(newobj) == 0:
        raise DecoderException("Empty data")

    return newobj


def loadb(
    chars,
    no_bytes=False,
    uint8_bytes=False,
    object_hook=None,
    object_pairs_hook=None,
    intern_object_keys=False,
    islittle=True,
):
    """Decodes and returns BJData/UBJSON from the given bytes or bytesarray object. See
    load() for available arguments."""
    with BytesIO(chars) as fp:
        return load(
            fp,
            no_bytes=no_bytes,
            uint8_bytes=uint8_bytes,
            object_hook=object_hook,
            object_pairs_hook=object_pairs_hook,
            intern_object_keys=intern_object_keys,
            islittle=islittle,
        )
