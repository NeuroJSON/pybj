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


"""BJData (Draft 4) and UBJSON encoder with SOA support"""

from struct import pack, Struct
from decimal import Decimal
from io import BytesIO
from math import isinf, isnan
from itertools import accumulate

from .compat import (
    Mapping,
    Sequence,
    INTEGER_TYPES,
    UNICODE_TYPE,
    TEXT_TYPES,
    BYTES_TYPES,
)
from .markers import (
    TYPE_NULL,
    TYPE_BOOL_TRUE,
    TYPE_BOOL_FALSE,
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
    TYPE_HIGH_PREC,
    TYPE_CHAR,
    TYPE_STRING,
    OBJECT_START,
    OBJECT_END,
    ARRAY_START,
    ARRAY_END,
    CONTAINER_TYPE,
    CONTAINER_COUNT,
)

# Lookup tables for encoding small intergers, pre-initialised larger integer & float packers
__SMALL_INTS_ENCODED = [
    {i: TYPE_INT8 + pack(">b", i) for i in range(-128, 128)},
    {i: TYPE_INT8 + pack("<b", i) for i in range(-128, 128)},
]
__SMALL_UINTS_ENCODED = [
    {i: TYPE_UINT8 + pack(">B", i) for i in range(256)},
    {i: TYPE_UINT8 + pack("<B", i) for i in range(256)},
]
__PACK_INT16 = [Struct(">h").pack, Struct("<h").pack]
__PACK_INT32 = [Struct(">i").pack, Struct("<i").pack]
__PACK_INT64 = [Struct(">q").pack, Struct("<q").pack]
__PACK_UINT16 = [Struct(">H").pack, Struct("<H").pack]
__PACK_UINT32 = [Struct(">I").pack, Struct("<I").pack]
__PACK_UINT64 = [Struct(">Q").pack, Struct("<Q").pack]
__PACK_FLOAT16 = [Struct(">h").pack, Struct("<h").pack]
__PACK_FLOAT32 = [Struct(">f").pack, Struct("<f").pack]
__PACK_FLOAT64 = [Struct(">d").pack, Struct("<d").pack]

__DTYPE_TO_MARKER = {
    "i1": TYPE_INT8,
    "i2": TYPE_INT16,
    "i4": TYPE_INT32,
    "i8": TYPE_INT64,
    "u1": TYPE_UINT8,
    "u2": TYPE_UINT16,
    "u4": TYPE_UINT32,
    "u8": TYPE_UINT64,
    "f2": TYPE_FLOAT16,
    "f4": TYPE_FLOAT32,
    "f8": TYPE_FLOAT64,
    "b1": TYPE_INT8,
    "S1": TYPE_CHAR,
}

# For SOA encoding
__NUMPY_DTYPE_TO_MARKER = {
    "i1": TYPE_INT8,
    "i2": TYPE_INT16,
    "i4": TYPE_INT32,
    "i8": TYPE_INT64,
    "u1": TYPE_UINT8,
    "u2": TYPE_UINT16,
    "u4": TYPE_UINT32,
    "u8": TYPE_UINT64,
    "f2": TYPE_FLOAT16,
    "f4": TYPE_FLOAT32,
    "f8": TYPE_FLOAT64,
    "b1": TYPE_BOOL_TRUE,  # Boolean uses T marker in schema
    "?": TYPE_BOOL_TRUE,  # Boolean
}

# Prefix applicable to specialised byte array container
__BYTES_ARRAY_PREFIX = ARRAY_START + CONTAINER_TYPE + TYPE_BYTE + CONTAINER_COUNT
__BYTES_ARRAY_PREFIX_DRAFT2 = (
    ARRAY_START + CONTAINER_TYPE + TYPE_UINT8 + CONTAINER_COUNT
)


class EncoderException(TypeError):
    """Raised when encoding of an object fails."""


def __encode_decimal(fp_write, item, le=1):
    if item.is_finite():
        fp_write(TYPE_HIGH_PREC)
        encoded_val = str(item).encode("utf-8")
        __encode_int(fp_write, len(encoded_val), le)
        fp_write(encoded_val)
    else:
        fp_write(TYPE_NULL)


def __encode_int(fp_write, item, le=1):
    if item >= 0:
        if item < 2**8:
            fp_write(__SMALL_UINTS_ENCODED[le][item])
        elif item < 2**16:
            fp_write(TYPE_UINT16)
            fp_write(__PACK_UINT16[le](item))
        elif item < 2**32:
            fp_write(TYPE_UINT32)
            fp_write(__PACK_UINT32[le](item))
        elif item < 2**64:
            fp_write(TYPE_UINT64)
            fp_write(__PACK_UINT64[le](item))
        else:
            __encode_decimal(fp_write, Decimal(item), le)
    elif item >= -(2**7):
        fp_write(__SMALL_INTS_ENCODED[le][item])
    elif item >= -(2**15):
        fp_write(TYPE_INT16)
        fp_write(__PACK_INT16[le](item))
    elif item >= -(2**31):
        fp_write(TYPE_INT32)
        fp_write(__PACK_INT32[le](item))
    elif item >= -(2**63):
        fp_write(TYPE_INT64)
        fp_write(__PACK_INT64[le](item))
    else:
        __encode_decimal(fp_write, Decimal(item), le)


def __encode_float(fp_write, item, le=1):
    if 1.18e-38 <= abs(item) <= 3.4e38 or item == 0:
        fp_write(TYPE_FLOAT32)
        fp_write(__PACK_FLOAT32[le](item))
    elif 2.23e-308 <= abs(item) < 1.8e308:
        fp_write(TYPE_FLOAT64)
        fp_write(__PACK_FLOAT64[le](item))
    elif isinf(item) or isnan(item):
        fp_write(TYPE_FLOAT32)
        fp_write(__PACK_FLOAT32[le](item))
    else:
        __encode_decimal(fp_write, Decimal(item), le)


def __encode_float64(fp_write, item, le=1):
    if 2.23e-308 <= abs(item) < 1.8e308:
        fp_write(TYPE_FLOAT64)
        fp_write(__PACK_FLOAT64[le](item))
    elif item == 0:
        fp_write(TYPE_FLOAT32)
        fp_write(__PACK_FLOAT32[le](item))
    elif isinf(item) or isnan(item):
        fp_write(TYPE_FLOAT64)
        fp_write(__PACK_FLOAT64[le](item))
    else:
        __encode_decimal(fp_write, Decimal(item), le)


def __encode_string(fp_write, item, le=1):
    encoded_val = item.encode("utf-8")
    length = len(encoded_val)
    if length == 1:
        fp_write(TYPE_CHAR)
    else:
        fp_write(TYPE_STRING)
        if length < 2**8:
            fp_write(__SMALL_UINTS_ENCODED[le][length])
        else:
            __encode_int(fp_write, length, le)
    fp_write(encoded_val)


def __encode_bytes(fp_write, item, uint8_bytes, le=1):
    fp_write(__BYTES_ARRAY_PREFIX_DRAFT2 if uint8_bytes else __BYTES_ARRAY_PREFIX)
    length = len(item)
    if length < 2**8:
        fp_write(__SMALL_UINTS_ENCODED[le][length])
    else:
        __encode_int(fp_write, length, le)
    fp_write(item)
    # no ARRAY_END since length was specified


def __get_numpy_dtype_marker(dtype_str):
    """Get BJData type marker from numpy dtype string"""
    # Handle endianness prefix
    if len(dtype_str) >= 2 and dtype_str[0] in "<>|":
        dtype_str = dtype_str[1:]

    if dtype_str in __NUMPY_DTYPE_TO_MARKER:
        return __NUMPY_DTYPE_TO_MARKER[dtype_str]

    # Handle boolean
    if dtype_str.startswith("b") or dtype_str == "?":
        return TYPE_BOOL_TRUE

    return None


def __analyze_string_field(values, soa_threshold=None):
    """Analyze string field to determine best encoding: fixed, dict, or offset.

    Args:
        values: List of string values
        soa_threshold: If 0, force offset encoding. If None, auto-select.
                      Otherwise, use as dict threshold ratio.

    Returns:
        tuple: (encoding_type, param, extra_data)
            - 'fixed': param is max_length, extra_data is None
            - 'dict': param is index_bytes, extra_data is list of unique values
            - 'offset': param is offset_bytes, extra_data is None
    """
    if not values:
        return "fixed", 1, None

    # Force offset encoding if threshold is 0
    if soa_threshold == 0:
        total_len = sum(len(v.encode("utf-8")) for v in values)
        off_bytes = 1 if total_len <= 255 else (2 if total_len <= 65535 else 4)
        return "offset", off_bytes, None

    unique_values = list(dict.fromkeys(values))
    max_len = max(len(v.encode("utf-8")) for v in values)
    total_len = sum(len(v.encode("utf-8")) for v in values)

    count = len(values)
    num_unique = len(unique_values)

    # Fixed: max_len * count
    fixed_cost = max_len * count

    # Dict: index_bytes * count + dict_overhead
    idx_bytes = 1 if num_unique <= 255 else (2 if num_unique <= 65535 else 4)
    dict_overhead = sum(len(v.encode("utf-8")) + 2 for v in unique_values)
    dict_cost = idx_bytes * count + dict_overhead

    # Offset: index_bytes * count + (count+1) * offset_bytes + total_len
    off_bytes = 1 if total_len <= 255 else (2 if total_len <= 65535 else 4)
    offset_cost = idx_bytes * count + (count + 1) * off_bytes + total_len

    # Use custom threshold or default 0.3
    dict_threshold = soa_threshold if soa_threshold is not None else 0.3

    if (
        num_unique <= count * dict_threshold
        and dict_cost < fixed_cost
        and dict_cost < offset_cost
    ):
        return "dict", idx_bytes, unique_values
    elif max_len > 32 and offset_cost < fixed_cost:
        return "offset", off_bytes, None
    else:
        return "fixed", max_len, None


def __can_encode_as_soa(item):
    """Check if numpy structured array or list of dicts can be encoded as SOA"""
    try:
        import numpy as np
    except ImportError:
        return False
    if isinstance(item, np.ndarray):
        if item.dtype.names is None:
            return False
        return __check_soa_dtype(item.dtype)
    if isinstance(item, (list, tuple)) and len(item) > 0:
        if all(isinstance(x, dict) for x in item):
            keys = set(item[0].keys())
            return all(set(x.keys()) == keys for x in item)
    return False


def __check_soa_dtype(dtype):
    """Recursively check if dtype fields are SOA-compatible"""
    for name in dtype.names:
        field_dtype = dtype.fields[name][0]
        if field_dtype.names is not None:
            if not __check_soa_dtype(field_dtype):
                return False
            continue
        base = field_dtype.base if field_dtype.subdtype else field_dtype
        dstr = base.str[1:] if base.str[0] in "<>|" else base.str
        if not (
            dstr.startswith("U")
            or dstr.startswith("S")
            or __get_numpy_dtype_marker(dstr) is not None
        ):
            return False
    return True


def __encode_soa_schema_field(fp_write, field_name, field_info, le):
    """Write a single field definition in SOA schema."""
    encoded_name = field_name.encode("utf-8")
    __encode_int(fp_write, len(encoded_name), le)
    fp_write(encoded_name)

    ftype = field_info["type"]
    if ftype == "numeric":
        fp_write(field_info["marker"])
    elif ftype == "bool":
        fp_write(TYPE_BOOL_TRUE)
    elif ftype == "null":
        fp_write(TYPE_NULL)
    elif ftype == "array":
        fp_write(ARRAY_START)
        for _ in range(field_info["count"]):
            fp_write(field_info["marker"])
        fp_write(ARRAY_END)
    elif ftype == "nested":
        fp_write(OBJECT_START)
        for f in field_info["schema"]:
            __encode_soa_schema_field(fp_write, f["name"], f, le)
        fp_write(OBJECT_END)
    elif ftype == "string":
        enc = field_info["encoding"]
        if enc == "fixed":
            fp_write(TYPE_STRING)
            __encode_int(fp_write, field_info["length"], le)
        elif enc == "dict":
            fp_write(ARRAY_START + CONTAINER_TYPE + TYPE_STRING + CONTAINER_COUNT)
            __encode_int(fp_write, len(field_info["dict"]), le)
            for s in field_info["dict"]:
                enc_s = s.encode("utf-8")
                __encode_int(fp_write, len(enc_s), le)
                fp_write(enc_s)
        elif enc == "offset":
            fp_write(
                ARRAY_START + CONTAINER_TYPE + field_info["index_marker"] + ARRAY_END
            )


def __write_soa_field_value(fp_write, field, index, le):
    """Write a single field value in SOA payload."""
    ftype, values = field["type"], field["values"]

    if ftype == "numeric":
        if hasattr(values, "tobytes"):
            fp_write(values[index].tobytes())
        else:
            __write_numeric_value(fp_write, field["marker"], values[index], le)
    elif ftype == "bool":
        fp_write(TYPE_BOOL_TRUE if values[index] else TYPE_BOOL_FALSE)
    elif ftype == "null":
        pass
    elif ftype == "array":
        fp_write(values[index].tobytes())
    elif ftype == "nested":
        for f in field["schema"]:
            __write_soa_field_value(fp_write, f, index, le)
    elif ftype == "string":
        enc, val = field["encoding"], values[index]
        if enc == "fixed":
            encoded = val.encode("utf-8")
            fp_write((encoded + b"\x00" * field["length"])[: field["length"]])
        elif enc == "dict":
            __write_index(fp_write, field["index_marker"], field["dict"].index(val), le)
        elif enc == "offset":
            __write_index(fp_write, field["index_marker"], index, le)


def __write_numeric_value(fp_write, marker, val, le):
    """Write a numeric value based on marker type."""
    val = val if val is not None else 0
    if marker == TYPE_INT8:
        fp_write(pack("<b" if le else ">b", val))
    elif marker == TYPE_UINT8:
        fp_write(pack("<B" if le else ">B", val))
    elif marker == TYPE_INT16:
        fp_write(__PACK_INT16[le](val))
    elif marker == TYPE_UINT16:
        fp_write(__PACK_UINT16[le](val))
    elif marker == TYPE_INT32:
        fp_write(__PACK_INT32[le](val))
    elif marker == TYPE_UINT32:
        fp_write(__PACK_UINT32[le](val))
    elif marker == TYPE_INT64:
        fp_write(__PACK_INT64[le](val))
    elif marker == TYPE_UINT64:
        fp_write(__PACK_UINT64[le](val))
    elif marker == TYPE_FLOAT32:
        fp_write(__PACK_FLOAT32[le](val))
    elif marker == TYPE_FLOAT64:
        fp_write(__PACK_FLOAT64[le](val))


def __write_index(fp_write, marker, idx, le):
    """Write an index value for dict/offset string encoding."""
    if marker == TYPE_UINT8:
        fp_write(pack("<B" if le else ">B", idx))
    elif marker == TYPE_UINT16:
        fp_write(__PACK_UINT16[le](idx))
    else:
        fp_write(__PACK_UINT32[le](idx))


def __build_field_schema(field_dtype, values, count, soa_threshold=None):
    """Build schema for a single field (handles nested/array/string/numeric)."""
    if field_dtype.names is not None:
        nested_schema = []
        for n in field_dtype.names:
            inner = __build_field_schema(
                field_dtype.fields[n][0], values[n], count, soa_threshold
            )
            inner["name"] = n
            nested_schema.append(inner)
        return {"type": "nested", "schema": nested_schema, "values": values}

    if field_dtype.subdtype:
        base, shape = field_dtype.subdtype
        dstr = base.str[1:] if base.str[0] in "<>|" else base.str
        return {
            "type": "array",
            "marker": __get_numpy_dtype_marker(dstr),
            "count": shape[0],
            "values": values,
        }

    dstr = field_dtype.str[1:] if field_dtype.str[0] in "<>|" else field_dtype.str

    if dstr.startswith("U") or dstr.startswith("S"):
        str_vals = [str(values[i]) for i in range(count)]
        enc_type, enc_param, enc_dict = __analyze_string_field(str_vals, soa_threshold)
        idx_marker = (
            TYPE_UINT8
            if enc_param == 1
            else (TYPE_UINT16 if enc_param == 2 else TYPE_UINT32)
        )
        return {
            "type": "string",
            "encoding": enc_type,
            "length": enc_param,
            "dict": enc_dict,
            "index_marker": idx_marker,
            "values": str_vals,
        }

    if dstr in ("?", "b1"):
        return {"type": "bool", "values": [bool(values[i]) for i in range(count)]}

    return {
        "type": "numeric",
        "marker": __get_numpy_dtype_marker(dstr),
        "values": values,
    }


def __collect_offset_fields(schema):
    """Recursively collect offset-based string fields."""
    result = []
    for f in schema:
        if f.get("encoding") == "offset":
            result.append(f)
        if f.get("type") == "nested":
            result.extend(__collect_offset_fields(f["schema"]))
    return result


def __encode_soa(fp_write, item, soa_format, le, soa_threshold=None):
    """Encode numpy structured array or list of dicts as SOA format."""
    import numpy as np
    from itertools import accumulate

    is_row_major = soa_format in ("row", "r")

    if isinstance(item, np.ndarray):
        count = item.size
        dims = item.shape
        flat = item.flatten()
        schema = []
        for n in item.dtype.names:
            fs = __build_field_schema(
                item.dtype.fields[n][0], flat[n], count, soa_threshold
            )
            fs["name"] = n
            schema.append(fs)
    else:
        count = len(item)
        dims = [count]
        schema = __build_dict_schema(item, count, soa_threshold)

    fp_write(ARRAY_START if is_row_major else OBJECT_START)
    fp_write(CONTAINER_TYPE + OBJECT_START)
    for f in schema:
        __encode_soa_schema_field(fp_write, f["name"], f, le)
    fp_write(OBJECT_END + CONTAINER_COUNT)

    if len(dims) > 1:
        fp_write(ARRAY_START)
        for d in dims:
            __encode_int(fp_write, d, le)
        fp_write(ARRAY_END)
    else:
        __encode_int(fp_write, count, le)

    if is_row_major:
        for i in range(count):
            for f in schema:
                __write_soa_field_value(fp_write, f, i, le)
    else:
        for f in schema:
            for i in range(count):
                __write_soa_field_value(fp_write, f, i, le)

    for f in __collect_offset_fields(schema):
        encoded = [v.encode("utf-8") for v in f["values"]]
        offsets = [0] + list(accumulate(len(e) for e in encoded))
        for off in offsets:
            __write_index(fp_write, f["index_marker"], off, le)
        for e in encoded:
            fp_write(e)


def __collect_offset_fields(schema):
    """Recursively collect offset-based string fields."""
    for f in schema:
        if f.get("encoding") == "offset":
            yield f
        if f.get("type") == "nested":
            yield from __collect_offset_fields(f["schema"])


def __write_soa_field_value(fp_write, field, index, le):
    """Write a single field value in SOA payload."""
    ftype, values = field["type"], field["values"]

    if ftype == "numeric":
        if hasattr(values, "tobytes"):
            fp_write(values[index].tobytes())
        else:
            __write_numeric_value(fp_write, field["marker"], values[index], le)
    elif ftype == "bool":
        fp_write(TYPE_BOOL_TRUE if values[index] else TYPE_BOOL_FALSE)
    elif ftype == "null":
        pass
    elif ftype == "array":
        fp_write(values[index].tobytes())
    elif ftype == "nested":
        for f in field["schema"]:
            __write_soa_field_value(fp_write, f, index, le)
    elif ftype == "string":
        enc, val = field["encoding"], values[index]
        if enc == "fixed":
            encoded = val.encode("utf-8")
            fp_write((encoded + b"\x00" * field["length"])[: field["length"]])
        elif enc == "dict":
            __write_index(fp_write, field["index_marker"], field["dict"].index(val), le)
        elif enc == "offset":
            __write_index(fp_write, field["index_marker"], index, le)


def __write_numeric_value(fp_write, marker, val, le):
    """Write a numeric value based on marker type."""
    val = val if val is not None else 0
    packers = {
        TYPE_INT8: lambda v: pack("<b" if le else ">b", v),
        TYPE_UINT8: lambda v: pack("<B" if le else ">B", v),
        TYPE_INT16: lambda v: __PACK_INT16[le](v),
        TYPE_UINT16: lambda v: __PACK_UINT16[le](v),
        TYPE_INT32: lambda v: __PACK_INT32[le](v),
        TYPE_UINT32: lambda v: __PACK_UINT32[le](v),
        TYPE_INT64: lambda v: __PACK_INT64[le](v),
        TYPE_UINT64: lambda v: __PACK_UINT64[le](v),
        TYPE_FLOAT32: lambda v: __PACK_FLOAT32[le](v),
        TYPE_FLOAT64: lambda v: __PACK_FLOAT64[le](v),
    }
    fp_write(packers[marker](val))


def __write_index(fp_write, marker, idx, le):
    """Write an index value for dict/offset string encoding."""
    if marker == TYPE_UINT8:
        fp_write(pack("<B" if le else ">B", idx))
    elif marker == TYPE_UINT16:
        fp_write(__PACK_UINT16[le](idx))
    else:
        fp_write(__PACK_UINT32[le](idx))


def __encode_value(
    fp_write,
    item,
    seen_containers,
    container_count,
    sort_keys,
    no_float32,
    uint8_bytes,
    islittle,
    default,
    soa_format,
    soa_threshold=None,
):
    le = islittle

    if isinstance(item, UNICODE_TYPE):
        __encode_string(fp_write, item, le)

    elif item is None:
        fp_write(TYPE_NULL)

    elif item is True:
        fp_write(TYPE_BOOL_TRUE)

    elif item is False:
        fp_write(TYPE_BOOL_FALSE)

    elif isinstance(item, INTEGER_TYPES) and not (type(item).__module__ == "numpy"):
        __encode_int(fp_write, item, le)

    elif isinstance(item, float):
        if no_float32:
            __encode_float64(fp_write, item, le)
        else:
            __encode_float(fp_write, item, le)

    elif isinstance(item, Decimal):
        __encode_decimal(fp_write, item, le)

    elif isinstance(item, BYTES_TYPES):
        __encode_bytes(fp_write, item, uint8_bytes, le)

    # order important since mappings could also be sequences
    elif isinstance(item, Mapping):
        __encode_object(
            fp_write,
            item,
            seen_containers,
            container_count,
            sort_keys,
            no_float32,
            uint8_bytes,
            islittle,
            default,
            soa_format,
        )

    elif isinstance(item, Sequence):
        # Check for SOA-encodable list of dicts
        if soa_format and __can_encode_as_soa(item):
            __encode_soa(
                fp_write, item, soa_format, le, soa_threshold
            )  # Added soa_threshold
        else:
            __encode_array(
                fp_write,
                item,
                seen_containers,
                container_count,
                sort_keys,
                no_float32,
                uint8_bytes,
                islittle,
                default,
                soa_format,
            )
    elif default is not None:
        __encode_value(
            fp_write,
            default(item),
            seen_containers,
            container_count,
            sort_keys,
            no_float32,
            uint8_bytes,
            islittle,
            default,
            soa_format,
            soa_threshold,  # Added soa_threshold
        )
    elif type(item).__module__ == "numpy":
        # Check for SOA-compatible structured array
        if soa_format and __can_encode_as_soa(item):
            __encode_soa(
                fp_write, item, soa_format, le, soa_threshold
            )  # Added soa_threshold
        elif soa_format is None and __can_encode_as_soa(item):
            # Auto-enable column-major SOA for structured arrays
            __encode_soa(
                fp_write, item, "col", le, soa_threshold
            )  # Added soa_threshold
        else:
            __encode_numpy(fp_write, item, uint8_bytes, islittle, default)

    else:
        raise EncoderException("Cannot encode item of type %s" % type(item))


def __encode_array(
    fp_write,
    item,
    seen_containers,
    container_count,
    sort_keys,
    no_float32,
    uint8_bytes,
    islittle,
    default,
    soa_format,
):
    # circular reference check
    container_id = id(item)
    if container_id in seen_containers:
        raise ValueError("Circular reference detected")
    seen_containers[container_id] = item

    fp_write(ARRAY_START)
    if container_count:
        fp_write(CONTAINER_COUNT)
        __encode_int(fp_write, len(item), islittle)

    for value in item:
        __encode_value(
            fp_write,
            value,
            seen_containers,
            container_count,
            sort_keys,
            no_float32,
            uint8_bytes,
            islittle,
            default,
            soa_format,
        )

    if not container_count:
        fp_write(ARRAY_END)

    del seen_containers[container_id]


def __encode_object(
    fp_write,
    item,
    seen_containers,
    container_count,
    sort_keys,
    no_float32,
    uint8_bytes,
    islittle,
    default,
    soa_format,
):
    le = islittle
    # circular reference check
    container_id = id(item)
    if container_id in seen_containers:
        raise ValueError("Circular reference detected")
    seen_containers[container_id] = item

    fp_write(OBJECT_START)
    if container_count:
        fp_write(CONTAINER_COUNT)
        __encode_int(fp_write, len(item), le)

    for key, value in sorted(item.items()) if sort_keys else item.items():
        # allow both str & unicode for Python 2
        if not isinstance(key, TEXT_TYPES):
            raise EncoderException("Mapping keys can only be strings")
        encoded_key = key.encode("utf-8")
        length = len(encoded_key)
        if length < 2**8:
            fp_write(__SMALL_UINTS_ENCODED[le][length])
        else:
            __encode_int(fp_write, length, le)
        fp_write(encoded_key)

        __encode_value(
            fp_write,
            value,
            seen_containers,
            container_count,
            sort_keys,
            no_float32,
            uint8_bytes,
            islittle,
            default,
            soa_format,
        )

    if not container_count:
        fp_write(OBJECT_END)

    del seen_containers[container_id]


def __map_dtype(dtypestr):
    if len(dtypestr) == 3 and (
        dtypestr.startswith("<") or dtypestr.startswith("|") or dtypestr.startswith(">")
    ):
        return __DTYPE_TO_MARKER[dtypestr[1:3]]
    else:
        raise Exception("bjdata", "numpy dtype {} is not supported".format(dtypestr))


def __encode_numpy(fp_write, item, uint8_bytes, islittle, default):
    try:
        import numpy as np
    except ImportError:
        raise Exception("bjdata", "you must install 'numpy' to encode this data")

    # TODO: need to detect big-endian data and swap bytes
    if np.isscalar(item):
        fp_write(__map_dtype(item.dtype.str))
        fp_write(item.data)
        return

    if not (type(item).__name__ == "ndarray" or type(item).__name__ == "chararray"):
        raise Exception(
            "bjdata", "only numerical scalars and ndarrays are supported for numpy data"
        )

    if (item.dtype.str[1] == "U" or item.dtype.str[1] == "S") and item.ndim == 0:
        fp_write(TYPE_STRING)
        __encode_int(
            fp_write,
            int(item.dtype.str[2:]) * (4 if item.dtype.str[1] == "U" else 1),
            islittle,
        )
        fp_write(item.data)
        return

    if np.isfortran(item):
        item = np.array(
            item, order="C"
        )  # currently, BJData ND-array syntax only support row-major

    fp_write(
        ARRAY_START + CONTAINER_TYPE + __map_dtype(item.dtype.str) + CONTAINER_COUNT
    )
    fp_write(ARRAY_START)
    for value in item.shape:
        __encode_int(fp_write, value, islittle)
    fp_write(ARRAY_END)

    fp_write(item.data)


def dump(
    obj,
    fp,
    container_count=False,
    sort_keys=False,
    no_float32=True,
    uint8_bytes=False,
    islittle=True,
    default=None,
    soa_format=None,
    soa_threshold=None,
):
    """Writes the given object as BJData/UBJSON to the provided file-like object

    Args:
        obj: The object to encode
        fp: write([size])-able object
        container_count (bool): Specify length for container types (including
                                for empty ones). This can aid decoding speed
                                depending on implementation but requires a bit
                                more space and encoding speed could be reduced
                                if getting length of any of the containers is
                                expensive.
        sort_keys (bool): Sort keys of mappings
        no_float32 (bool): Never use float32 to store float numbers (other than
                           for zero). Disabling this might save space at the
                           loss of precision.
        uint8_bytes (bool): If set, typed UBJSON arrays (uint8) will be
                         converted to a bytes instance instead of being
                         treated as an array (for UBJSON & BJData Draft 4).
                         Ignored if no_bytes is set.
        islittle (1 or 0): default is 1 for little-endian for all numerics (for
                            BJData Draft 4), change to 0 to use big-endian
                            (for UBJSON for BJData Draft 1)
        default (callable): Called for objects which cannot be serialised.
                            Should return a UBJSON-encodable version of the
                            object or raise an EncoderException.
        soa_format (str): SOA format for numpy structured arrays:
                         'col' or 'column' - column-major (columnar)
                         'row' - row-major (interleaved)
                         None - auto-enable column-major for structured arrays
        soa_threshold: Controls string encoding in SOA format:
                      - None: auto-select based on data analysis
                      - 0: force offset-table encoding for all strings
                      - 0.0-1.0: ratio threshold for dictionary encoding

    Raises:
        EncoderException: If an encoding failure occured.

    The following Python types and interfaces (ABCs) are supported (as are any
    subclasses):

    +------------------------------+-----------------------------------+
    | Python                       | BJData/UBJSON                     |
    +==============================+===================================+
    | (3) str                      | string                            |
    | (2) unicode                  |                                   |
    +------------------------------+-----------------------------------+
    | None                         | null                              |
    +------------------------------+-----------------------------------+
    | bool                         | true, false                       |
    +------------------------------+-----------------------------------+
    | (3) int                      | uint8, int8, int16, int32, int64, |
    | (2) int, long                | high_precision                    |
    +------------------------------+-----------------------------------+
    | float                        | float32, float64, high_precision  |
    +------------------------------+-----------------------------------+
    | Decimal                      | high_precision                    |
    +------------------------------+-----------------------------------+
    | (3) bytes, bytearray         | array (type, byte)                |
    | (2) str                      | array (type, byte)                |
    +------------------------------+-----------------------------------+
    | (3) collections.abc.Mapping  | object                            |
    | (2) collections.Mapping      |                                   |
    +------------------------------+-----------------------------------+
    | (3) collections.abc.Sequence | array                             |
    | (2) collections.Sequence     |                                   |
    +------------------------------+-----------------------------------+

    Notes:
    - Items are resolved in the order of this table, e.g. if the item implements
      both Mapping and Sequence interfaces, it will be encoded as a mapping.
    - None and bool do not use an isinstance check
    - Numbers in brackets denote Python version.
    - Only unicode strings in Python 2 are encoded as strings, i.e. for
      compatibility with e.g. Python 3 one MUST NOT use str in Python 2 (as that
      will be interpreted as a byte array).
    - Mapping keys have to be strings: str for Python3 and unicode or str in
      Python 2.
    - float conversion rules (depending on no_float32 setting):
        float32: 1.18e-38 <= abs(value) <= 3.4e38 or value == 0
        float64: 2.23e-308 <= abs(value) < 1.8e308
        For other values Decimal is used.

    SOA Encoding:
        When soa_format is set and the object is a numpy structured array
        (record array) or list of dicts with consistent keys, it will be
        encoded using the BJData Draft 4 SOA format.

        String fields are automatically encoded using the most efficient method:
        - fixed: for short strings with similar lengths
        - dict: for categorical data with few unique values (<30% unique)
        - offset: for variable-length strings

        Example:
            import numpy as np
            dt = np.dtype([('x', 'u1'), ('y', 'f4'), ('name', 'U32')])
            data = np.array([(65, 1.5, 'Alice'), (66, 2.5, 'Bob')], dtype=dt)
            bjd = dumpb(data, soa_format='col')

            # Or with list of dicts:
            data = [{'id': 1, 'name': 'Alice'}, {'id': 2, 'name': 'Bob'}]
            bjd = dumpb(data, soa_format='col')
    """
    if not callable(fp.write):
        raise TypeError("fp.write not callable")
    fp_write = fp.write

    __encode_value(
        fp_write,
        obj,
        {},
        container_count,
        sort_keys,
        no_float32,
        uint8_bytes,
        islittle,
        default,
        soa_format,
        soa_threshold,
    )


def dumpb(
    obj,
    container_count=False,
    sort_keys=False,
    no_float32=True,
    uint8_bytes=False,
    islittle=True,
    default=None,
    soa_format=None,
    soa_threshold=None,
):
    """Returns the given object as BJData/UBJSON in a bytes instance. See dump() for
    available arguments."""
    with BytesIO() as fp:
        dump(
            obj,
            fp,
            container_count=container_count,
            sort_keys=sort_keys,
            no_float32=no_float32,
            uint8_bytes=uint8_bytes,
            islittle=islittle,
            default=default,
            soa_format=soa_format,
            soa_threshold=soa_threshold,
        )
        return fp.getvalue()
