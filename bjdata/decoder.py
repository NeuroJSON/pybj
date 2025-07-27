# Copyright (c) 2020-2023 Qianqian Fang <q.fang at neu.edu>. All rights reserved.
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


"""BJData (Draft 2) and UBJSON encoder"""

from io import BytesIO
from struct import Struct, pack, error as StructError
from decimal import Decimal, DecimalException
from functools import reduce

from .compat import raise_from, intern_unicode
from .markers import (TYPE_NONE, TYPE_NULL, TYPE_NOOP, TYPE_BOOL_TRUE, TYPE_BOOL_FALSE, TYPE_BYTE, TYPE_INT8, TYPE_UINT8,
                      TYPE_INT16, TYPE_INT32, TYPE_INT64, TYPE_FLOAT32, TYPE_FLOAT64, TYPE_HIGH_PREC, TYPE_CHAR,
		      TYPE_UINT16, TYPE_UINT32, TYPE_UINT64, TYPE_FLOAT16,
                      TYPE_STRING, OBJECT_START, OBJECT_END, ARRAY_START, ARRAY_END, CONTAINER_TYPE, CONTAINER_COUNT)
from numpy import array as ndarray, dtype as npdtype, frombuffer as buffer2numpy, half as halfprec
from array import array as typedarray

__TYPES = frozenset((TYPE_NULL, TYPE_BOOL_TRUE, TYPE_BOOL_FALSE, TYPE_BYTE, TYPE_INT8, TYPE_UINT8, TYPE_INT16,
                     TYPE_INT32, TYPE_INT64, TYPE_FLOAT32, TYPE_FLOAT64, TYPE_UINT16, TYPE_UINT32, TYPE_UINT64,
                     TYPE_FLOAT16, TYPE_HIGH_PREC, TYPE_CHAR, TYPE_STRING, ARRAY_START, OBJECT_START))
__TYPES_NO_DATA = frozenset((TYPE_NULL, TYPE_BOOL_FALSE, TYPE_BOOL_TRUE))
__TYPES_INT = frozenset((TYPE_BYTE, TYPE_INT8, TYPE_UINT8, TYPE_INT16, TYPE_INT32, TYPE_INT64, TYPE_UINT16,
                         TYPE_UINT32, TYPE_UINT64))
__TYPES_FIXLEN = frozenset((TYPE_BYTE, TYPE_INT8, TYPE_UINT8, TYPE_INT16, TYPE_INT32, TYPE_INT64, TYPE_UINT16, TYPE_UINT32,
                            TYPE_UINT64, TYPE_FLOAT16, TYPE_FLOAT32, TYPE_FLOAT64, TYPE_CHAR))

__SMALL_INTS_DECODED = [{pack('>b', i): i for i in range(-128, 128)}, {pack('<b', i): i for i in range(-128, 128)}]
__SMALL_UINTS_DECODED = [{pack('>B', i): i for i in range(256)}, {pack('<B', i): i for i in range(256)}]
__UNPACK_INT16 = [Struct('>h').unpack, Struct('<h').unpack]
__UNPACK_INT32 = [Struct('>i').unpack, Struct('<i').unpack]
__UNPACK_INT64 = [Struct('>q').unpack, Struct('<q').unpack]
__UNPACK_UINT16 = [Struct('>H').unpack, Struct('<H').unpack]
__UNPACK_UINT32 = [Struct('>I').unpack, Struct('<I').unpack]
__UNPACK_UINT64 = [Struct('>Q').unpack, Struct('<Q').unpack]
__UNPACK_FLOAT16 = [Struct('>h').unpack, Struct('<h').unpack]
__UNPACK_FLOAT32 = [Struct('>f').unpack, Struct('<f').unpack]
__UNPACK_FLOAT64 = [Struct('>d').unpack, Struct('<d').unpack]

__DTYPE_MAP = { TYPE_BYTE: 'B',
                TYPE_INT8: 'b',
                TYPE_UINT8: 'B',
                TYPE_INT16: 'h',
                TYPE_UINT16: 'H',
                TYPE_INT32: 'i',
                TYPE_UINT32: 'I',
                TYPE_INT64: 'q',
                TYPE_UINT64: 'Q',
                TYPE_FLOAT16: 'h',
                TYPE_FLOAT32: 'f',
                TYPE_FLOAT64: 'd',
                TYPE_CHAR: 'c'}

__DTYPELEN_MAP={ TYPE_BYTE: 1,
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
                TYPE_CHAR: 1}

class DecoderException(ValueError):
    """Raised when decoding of a UBJSON stream fails."""

    def __init__(self, message, position=None):
        if position is not None:
            super(DecoderException, self).__init__('%s (at byte %d)' % (message, position), position)
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
    length = __decode_int_non_negative(fp_read, fp_read(1),le)
    raw = fp_read(length)
    if len(raw) < length:
        raise DecoderException('High prec. too short')
    try:
        return Decimal(raw.decode('utf-8'))
    except UnicodeError as ex:
        raise_from(DecoderException('Failed to decode decimal string'), ex)
    except DecimalException as ex:
        raise_from(DecoderException('Failed to decode decimal'), ex)


def __decode_int_non_negative(fp_read, marker, le=1):
    if marker not in __TYPES_INT:
        raise DecoderException('Integer marker expected')
    value = __METHOD_MAP[marker](fp_read, marker, le)
    if value < 0:
        raise DecoderException('Negative count/length unexpected')
    return value

def __decode_byte(fp_read, marker, le=1):
    try:
        return __SMALL_UINTS_DECODED[le][fp_read(1)]
    except KeyError as ex:
        raise_from(DecoderException('Failed to unpack byte'), ex)

def __decode_int8(fp_read, marker, le=1):
    try:
        return __SMALL_INTS_DECODED[le][fp_read(1)]
    except KeyError as ex:
        raise_from(DecoderException('Failed to unpack int8'), ex)


def __decode_uint8(fp_read, marker, le=1):
    try:
        return __SMALL_UINTS_DECODED[le][fp_read(1)]
    except KeyError as ex:
        raise_from(DecoderException('Failed to unpack uint8'), ex)


def __decode_int16(fp_read, marker, le=1):
    try:
        return __UNPACK_INT16[le](fp_read(2))[0]
    except StructError as ex:
        raise_from(DecoderException('Failed to unpack int16'), ex)


def __decode_int32(fp_read, marker, le=1):
    try:
        return __UNPACK_INT32[le](fp_read(4))[0]
    except StructError as ex:
        raise_from(DecoderException('Failed to unpack int32'), ex)


def __decode_int64(fp_read, marker, le=1):
    try:
        return __UNPACK_INT64[le](fp_read(8))[0]
    except StructError as ex:
        raise_from(DecoderException('Failed to unpack int64'), ex)

def __decode_uint16(fp_read, marker, le=1):
    try:
        return __UNPACK_UINT16[le](fp_read(2))[0]
    except StructError as ex:
        raise_from(DecoderException('Failed to unpack uint16'), ex)


def __decode_uint32(fp_read, marker, le=1):
    try:
        return __UNPACK_UINT32[le](fp_read(4))[0]
    except StructError as ex:
        raise_from(DecoderException('Failed to unpack uint32'), ex)


def __decode_uint64(fp_read, marker, le=1):
    try:
        return __UNPACK_UINT64[le](fp_read(8))[0]
    except StructError as ex:
        raise_from(DecoderException('Failed to unpack uint64'), ex)


def __decode_float16(fp_read, marker, le=1):
    try:
        return __UNPACK_FLOAT16[le](fp_read(2))[0]
    except StructError as ex:
        raise_from(DecoderException('Failed to unpack float16'), ex)

def __decode_float32(fp_read, marker, le=1):
    try:
        return __UNPACK_FLOAT32[le](fp_read(4))[0]
    except StructError as ex:
        raise_from(DecoderException('Failed to unpack float32'), ex)


def __decode_float64(fp_read, marker, le=1):
    try:
        return __UNPACK_FLOAT64[le](fp_read(8))[0]
    except StructError as ex:
        raise_from(DecoderException('Failed to unpack float64'), ex)


def __decode_char(fp_read, marker, le=1):
    raw = fp_read(1)
    if not raw:
        raise DecoderException('Char missing')
    try:
        return raw.decode('utf-8')
    except UnicodeError as ex:
        raise_from(DecoderException('Failed to decode char'), ex)


def __decode_string(fp_read, marker, le=1):
    # current marker is string identifier, so read next byte which identifies integer type
    length = __decode_int_non_negative(fp_read, fp_read(1), le)
    raw = fp_read(length)
    if len(raw) < length:
        raise DecoderException('String too short')
    try:
        return raw.decode('utf-8')
    except UnicodeError as ex:
        raise_from(DecoderException('Failed to decode string'), ex)


# same as string, except there is no 'S' marker
def __decode_object_key(fp_read, marker, intern_object_keys, le=1):
    length = __decode_int_non_negative(fp_read, marker, le)
    raw = fp_read(length)
    if len(raw) < length:
        raise DecoderException('String too short')
    try:
        return intern_unicode(raw.decode('utf-8')) if intern_object_keys else raw.decode('utf-8')
    except UnicodeError as ex:
        raise_from(DecoderException('Failed to decode object key'), ex)


__METHOD_MAP = {TYPE_NULL: (lambda _, __, ___: None),
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
                TYPE_STRING: __decode_string}

def prodlist(mylist):
    result = 1
    for x in mylist: 
         result = result * x
    return result

def __get_container_params(fp_read, in_mapping, no_bytes, uint8_bytes, object_hook, object_pairs_hook, intern_object_keys, islittle):
    marker = fp_read(1)
    dims = []
    if marker == CONTAINER_TYPE:
        marker = fp_read(1)
        if marker not in __TYPES:
            raise DecoderException('Invalid container type')
        type_ = marker
        marker = fp_read(1)
    else:
        type_ = TYPE_NONE
    if marker == CONTAINER_COUNT:
        marker = fp_read(1)
        if marker == ARRAY_START:
            dims = __decode_array(fp_read, no_bytes, uint8_bytes, object_hook, object_pairs_hook, intern_object_keys, islittle)
            count = prodlist(dims)
        else:
            count = __decode_int_non_negative(fp_read, marker, islittle)
        counting = True

        # special cases (no data (None or bool) / bytes array) will be handled in calling functions
        if not (type_ in __TYPES_NO_DATA or
                (type_ == (TYPE_UINT8 if uint8_bytes else TYPE_BYTE) and not in_mapping and not no_bytes)):
            # Reading ahead is just to capture type, which will not exist if type is fixed
            marker = fp_read(1) if (in_mapping or type_ == TYPE_NONE) else type_

    elif type_ == TYPE_NONE:
        # set to one to indicate that not finished yet
        count = 1
        counting = False
    else:
        raise DecoderException('Container type without count')
    return marker, counting, count, type_, dims


def __decode_object(fp_read, no_bytes, uint8_bytes, object_hook, object_pairs_hook,  # pylint: disable=too-many-branches
                    intern_object_keys, islittle):
    marker, counting, count, type_, dims = __get_container_params(fp_read, True, no_bytes, uint8_bytes, object_hook, object_pairs_hook,intern_object_keys, islittle)
    has_pairs_hook = object_pairs_hook is not None
    obj = [] if has_pairs_hook else {}

    le=islittle

    # special case - no data (None or bool)
    if type_ in __TYPES_NO_DATA:
        value = __METHOD_MAP[type_](fp_read, type_, le)
        if has_pairs_hook:
            for _ in range(count):
                obj.append((__decode_object_key(fp_read, fp_read(1), intern_object_keys, le), value))
            return object_pairs_hook(obj)

        for _ in range(count):
            obj[__decode_object_key(fp_read, fp_read(1), intern_object_keys, le)] = value
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
                value = __decode_array(fp_read, no_bytes, uint8_bytes, object_hook, object_pairs_hook, intern_object_keys, islittle)
            elif marker == OBJECT_START:
                value = __decode_object(fp_read, no_bytes, uint8_bytes, object_hook, object_pairs_hook, intern_object_keys, islittle)
            else:
                raise DecoderException('Invalid marker within object')

        if has_pairs_hook:
            obj.append((key, value))
        else:
            obj[key] = value
        if counting:
            count -= 1
        if count > 0:
            marker = fp_read(1)

    return object_pairs_hook(obj) if has_pairs_hook else object_hook(obj)


def __decode_array(fp_read, no_bytes, uint8_bytes, object_hook, object_pairs_hook, intern_object_keys, islittle):
    marker, counting, count, type_, dims = __get_container_params(fp_read, False, no_bytes, uint8_bytes, object_hook, object_pairs_hook, intern_object_keys, islittle)

    # special case - no data (None or bool)
    if type_ in __TYPES_NO_DATA:
        return [__METHOD_MAP[type_](fp_read, type_, islittle)] * count

    # special case - bytes array
    if type_ == (TYPE_UINT8 if uint8_bytes else TYPE_BYTE) and not no_bytes and len(dims)==0:
        container = fp_read(count)
        if len(container) < count:
            raise DecoderException('Container bytes array too short')
        return container

    if type_ in __TYPES_FIXLEN and count>0:
        if hasattr(count, 'dtype'):
            container = fp_read(count.item()*__DTYPELEN_MAP[type_])
        else:
            container = fp_read(count*__DTYPELEN_MAP[type_])
        if len(container) < count*__DTYPELEN_MAP[type_]:
            raise DecoderException('Container bytes array too short')

        #container=typedarray(__DTYPE_MAP[type_], container)
        if len(dims)>0:
            container=buffer2numpy(container, dtype=npdtype(__DTYPE_MAP[type_]))
            container=container.reshape(dims)
        else:
            container=buffer2numpy(container, dtype=npdtype(__DTYPE_MAP[type_]))
        return container

    container = []
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
                value = __decode_array(fp_read, no_bytes, uint8_bytes, object_hook, object_pairs_hook, intern_object_keys, islittle)
            elif marker == OBJECT_START:
                value = __decode_object(fp_read, no_bytes, uint8_bytes, object_hook, object_pairs_hook, intern_object_keys, islittle)
            else:
                raise DecoderException('Invalid marker within array')

        container.append(value)
        if counting:
            count -= 1
        if count and type_ == TYPE_NONE:
            marker = fp_read(1)

    if len(dims)>0:
        container=list(reduce(lambda x, y: map(list, zip(*y*(x,))), (iter(container), ) +tuple(dims[:0:-1])))
        container=ndarray(container, dtype=npdtype(__DTYPE_MAP[type_]))

    return container


def __object_hook_noop(obj):
    return obj


def load(fp, no_bytes=False, uint8_bytes=False, object_hook=None, object_pairs_hook=None, intern_object_keys=False, islittle=True):
    """Decodes and returns BJData/UBJSON from the given file-like object

    Args:
        fp: read([size])-able object
        no_bytes (bool): If set, typed UBJSON arrays (byte) will not be
                         converted to a bytes instance and instead treated like
                         any other array (i.e. result in a list).
        uint8_bytes (bool): If set, typed UBJSON arrays (uint8) will be
                         converted to a bytes instance instead of being
                         treated as an array (for UBJSON & BJData Draft 2).
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
                            BJData Draft 2), change to 0 to use big-endian
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
    """
    if object_pairs_hook is None and object_hook is None:
        object_hook = __object_hook_noop

    if not callable(fp.read):
        raise TypeError('fp.read not callable')
    fp_read = fp.read

    newobj=[]

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
                newobj.append(__decode_array(fp_read, bool(no_bytes), bool(uint8_bytes), object_hook, object_pairs_hook, intern_object_keys, islittle))
            if marker == OBJECT_START:
                newobj.append(__decode_object(fp_read, bool(no_bytes), bool(uint8_bytes), object_hook, object_pairs_hook, intern_object_keys, islittle))
            raise DecoderException('Invalid marker')
        except DecoderException as ex:
            if len(newobj)>0:
                pass
            else:
                raise_from(DecoderException(ex.args[0], position=(fp.tell() if hasattr(fp, 'tell') else None)), ex)
    if(len(newobj)==1):
        newobj=newobj[0];
    elif(len(newobj)==0):
        raise DecoderException('Empty data');

    return newobj;

def loadb(chars, no_bytes=False, uint8_bytes=False, object_hook=None, object_pairs_hook=None, intern_object_keys=False, islittle=True):
    """Decodes and returns BJData/UBJSON from the given bytes or bytesarray object. See
       load() for available arguments."""
    with BytesIO(chars) as fp:
        return load(fp, no_bytes=no_bytes, uint8_bytes=uint8_bytes, object_hook=object_hook, object_pairs_hook=object_pairs_hook,
                    intern_object_keys=intern_object_keys, islittle=islittle)
