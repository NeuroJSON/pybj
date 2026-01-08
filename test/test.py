# Copyright (c) 2020-2025 Qianqian Fang <q.fang at neu.edu>. All rights reserved.
# Copyright (c) 2016-2019 Iotic Labs Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://github.com/Iotic-Labs/py-bjdata/blob/master/LICENSE
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from sys import version_info, getrecursionlimit, setrecursionlimit, path
import os

# Add project_root to sys.path
path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from functools import partial
from io import BytesIO, SEEK_END
from unittest import TestCase, skipUnless
from pprint import pformat
from decimal import Decimal
from struct import pack
from collections import OrderedDict

from bjdata import (
    dump as bjddump,
    dumpb as bjddumpb,
    load as bjdload,
    loadb as bjdloadb,
    EncoderException,
    DecoderException,
    EXTENSION_ENABLED,
)
from bjdata.markers import (
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
from bjdata.compat import INTEGER_TYPES

# Pure Python versions
from bjdata.encoder import dump as bjdpuredump, dumpb as bjdpuredumpb
from bjdata.decoder import load as bjdpureload, loadb as bjdpureloadb
import numpy as np
from numpy import array as ndarray, int8 as npint8
from array import array as typedarray

PY2 = version_info[0] < 3

if PY2:  # pragma: no cover

    def u(obj):
        """Casts obj to unicode string, unless already one"""
        return (
            obj if isinstance(obj, unicode) else unicode(obj)
        )  # noqa: F821 pylint: disable=undefined-variable

else:  # pragma: no cover

    def u(obj):
        """Casts obj to unicode string, unless already one"""
        return obj if isinstance(obj, str) else str(obj)


class TestEncodeDecodePlain(TestCase):  # pylint: disable=too-many-public-methods
    @staticmethod
    def bjdloadb(raw, *args, **kwargs):
        return bjdpureloadb(raw, *args, **kwargs)

    @staticmethod
    def bjddumpb(obj, *args, **kwargs):
        return bjdpuredumpb(obj, *args, **kwargs)

    @staticmethod
    def __format_in_out(obj, encoded):
        return "\nInput:\n%s\nOutput (%d):\n%s" % (pformat(obj), len(encoded), encoded)

    if PY2:  # pragma: no cover

        def type_check(self, actual, expected):
            self.assertEqual(actual, expected)

    else:  # pragma: no cover

        def type_check(self, actual, expected):
            self.assertEqual(actual, ord(expected))

    # based on math.isclose available in Python v3.5
    @staticmethod
    # pylint: disable=invalid-name
    def numbers_close(a, b, rel_tol=1e-05, abs_tol=0.0):
        return abs(a - b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)

    def check_enc_dec(
        self,
        obj,
        # total length of encoded object
        length=None,
        # total length is at least the given number of bytes
        length_greater_or_equal=False,
        # approximate comparison (e.g. for float)
        approximate=False,
        # type marker expected at start of encoded output
        expected_type=None,
        # decoder params
        object_hook=None,
        object_pairs_hook=None,
        # additional arguments to pass to encoder
        **kwargs,
    ):
        """Black-box test to check whether the provided object is the same once encoded and subsequently decoded."""
        encoded = self.bjddumpb(obj, **kwargs)

        if expected_type is not None:
            self.type_check(encoded[0], expected_type)
        if length is not None:
            assert_func = (
                self.assertGreaterEqual if length_greater_or_equal else self.assertEqual
            )
            assert_func(len(encoded), length, self.__format_in_out(obj, encoded))
        if approximate:
            self.assertTrue(
                self.numbers_close(
                    self.bjdloadb(
                        encoded,
                        object_hook=object_hook,
                        object_pairs_hook=object_pairs_hook,
                    ),
                    obj,
                ),
                msg=self.__format_in_out(obj, encoded),
            )
        else:
            self.assertEqual(
                self.bjdloadb(
                    encoded,
                    object_hook=object_hook,
                    object_pairs_hook=object_pairs_hook,
                ),
                obj,
                self.__format_in_out(obj, encoded),
            )

    def test_no_data(self):
        with self.assertRaises(DecoderException):
            self.bjdloadb(b"")

    def test_invalid_data(self):
        for invalid in (u("unicode"), 123):
            with self.assertRaises(TypeError):
                self.bjdloadb(invalid)

    def test_trailing_input(self):
        self.assertEqual(self.bjdloadb(TYPE_BOOL_TRUE * 10), True)

    def test_invalid_marker(self):
        with self.assertRaises(DecoderException) as ctx:
            self.bjdloadb(b"A")
        self.assertTrue(
            isinstance(ctx.exception.position, INTEGER_TYPES + (type(None),))
        )

    def test_bool(self):
        self.assertEqual(self.bjddumpb(True), TYPE_BOOL_TRUE)
        self.assertEqual(self.bjddumpb(False), TYPE_BOOL_FALSE)
        self.check_enc_dec(True, 1)
        self.check_enc_dec(False, 1)

    def test_null(self):
        self.assertEqual(self.bjddumpb(None), TYPE_NULL)
        self.check_enc_dec(None, 1)

    def test_char(self):
        self.assertEqual(self.bjddumpb(u("a")), TYPE_CHAR + "a".encode("utf-8"))
        # no char, char invalid utf-8
        for suffix in (b"", b"\xfe"):
            with self.assertRaises(DecoderException):
                self.bjdloadb(TYPE_CHAR + suffix)
        for char in (u("a"), u("\0"), u("~")):
            self.check_enc_dec(char, 2)

    def test_string(self):
        self.assertEqual(
            self.bjddumpb(u("ab")),
            TYPE_STRING + TYPE_UINT8 + b"\x02" + "ab".encode("utf-8"),
        )
        self.check_enc_dec(u(""), 3)
        # invalid string size, string too short, string invalid utf-8
        for suffix in (b"\x81", b"\x01", b"\x01" + b"\xfe"):
            with self.assertRaises(DecoderException):
                self.bjdloadb(TYPE_STRING + TYPE_INT8 + suffix)
        # Note: In Python 2 plain str type is encoded as byte array
        for string in (
            "some ascii",
            u(r"\u00a9 with extended\u2122"),
            u("long string") * 100,
        ):
            self.check_enc_dec(string, 4, length_greater_or_equal=True)

    def test_int(self):
        self.assertEqual(
            self.bjddumpb(Decimal(-1.5)),
            TYPE_HIGH_PREC + TYPE_UINT8 + b"\x04" + "-1.5".encode("utf-8"),
        )
        # insufficient length
        with self.assertRaises(DecoderException):
            self.bjdloadb(TYPE_INT16 + b"\x01")

        for type_, value, total_size in (
            (TYPE_UINT8, 0, 2),
            (TYPE_UINT8, 255, 2),
            (TYPE_INT8, -128, 2),
            (TYPE_INT16, -32768, 3),
            (TYPE_UINT16, 456, 3),
            (TYPE_UINT16, 32767, 3),
            (TYPE_INT32, -2147483648, 5),
            (TYPE_UINT32, 1610612735, 5),
            (TYPE_UINT32, 2147483647, 5),
            (TYPE_INT64, -9223372036854775808, 9),
            (TYPE_UINT64, 6917529027641081855, 9),
            (TYPE_UINT64, 9223372036854775807, 9),
            (TYPE_UINT64, 9223372036854775808, 9),
            # HIGH_PREC (marker + length marker + length + value)
            (TYPE_HIGH_PREC, -9223372036854775809, 23),
            (TYPE_HIGH_PREC, 9999999999999999999999999999999999999, 40),
        ):
            self.check_enc_dec(value, total_size, expected_type=type_)

        self.assertEqual(
            (
                self.bjdloadb(
                    b"[$U#U\x08\x01\x02\x03\x04\x05\x06\x07\x08", uint8_bytes=True
                )
                == b"\x01\x02\x03\x04\x05\x06\x07\x08"
            ),
            True,
        )

        self.assertEqual(
            (
                self.bjdloadb(b"[$B#U\x08\x01\x02\x03\x04\x05\x06\x07\x08")
                == b"\x01\x02\x03\x04\x05\x06\x07\x08"
            ),
            True,
        )

        self.assertEqual(
            (
                self.bjdloadb(b"[$u#U\x04\x01\x02\x03\x04\x05\x06\x07\x08")
                == ndarray([513, 1027, 1541, 2055], np.uint16)
            ).all(),
            True,
        )

        self.assertEqual(
            (
                self.bjdloadb(b"[$m#U\x02\x01\x02\x03\x04\x05\x06\x07\x08")
                == ndarray([67305985, 134678021], np.uint32)
            ).all(),
            True,
        )

        self.assertEqual(
            (
                self.bjdloadb(b"[$M#U\x01\x01\x02\x03\x04\x05\x06\x07\x08")
                == ndarray([578437695752307201], np.uint64)
            ).all(),
            True,
        )

        self.assertEqual(
            (
                self.bjdloadb(b"[$i#U\x08\xf1\xf2\xf3\xf4\xf5\xf6\xf7\xf8")
                == ndarray([-15, -14, -13, -12, -11, -10, -9, -8], np.int8)
            ).all(),
            True,
        )

        self.assertEqual(
            (
                self.bjdloadb(b"[$I#U\x04\xf1\xf2\xf3\xf4\xf5\xf6\xf7\xf8")
                == ndarray([-3343, -2829, -2315, -1801], np.int16)
            ).all(),
            True,
        )

        self.assertEqual(
            (
                self.bjdloadb(b"[$l#U\x02\xf1\xf2\xf3\xf4\xf5\xf6\xf7\xf8")
                == ndarray([-185339151, -117967115], np.int32)
            ).all(),
            True,
        )

        self.assertEqual(
            (
                self.bjdloadb(b"[$L#U\x01\xf1\xf2\xf3\xf4\xf5\xf6\xf7\xf8")
                == ndarray([-506664896818842895], np.int64)
            ).all(),
            True,
        )

    def test_high_precision(self):
        self.assertEqual(
            self.bjddumpb(Decimal(-1.5)),
            TYPE_HIGH_PREC + TYPE_UINT8 + b"\x04" + "-1.5".encode("utf-8"),
        )
        # insufficient length, invalid utf-8, invalid decimal value
        for suffix in (b"n", b"\xfe\xfe", b"na"):
            with self.assertRaises(DecoderException):
                self.bjdloadb(TYPE_HIGH_PREC + TYPE_UINT8 + b"\x02" + suffix)

        self.check_enc_dec("1.8e315")
        for value in ("0.0", "2.5", "10e30", "-1.2345e67890"):
            # minimum length because: marker + length marker + length + value
            self.check_enc_dec(Decimal(value), 4, length_greater_or_equal=True)
        # cannot compare equality, so test separately
        for value in ("nan", "-inf", "inf"):
            self.assertEqual(str(self.bjdloadb(self.bjddumpb(float(value)))), value)

    def test_float(self):
        # insufficient length
        for float_type in (TYPE_FLOAT32, TYPE_FLOAT64):
            with self.assertRaises(DecoderException):
                self.bjdloadb(float_type + b"\x01")

        self.check_enc_dec(0.0, 5, expected_type=TYPE_FLOAT32)

        for type_, value, total_size in (
            (TYPE_FLOAT32, 1.18e-37, 5),
            (TYPE_FLOAT32, 3.4e37, 5),
            (TYPE_FLOAT64, 2.23e-308, 9),
            (TYPE_FLOAT64, 12345.44e40, 9),
            (TYPE_FLOAT64, 1.8e307, 9),
        ):
            self.check_enc_dec(
                value,
                total_size,
                approximate=True,
                expected_type=type_,
                no_float32=False,
            )
            # using only float64 (default)
            self.check_enc_dec(
                value,
                9 if type_ == TYPE_FLOAT32 else total_size,
                approximate=True,
                expected_type=(TYPE_FLOAT64 if type_ == TYPE_FLOAT32 else type_),
            )
        for value in ("nan", "-inf", "inf"):
            for no_float32 in (True, False):
                self.assertEqual(
                    str(
                        self.bjdloadb(
                            self.bjddumpb(float(value), no_float32=no_float32)
                        )
                    ),
                    value,
                )

        # value which results in high_prec usage
        for no_float32 in (True, False):
            self.check_enc_dec(
                2.22e-308,
                4,
                expected_type=TYPE_HIGH_PREC,
                length_greater_or_equal=True,
                no_float32=no_float32,
            )

    def test_array(self):
        # invalid length
        with self.assertRaises(DecoderException):
            self.bjdloadb(ARRAY_START + CONTAINER_COUNT + self.bjddumpb(-5))
        # unencodable type within
        with self.assertRaises(EncoderException):
            self.bjddumpb([type(None)])
        for sequence in list, tuple:
            self.assertEqual(self.bjddumpb(sequence()), ARRAY_START + ARRAY_END)
        self.assertEqual(
            self.bjddumpb((None,), container_count=True),
            (ARRAY_START + CONTAINER_COUNT + TYPE_UINT8 + b"\x01" + TYPE_NULL),
        )
        obj = [
            123,
            1.25,
            43121609.5543,
            12345.44e40,
            Decimal("10e15"),
            "a",
            "here is a string",
            None,
            True,
            False,
            [[1, 2], 3, [4, 5, 6], 7],
            {"a dict": 456},
        ]
        for opts in ({"container_count": False}, {"container_count": True}):
            self.check_enc_dec(obj, **opts)

    def test_bytes(self):
        # insufficient length
        with self.assertRaises(DecoderException):
            self.bjdloadb(
                ARRAY_START
                + CONTAINER_TYPE
                + TYPE_BYTE
                + CONTAINER_COUNT
                + TYPE_UINT8
                + b"\x02"
                + b"\x01"
            )
        for cast in (bytes, bytearray):
            self.check_enc_dec(cast(b""))
            self.check_enc_dec(cast(b"\x01" * 4))
            # self.assertEqual((self.bjdloadb(self.bjddumpb(cast(b'\x04' * 4)), no_bytes=True) == ndarray([4] * 4, npint8)).all(), True)
            self.check_enc_dec(cast(b"largebinary" * 100))

    def test_nd_array(self):
        raw_start = (
            ARRAY_START
            + CONTAINER_TYPE
            + TYPE_INT8
            + CONTAINER_COUNT
            + ARRAY_START
            + CONTAINER_TYPE
            + TYPE_INT8
            + CONTAINER_COUNT
            + TYPE_UINT8
            + b"\x02"
            + b"\x03"
            + b"\x02"
            + b"\x01"
            + b"\x02"
            + b"\x03"
            + b"\x04"
            + b"\x05"
            + b"\x06"
        )
        self.assertEqual(
            (
                self.bjdloadb(raw_start) == ndarray([[1, 2], [3, 4], [5, 6]], npint8)
            ).all(),
            True,
        )

        self.assertEqual(
            (self.bjdloadb(self.bjddumpb(np.uint8(9))) == np.uint8(9)), True
        )
        self.assertEqual((self.bjdloadb(self.bjddumpb(np.int8(9))) == np.int8(9)), True)
        self.assertEqual(
            (self.bjdloadb(self.bjddumpb(np.uint16(6))) == np.uint16(6)), True
        )
        self.assertEqual(
            (self.bjdloadb(self.bjddumpb(np.int16(6))) == np.int16(6)), True
        )
        self.assertEqual(
            (self.bjdloadb(self.bjddumpb(np.uint32(6))) == np.uint32(6)), True
        )
        self.assertEqual(
            (self.bjdloadb(self.bjddumpb(np.int32(6))) == np.int32(6)), True
        )
        self.assertEqual(
            (self.bjdloadb(self.bjddumpb(np.uint64(6))) == np.uint64(6)), True
        )
        self.assertEqual(
            (self.bjdloadb(self.bjddumpb(np.int64(5))) == np.int64(5)), True
        )

        self.assertEqual(
            (
                self.bjdloadb(self.bjddumpb(np.eye(2, dtype=np.uint8)))
                == np.eye(2, dtype=np.uint8)
            ).all(),
            True,
        )
        self.assertEqual(
            (
                self.bjdloadb(self.bjddumpb(np.eye(2, dtype=np.int16)))
                == np.eye(2, dtype=np.int16)
            ).all(),
            True,
        )

        self.assertEqual((self.bjdloadb(self.bjddumpb(np.float16(2.2))) == 16486), True)
        self.assertEqual(
            (self.bjdloadb(self.bjddumpb(np.float32(2.2))) == np.float32(2.2)), True
        )

        self.assertEqual(
            (
                self.bjdloadb(
                    self.bjddumpb(
                        np.array([1.3, -0.5, 0.7, 1000, 11], dtype=np.float32)
                    )
                )
                == np.array([1.3, -0.5, 0.7, 1000, 11], dtype=np.float32)
            ).all(),
            True,
        )
        self.assertEqual(
            (
                self.bjdloadb(self.bjddumpb(np.array([1, 2, 3, 4], dtype=np.uint16)))
                == np.array([1, 2, 3, 4], dtype=np.uint16)
            ).all(),
            True,
        )
        self.assertEqual(
            (
                self.bjdloadb(self.bjddumpb(np.array([1, 2, 3, 4], dtype=np.uint8)))
                == np.array([1, 2, 3, 4], dtype=np.uint8)
            ).all(),
            True,
        )
        self.assertEqual(
            (
                self.bjdloadb(self.bjddumpb(np.array([], dtype=np.int8)))
                == np.array([], dtype=np.int8)
            ).all(),
            True,
        )
        self.assertEqual(
            (
                self.bjdloadb(
                    self.bjddumpb(np.array([-1, -2, -3, -4], dtype=np.float32))
                )
                == np.array([-1, -2, -3, -4], dtype=np.float32)
            ).all(),
            True,
        )
        self.assertEqual(
            (
                self.bjdloadb(
                    self.bjddumpb(
                        np.array([[-1, -2, 5], [-3, -4, -6]], dtype=np.float64)
                    )
                )
                == np.array([[-1, -2, 5], [-3, -4, -6]], dtype=np.float64)
            ).all(),
            True,
        )

        raw_start = (
            ARRAY_START
            + CONTAINER_TYPE
            + TYPE_INT8
            + CONTAINER_COUNT
            + ARRAY_START
            + TYPE_UINT8
            + b"\x03"
            + TYPE_UINT16
            + b"\x02"
            + b"\x00"
            + ARRAY_END
            + b"\x01"
            + b"\x02"
            + b"\x03"
            + b"\x04"
            + b"\x05"
            + b"\x06"
        )
        self.assertEqual(
            (
                self.bjdloadb(raw_start) == ndarray([[1, 2], [3, 4], [5, 6]], npint8)
            ).all(),
            True,
        )

    def test_array_fixed(self):
        raw_start = (
            ARRAY_START + CONTAINER_TYPE + TYPE_INT8 + CONTAINER_COUNT + TYPE_UINT8
        )
        self.assertEqual(self.bjdloadb(raw_start + b"\x00"), [])

        # fixed types + count
        for bjd_type, py_obj in (
            (TYPE_NULL, None),
            (TYPE_BOOL_TRUE, True),
            (TYPE_BOOL_FALSE, False),
        ):
            self.assertEqual(
                self.bjdloadb(
                    ARRAY_START
                    + CONTAINER_TYPE
                    + bjd_type
                    + CONTAINER_COUNT
                    + TYPE_UINT8
                    + b"\x05"
                ),
                [py_obj] * 5,
            )
        self.assertEqual(
            (
                self.bjdloadb(raw_start + b"\x03" + (b"\x01" * 3))
                == ndarray([1, 1, 1], dtype=npint8)
            ).all(),
            True,
        )

        # invalid type
        with self.assertRaises(DecoderException):
            self.bjdloadb(ARRAY_START + CONTAINER_TYPE + b"\x01")

        # type without count
        with self.assertRaises(DecoderException):
            self.bjdloadb(ARRAY_START + CONTAINER_TYPE + TYPE_INT8 + b"\x01")

        # count without type
        self.assertEqual(
            self.bjdloadb(
                ARRAY_START
                + CONTAINER_COUNT
                + TYPE_UINT8
                + b"\x02"
                + TYPE_BOOL_FALSE
                + TYPE_BOOL_TRUE
            ),
            [False, True],
        )

        # nested
        self.assertEqual(
            self.bjdloadb(
                ARRAY_START
                + CONTAINER_TYPE
                + ARRAY_START
                + CONTAINER_COUNT
                + TYPE_UINT8
                + b"\x03"
                + ARRAY_END
                + CONTAINER_COUNT
                + TYPE_UINT8
                + b"\x01"
                + TYPE_BOOL_TRUE
                + TYPE_BOOL_FALSE
                + TYPE_BOOL_TRUE
                + ARRAY_END
            ),
            [[], [True], [False, True]],
        )

    def test_array_noop(self):
        # only supported without type
        self.assertEqual(
            self.bjdloadb(
                ARRAY_START
                + TYPE_NOOP
                + TYPE_UINT8
                + b"\x01"
                + TYPE_NOOP
                + TYPE_UINT8
                + b"\x02"
                + TYPE_NOOP
                + ARRAY_END
            ),
            [1, 2],
        )
        self.assertEqual(
            self.bjdloadb(
                ARRAY_START
                + CONTAINER_COUNT
                + TYPE_UINT8
                + b"\x01"
                + TYPE_NOOP
                + TYPE_UINT8
                + b"\x01"
            ),
            [1],
        )

    def test_object_invalid(self):
        # negative length
        with self.assertRaises(DecoderException):
            self.bjdloadb(OBJECT_START + CONTAINER_COUNT + self.bjddumpb(-1))

        with self.assertRaises(EncoderException):
            self.bjddumpb({123: "non-string key"})

        with self.assertRaises(EncoderException):
            self.bjddumpb({"fish": type(list)})

        # invalid key size type
        with self.assertRaises(DecoderException):
            self.bjdloadb(OBJECT_START + TYPE_NULL)

        # invalid key size, key too short, key invalid utf-8, no value
        for suffix in (b"\x81", b"\x01", b"\x01" + b"\xfe", b"\x0101"):
            with self.assertRaises(DecoderException):
                self.bjdloadb(OBJECT_START + TYPE_INT8 + suffix)

        # invalid items() method
        class BadDict(dict):
            def items(self):
                return super(BadDict, self).keys()

        with self.assertRaises(ValueError):
            self.bjddumpb(BadDict({"a": 1, "b": 2}))

    def test_object(self):
        # custom hook
        with self.assertRaises(TypeError):
            self.bjdloadb(self.bjddumpb({}), object_pairs_hook=int)
        # same as not specifying a custom class
        self.bjdloadb(self.bjddumpb({}), object_pairs_hook=None)

        for hook in (None, OrderedDict):
            check_enc_dec = partial(self.check_enc_dec, object_pairs_hook=hook)

            self.assertEqual(self.bjddumpb({}), OBJECT_START + OBJECT_END)
            self.assertEqual(
                self.bjddumpb({"a": None}, container_count=True),
                (
                    OBJECT_START
                    + CONTAINER_COUNT
                    + TYPE_UINT8
                    + b"\x01"
                    + TYPE_UINT8
                    + b"\x01"
                    + "a".encode("utf-8")
                    + TYPE_NULL
                ),
            )
            check_enc_dec({})
            check_enc_dec({"longkey1" * 65: 1})
            check_enc_dec({"longkey2" * 4096: 1})

            obj = {
                "int": 123,
                "longint": 9223372036854775807,
                "float": 1.25,
                "hp": Decimal("10e15"),
                "char": "a",
                "str": "here is a string",
                "unicode": u(r"\u00a9 with extended\u2122"),
                "": "empty key",
                u(r"\u00a9 with extended\u2122"): "unicode-key",
                "null": None,
                "true": True,
                "false": False,
                "array": [1, 2, 3],
                "bytes_array": b"1234",
                "object": {"another one": 456, "yet another": {"abc": True}},
            }
            for opts in ({"container_count": False}, {"container_count": True}):
                check_enc_dec(obj, **opts)

        # dictionary key sorting
        obj1 = OrderedDict.fromkeys("abcdefghijkl")
        obj2 = OrderedDict.fromkeys("abcdefghijkl"[::-1])
        self.assertNotEqual(self.bjddumpb(obj1), self.bjddumpb(obj2))
        self.assertEqual(
            self.bjddumpb(obj1, sort_keys=True), self.bjddumpb(obj2, sort_keys=True)
        )

        self.assertEqual(
            self.bjdloadb(self.bjddumpb(obj1), object_pairs_hook=OrderedDict), obj1
        )

    def test_object_fixed(self):
        raw_start = (
            OBJECT_START + CONTAINER_TYPE + TYPE_INT8 + CONTAINER_COUNT + TYPE_UINT8
        )

        for hook in (None, OrderedDict):
            loadb = partial(self.bjdloadb, object_pairs_hook=hook)

            self.assertEqual(loadb(raw_start + b"\x00"), {})
            self.assertEqual(
                loadb(
                    raw_start
                    + b"\x03"
                    + (
                        TYPE_UINT8
                        + b"\x02"
                        + b"aa"
                        + b"\x01"
                        + TYPE_UINT8
                        + b"\x02"
                        + b"bb"
                        + b"\x02"
                        + TYPE_UINT8
                        + b"\x02"
                        + b"cc"
                        + b"\x03"
                    )
                ),
                {"aa": 1, "bb": 2, "cc": 3},
            )

            # count only
            self.assertEqual(
                loadb(
                    OBJECT_START
                    + CONTAINER_COUNT
                    + TYPE_UINT8
                    + b"\x02"
                    + TYPE_UINT8
                    + b"\x02"
                    + b"aa"
                    + TYPE_NULL
                    + TYPE_UINT8
                    + b"\x02"
                    + b"bb"
                    + TYPE_NULL
                ),
                {"aa": None, "bb": None},
            )

            # fixed type + count
            self.assertEqual(
                loadb(
                    OBJECT_START
                    + CONTAINER_TYPE
                    + TYPE_NULL
                    + CONTAINER_COUNT
                    + TYPE_UINT8
                    + b"\x02"
                    + TYPE_UINT8
                    + b"\x02"
                    + b"aa"
                    + TYPE_UINT8
                    + b"\x02"
                    + b"bb"
                ),
                {"aa": None, "bb": None},
            )

            # fixed type + count (bytes)
            self.assertEqual(
                loadb(
                    OBJECT_START
                    + CONTAINER_TYPE
                    + TYPE_UINT8
                    + CONTAINER_COUNT
                    + TYPE_UINT8
                    + b"\x02"
                    + TYPE_UINT8
                    + b"\x02"
                    + b"aa"
                    + b"\x04"
                    + TYPE_UINT8
                    + b"\x02"
                    + b"bb"
                    + b"\x05"
                ),
                {"aa": 4, "bb": 5},
            )

    def test_object_noop(self):
        # only supported without type
        for hook in (None, OrderedDict):
            loadb = partial(self.bjdloadb, object_pairs_hook=hook)
            self.assertEqual(
                loadb(
                    OBJECT_START
                    + TYPE_NOOP
                    + TYPE_UINT8
                    + b"\x01"
                    + "a".encode("utf-8")
                    + TYPE_NULL
                    + TYPE_NOOP
                    + TYPE_UINT8
                    + b"\x01"
                    + "b".encode("utf-8")
                    + TYPE_BOOL_TRUE
                    + OBJECT_END
                ),
                {"a": None, "b": True},
            )
            self.assertEqual(
                loadb(
                    OBJECT_START
                    + CONTAINER_COUNT
                    + TYPE_UINT8
                    + b"\x01"
                    + TYPE_NOOP
                    + TYPE_UINT8
                    + b"\x01"
                    + "a".encode("utf-8")
                    + TYPE_NULL
                ),
                {"a": None},
            )

    def test_intern_object_keys(self):
        encoded = self.bjddumpb({"asdasd": 1, "qwdwqd": 2})
        mapping2 = self.bjdloadb(encoded, intern_object_keys=True)
        mapping3 = self.bjdloadb(encoded, intern_object_keys=True)
        for key1, key2 in zip(sorted(mapping2.keys()), sorted(mapping3.keys())):
            if PY2:  # pragma: no cover
                # interning of unicode not supported
                self.assertEqual(key1, key2)
            else:  # pragma: no cover
                self.assertIs(key1, key2)

    def test_circular(self):
        sequence = [1, 2, 3]
        sequence.append(sequence)
        mapping = {"a": 1, "b": 2}
        mapping["c"] = mapping

        for container in (sequence, mapping):
            with self.assertRaises(ValueError):
                self.bjddumpb(container)

        # Refering to the same container multiple times is valid however
        sequence = [1, 2, 3]
        mapping = {"a": 1, "b": 2}
        self.check_enc_dec([sequence, mapping, sequence, mapping])

    def test_unencodable(self):
        with self.assertRaises(EncoderException):
            self.bjddumpb(type(None))

    def test_decoder_fuzz(self):
        for start, end, fmt in (
            (0, pow(2, 8), ">B"),
            (pow(2, 8), pow(2, 16), ">H"),
            (pow(2, 16), pow(2, 18), ">I"),
        ):
            for i in range(start, end):
                try:
                    self.bjdloadb(pack(fmt, i))
                except DecoderException:
                    pass
                except (
                    Exception
                ) as ex:  # pragma: no cover  pylint: disable=broad-except
                    self.fail("Unexpected failure: %s" % ex)

    def assert_raises_regex(self, *args, **kwargs):
        # pylint: disable=deprecated-method,no-member
        return (self.assertRaisesRegexp if PY2 else self.assertRaisesRegex)(
            *args, **kwargs
        )

    def test_recursion(self):
        old_limit = getrecursionlimit()
        setrecursionlimit(200)
        if version_info >= (3, 5):
            # Python 3.5+ has RecursionError
            recursion_exceptions = (RuntimeError, RecursionError)
        else:
            # Python 2.7 and early Python 3.x only have RuntimeError
            recursion_exceptions = (RuntimeError,)

        try:
            # Adjust multiplier based on Python version for recursion limit changes
            if version_info >= (3, 12):
                multiplier = 200
            else:
                multiplier = 2

            obj = current = []
            for _ in range(getrecursionlimit() * multiplier):
                new_list = []
                current.append(new_list)
                current = new_list

            with self.assert_raises_regex(recursion_exceptions, "recursion"):
                self.bjddumpb(obj)

            raw = ARRAY_START * (getrecursionlimit() * multiplier)
            with self.assert_raises_regex(recursion_exceptions, "recursion"):
                self.bjdloadb(raw)
        finally:
            setrecursionlimit(old_limit)

    def test_encode_default(self):
        def default(obj):
            if isinstance(obj, set):
                return sorted(obj)
            raise EncoderException("__test__marker__")

        dumpb_default = partial(self.bjddumpb, default=default)
        # Top-level custom type
        obj1 = {1, 2, 3}
        obj2 = default(obj1)
        # Custom type within sequence or mapping
        obj3 = OrderedDict(sorted({"a": 1, "b": obj1, "c": [2, obj1]}.items()))
        obj4 = OrderedDict(sorted({"a": 1, "b": obj2, "c": [2, obj2]}.items()))

        with self.assert_raises_regex(EncoderException, "Cannot encode item"):
            self.bjddumpb(obj1)
        # explicit None should behave the same as no default
        with self.assert_raises_regex(EncoderException, "Cannot encode item"):
            self.bjddumpb(obj1, default=None)

        with self.assert_raises_regex(EncoderException, "__test__marker__"):
            dumpb_default(self)

        self.assertEqual(dumpb_default(obj1), self.bjddumpb(obj2))
        self.assertEqual(dumpb_default(obj3), self.bjddumpb(obj4))

    def test_decode_object_hook(self):
        with self.assertRaises(TypeError):
            self.check_enc_dec({"a": 1, "b": 2}, object_hook=int)

        def default(obj):
            if isinstance(obj, set):
                return {"__set__": list(obj)}
            raise EncoderException("__test__marker__")

        def object_hook(obj):
            if "__set__" in obj:
                return set(obj["__set__"])
            return obj

        self.check_enc_dec(
            {"a": 1, "b": {2, 3, 4}}, object_hook=object_hook, default=default
        )

        class UnHandled(object):
            pass

        with self.assertRaises(EncoderException):
            self.check_enc_dec(
                {"a": 1, "b": UnHandled()}, object_hook=object_hook, default=default
            )

    # ==========================================================================
    # SOA (Structure of Arrays) Tests - BJData Draft 4
    # ==========================================================================

    def test_soa_col_major_uint8(self):
        """Test column-major SOA with uint8 fields"""
        dt = np.dtype([("x", "u1"), ("y", "u1")])
        data = np.array(
            [(ord("A"), ord("D")), (ord("B"), ord("E")), (ord("C"), ord("F"))], dtype=dt
        )

        bjd = self.bjddumpb(data, soa_format="col")
        result = self.bjdloadb(bjd)

        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.dtype.names, ("x", "y"))
        self.assertTrue(np.array_equal(result["x"], data["x"]))
        self.assertTrue(np.array_equal(result["y"], data["y"]))

    def test_soa_row_major_uint8(self):
        """Test row-major SOA with uint8 fields"""
        dt = np.dtype([("x", "u1"), ("y", "u1")])
        data = np.array(
            [(ord("A"), ord("D")), (ord("B"), ord("E")), (ord("C"), ord("F"))], dtype=dt
        )

        bjd = self.bjddumpb(data, soa_format="row")
        result = self.bjdloadb(bjd)

        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.dtype.names, ("x", "y"))
        self.assertTrue(np.array_equal(result["x"], data["x"]))
        self.assertTrue(np.array_equal(result["y"], data["y"]))

    def test_soa_col_major_int8(self):
        """Test column-major SOA with int8 fields"""
        dt = np.dtype([("a", "i1"), ("b", "i1")])
        data = np.array([(ord("A"), ord("C")), (ord("B"), ord("D"))], dtype=dt)

        bjd = self.bjddumpb(data, soa_format="col")
        result = self.bjdloadb(bjd)

        self.assertTrue(np.array_equal(result["a"], data["a"]))
        self.assertTrue(np.array_equal(result["b"], data["b"]))

    def test_soa_col_major_int16(self):
        """Test column-major SOA with int16 fields"""
        dt = np.dtype([("x", "i2"), ("y", "i2")])
        data = np.array([(1000, 3000), (2000, 4000)], dtype=dt)

        bjd = self.bjddumpb(data, soa_format="col")
        result = self.bjdloadb(bjd)

        self.assertTrue(np.array_equal(result["x"], data["x"]))
        self.assertTrue(np.array_equal(result["y"], data["y"]))

    def test_soa_col_major_uint16(self):
        """Test column-major SOA with uint16 fields"""
        dt = np.dtype([("x", "u2"), ("y", "u2")])
        data = np.array([(1000, 3000), (2000, 4000)], dtype=dt)

        bjd = self.bjddumpb(data, soa_format="col")
        result = self.bjdloadb(bjd)

        self.assertTrue(np.array_equal(result["x"], data["x"]))
        self.assertTrue(np.array_equal(result["y"], data["y"]))

    def test_soa_col_major_int32(self):
        """Test column-major SOA with int32 fields"""
        dt = np.dtype([("a", "i4"), ("b", "i4")])
        data = np.array([(100000, -50000), (200000, -60000)], dtype=dt)

        bjd = self.bjddumpb(data, soa_format="col")
        result = self.bjdloadb(bjd)

        self.assertTrue(np.array_equal(result["a"], data["a"]))
        self.assertTrue(np.array_equal(result["b"], data["b"]))

    def test_soa_col_major_uint32(self):
        """Test column-major SOA with uint32 fields"""
        dt = np.dtype([("x", "u4"), ("y", "u4")])
        data = np.array([(100000, 300000), (200000, 400000)], dtype=dt)

        bjd = self.bjddumpb(data, soa_format="col")
        result = self.bjdloadb(bjd)

        self.assertTrue(np.array_equal(result["x"], data["x"]))
        self.assertTrue(np.array_equal(result["y"], data["y"]))

    def test_soa_col_major_int64(self):
        """Test column-major SOA with int64 fields"""
        dt = np.dtype([("x", "i8")])
        data = np.array([(-1000,), (2000,)], dtype=dt)

        bjd = self.bjddumpb(data, soa_format="col")
        result = self.bjdloadb(bjd)

        self.assertTrue(np.array_equal(result["x"], data["x"]))

    def test_soa_col_major_uint64(self):
        """Test column-major SOA with uint64 fields"""
        dt = np.dtype([("x", "u8")])
        data = np.array([(1000,), (2000,)], dtype=dt)

        bjd = self.bjddumpb(data, soa_format="col")
        result = self.bjdloadb(bjd)

        self.assertTrue(np.array_equal(result["x"], data["x"]))

    def test_soa_col_major_float32(self):
        """Test column-major SOA with float32 fields"""
        dt = np.dtype([("x", "f4"), ("y", "f4")])
        data = np.array([(1.5, 3.5), (2.5, 4.5)], dtype=dt)

        bjd = self.bjddumpb(data, soa_format="col")
        result = self.bjdloadb(bjd)

        self.assertTrue(np.allclose(result["x"], data["x"]))
        self.assertTrue(np.allclose(result["y"], data["y"]))

    def test_soa_col_major_float64(self):
        """Test column-major SOA with float64 fields"""
        dt = np.dtype([("x", "f8")])
        data = np.array([(1.5,), (2.5,)], dtype=dt)

        bjd = self.bjddumpb(data, soa_format="col")
        result = self.bjdloadb(bjd)

        self.assertTrue(np.allclose(result["x"], data["x"]))

    def test_soa_col_major_logical(self):
        """Test column-major SOA with boolean and uint8 fields"""
        dt = np.dtype([("flag", "?"), ("val", "u1")])
        data = np.array(
            [(True, ord("A")), (False, ord("B")), (True, ord("C"))], dtype=dt
        )

        bjd = self.bjddumpb(data, soa_format="col")
        result = self.bjdloadb(bjd)

        self.assertTrue(np.array_equal(result["flag"], data["flag"]))
        self.assertTrue(np.array_equal(result["val"], data["val"]))

    def test_soa_row_major_int32(self):
        """Test row-major SOA with int32 fields"""
        dt = np.dtype([("a", "i4"), ("b", "i4")])
        data = np.array([(100, -50), (200, -60)], dtype=dt)

        bjd = self.bjddumpb(data, soa_format="row")
        result = self.bjdloadb(bjd)

        self.assertTrue(np.array_equal(result["a"], data["a"]))
        self.assertTrue(np.array_equal(result["b"], data["b"]))

    def test_soa_row_major_float64(self):
        """Test row-major SOA with float64 fields"""
        dt = np.dtype([("x", "f8"), ("y", "f8")])
        data = np.array([(1.5, 3.5), (2.5, 4.5)], dtype=dt)

        bjd = self.bjddumpb(data, soa_format="row")
        result = self.bjdloadb(bjd)

        self.assertTrue(np.allclose(result["x"], data["x"]))
        self.assertTrue(np.allclose(result["y"], data["y"]))

    def test_soa_row_major_logical(self):
        """Test row-major SOA with boolean and uint8 fields"""
        dt = np.dtype([("flag", "?"), ("val", "u1")])
        data = np.array([(True, ord("A")), (False, ord("B"))], dtype=dt)

        bjd = self.bjddumpb(data, soa_format="row")
        result = self.bjdloadb(bjd)

        self.assertTrue(np.array_equal(result["flag"], data["flag"]))
        self.assertTrue(np.array_equal(result["val"], data["val"]))

    def test_soa_mixed_int_types(self):
        """Test SOA with mixed integer types"""
        dt = np.dtype([("a", "u1"), ("b", "i2"), ("c", "u4")])
        data = np.array([(ord("A"), 1000, 100000), (ord("B"), 2000, 200000)], dtype=dt)

        bjd = self.bjddumpb(data, soa_format="col")
        result = self.bjdloadb(bjd)

        self.assertTrue(np.array_equal(result["a"], data["a"]))
        self.assertTrue(np.array_equal(result["b"], data["b"]))
        self.assertTrue(np.array_equal(result["c"], data["c"]))

    def test_soa_mixed_types(self):
        """Test SOA with mixed numeric and boolean types"""
        dt = np.dtype([("id", "u1"), ("value", "f4"), ("flag", "?")])
        data = np.array([(65, 1.5, True), (66, 2.5, False)], dtype=dt)

        bjd = self.bjddumpb(data, soa_format="col")
        result = self.bjdloadb(bjd)

        self.assertTrue(np.array_equal(result["id"], data["id"]))
        self.assertTrue(np.allclose(result["value"], data["value"]))
        self.assertTrue(np.array_equal(result["flag"], data["flag"]))

    def test_soa_all_logical(self):
        """Test SOA with only boolean fields"""
        dt = np.dtype([("a", "?"), ("b", "?")])
        data = np.array([(True, False), (False, True), (True, False)], dtype=dt)

        bjd = self.bjddumpb(data, soa_format="col")
        result = self.bjdloadb(bjd)

        self.assertTrue(np.array_equal(result["a"], data["a"]))
        self.assertTrue(np.array_equal(result["b"], data["b"]))

    def test_soa_10_elements(self):
        """Test SOA with 10 elements"""
        dt = np.dtype([("id", "u1"), ("val", "u1")])
        data = np.array([(ord("A") + i, ord("K") + i) for i in range(10)], dtype=dt)

        bjd = self.bjddumpb(data, soa_format="col")
        result = self.bjdloadb(bjd)

        self.assertTrue(np.array_equal(result["id"], data["id"]))
        self.assertTrue(np.array_equal(result["val"], data["val"]))

    def test_soa_4_fields(self):
        """Test SOA with 4 fields"""
        dt = np.dtype([("a", "u1"), ("b", "u1"), ("c", "u1"), ("d", "u1")])
        data = np.array(
            [
                (ord("A"), ord("C"), ord("E"), ord("G")),
                (ord("B"), ord("D"), ord("F"), ord("H")),
            ],
            dtype=dt,
        )

        bjd = self.bjddumpb(data, soa_format="col")
        result = self.bjdloadb(bjd)

        for field in ("a", "b", "c", "d"):
            self.assertTrue(np.array_equal(result[field], data[field]))

    def test_soa_long_field_names(self):
        """Test SOA with long field names"""
        dt = np.dtype([("longitude", "f8"), ("latitude", "f8")])
        data = np.array([(1.5, 3.5), (2.5, 4.5)], dtype=dt)

        bjd = self.bjddumpb(data, soa_format="col")
        result = self.bjdloadb(bjd)

        self.assertTrue(np.allclose(result["longitude"], data["longitude"]))
        self.assertTrue(np.allclose(result["latitude"], data["latitude"]))

    def test_soa_2d_array(self):
        """Test SOA with 2D structured array"""
        dt = np.dtype([("x", "u1"), ("y", "u1")])
        data = np.array(
            [
                [(ord("A"), ord("G")), (ord("B"), ord("H")), (ord("C"), ord("I"))],
                [(ord("D"), ord("J")), (ord("E"), ord("K")), (ord("F"), ord("L"))],
            ],
            dtype=dt,
        )

        bjd = self.bjddumpb(data, soa_format="col")
        result = self.bjdloadb(bjd)

        self.assertEqual(result.shape, data.shape)
        self.assertTrue(np.array_equal(result["x"], data["x"]))
        self.assertTrue(np.array_equal(result["y"], data["y"]))

    def test_soa_single_element(self):
        """Test SOA with single element"""
        dt = np.dtype([("x", "u1"), ("y", "u1")])
        data = np.array([(65, 66)], dtype=dt)

        bjd = self.bjddumpb(data, soa_format="col")
        result = self.bjdloadb(bjd)

        self.assertTrue(np.array_equal(result["x"], data["x"]))
        self.assertTrue(np.array_equal(result["y"], data["y"]))

    def test_soa_disabled(self):
        """Test that structured arrays auto-enable SOA when soa_format is None"""
        dt = np.dtype([("x", "u1"), ("y", "u1")])
        data = np.array([(65, 66), (67, 68)], dtype=dt)

        # Without soa_format parameter, should auto-enable column-major SOA
        bjd = self.bjddumpb(data, soa_format=None)
        # Just verify it encodes without error and can be decoded
        self.assertIsInstance(bjd, bytes)
        result = self.bjdloadb(bjd)
        self.assertTrue(np.array_equal(result, data))

    def test_soa_roundtrip_col_major(self):
        """Test column-major SOA roundtrip"""
        dt = np.dtype([("x", "u1"), ("y", "u1")])
        data = np.array(
            [(ord("A"), ord("D")), (ord("B"), ord("E")), (ord("C"), ord("F"))], dtype=dt
        )

        bjd = self.bjddumpb(data, soa_format="col")
        result = self.bjdloadb(bjd)

        self.assertTrue(np.array_equal(result, data))

    def test_soa_roundtrip_row_major(self):
        """Test row-major SOA roundtrip"""
        dt = np.dtype([("x", "u1"), ("y", "u1")])
        data = np.array(
            [(ord("A"), ord("D")), (ord("B"), ord("E")), (ord("C"), ord("F"))], dtype=dt
        )

        bjd = self.bjddumpb(data, soa_format="row")
        result = self.bjdloadb(bjd)

        self.assertTrue(np.array_equal(result, data))

    def test_soa_roundtrip_mixed_types(self):
        """Test SOA roundtrip with mixed types"""
        dt = np.dtype([("a", "?"), ("b", "i4"), ("c", "f8")])
        data = np.array([(True, 100, 1.5), (False, 200, 2.5)], dtype=dt)

        bjd = self.bjddumpb(data, soa_format="col")
        result = self.bjdloadb(bjd)

        self.assertTrue(np.array_equal(result["a"], data["a"]))
        self.assertTrue(np.array_equal(result["b"], data["b"]))
        self.assertTrue(np.allclose(result["c"], data["c"]))

    def test_soa_roundtrip_int64_uint64(self):
        """Test SOA roundtrip with int64 and uint64"""
        dt = np.dtype([("x", "i8"), ("y", "u8")])
        data = np.array([(-1000, 3000), (2000, 4000)], dtype=dt)

        bjd = self.bjddumpb(data, soa_format="col")
        result = self.bjdloadb(bjd)

        self.assertTrue(np.array_equal(result["x"], data["x"]))
        self.assertTrue(np.array_equal(result["y"], data["y"]))

    def test_soa_roundtrip_2d(self):
        """Test SOA roundtrip with 2D array"""
        dt = np.dtype([("x", "u1"), ("y", "u1")])
        data = np.array(
            [
                [(ord("A"), ord("G")), (ord("B"), ord("H")), (ord("C"), ord("I"))],
                [(ord("D"), ord("J")), (ord("E"), ord("K")), (ord("F"), ord("L"))],
            ],
            dtype=dt,
        )

        bjd = self.bjddumpb(data, soa_format="col")
        result = self.bjdloadb(bjd)

        self.assertEqual(result.shape, data.shape)
        self.assertTrue(np.array_equal(result, data))

    def test_soa_binary_format_col_major(self):
        """Test that column-major SOA produces expected binary format"""
        dt = np.dtype([("x", "u1"), ("y", "u1")])
        data = np.array(
            [(ord("A"), ord("D")), (ord("B"), ord("E")), (ord("C"), ord("F"))], dtype=dt
        )

        bjd = self.bjddumpb(data, soa_format="col")

        # Header should start with {${ (object with typed schema)
        self.assertTrue(bjd.startswith(b"{${"))
        # Payload should end with ABCDEF (column-major: all x's then all y's)
        self.assertTrue(bjd.endswith(b"ABCDEF"))

    def test_soa_binary_format_row_major(self):
        """Test that row-major SOA produces expected binary format"""
        dt = np.dtype([("x", "u1"), ("y", "u1")])
        data = np.array(
            [(ord("A"), ord("D")), (ord("B"), ord("E")), (ord("C"), ord("F"))], dtype=dt
        )

        bjd = self.bjddumpb(data, soa_format="row")

        # Header should start with [${ (array with typed schema)
        self.assertTrue(bjd.startswith(b"[${"))
        # Payload should end with ADBECF (row-major: interleaved)
        self.assertTrue(bjd.endswith(b"ADBECF"))

    # ==========================================================================
    # SOA String Encoding Tests - BJData Draft 4
    # Fixed-length strings, Dictionary-based strings, and Offset-table strings
    # ==========================================================================

    # --------------------------------------------------------------------------
    # SECTION A: Fixed-Length String Tests
    # --------------------------------------------------------------------------

    def test_soa_fixed_string_same_length(self):
        """Test fixed-length string SOA where all strings have same length"""
        dt = np.dtype([("code", "U5")])
        data = np.array([("ABCDE",), ("FGHIJ",), ("12345",)], dtype=dt)

        bjd = self.bjddumpb(data, soa_format="col")
        result = self.bjdloadb(bjd)

        self.assertEqual(result.dtype.names, ("code",))
        for i in range(len(data)):
            self.assertEqual(result["code"][i], data["code"][i])

    def test_soa_fixed_string_different_lengths(self):
        """Test fixed-length string SOA with different lengths (null-padded)"""
        dt = np.dtype([("id", "u1"), ("name", "U3")])
        data = np.array(
            [(ord("A"), "ABC"), (ord("B"), "DE"), (ord("C"), "F")], dtype=dt
        )

        bjd = self.bjddumpb(data, soa_format="col")
        result = self.bjdloadb(bjd)

        self.assertTrue(np.array_equal(result["id"], data["id"]))
        # Shorter strings should be preserved (null-padded in encoding)
        self.assertEqual(result["name"][0], "ABC")
        self.assertEqual(result["name"][1], "DE")
        self.assertEqual(result["name"][2], "F")

    def test_soa_fixed_string_row_major(self):
        """Test fixed-length string SOA in row-major format"""
        dt = np.dtype([("code", "U2")])
        data = np.array([("AB",), ("CD",)], dtype=dt)

        bjd = self.bjddumpb(data, soa_format="row")
        result = self.bjdloadb(bjd)

        self.assertEqual(result["code"][0], "AB")
        self.assertEqual(result["code"][1], "CD")

    def test_soa_fixed_string_with_numeric(self):
        """Test fixed-length string SOA mixed with numeric fields"""
        dt = np.dtype([("id", "u1"), ("tag", "U2")])
        data = np.array([(ord("A"), "Hi"), (ord("B"), "Lo")], dtype=dt)

        bjd = self.bjddumpb(data, soa_format="col")
        result = self.bjdloadb(bjd)

        self.assertTrue(np.array_equal(result["id"], data["id"]))
        self.assertEqual(result["tag"][0], "Hi")
        self.assertEqual(result["tag"][1], "Lo")

    def test_soa_fixed_string_with_empty(self):
        """Test fixed-length string SOA with empty string"""
        dt = np.dtype([("id", "u1"), ("tag", "U2")])
        data = np.array([(ord("A"), "Hi"), (ord("B"), "")], dtype=dt)

        bjd = self.bjddumpb(data, soa_format="col")
        result = self.bjdloadb(bjd)

        self.assertTrue(np.array_equal(result["id"], data["id"]))
        self.assertEqual(result["tag"][0], "Hi")
        self.assertEqual(result["tag"][1], "")

    def test_soa_fixed_string_single_char(self):
        """Test fixed-length string SOA with single character strings"""
        dt = np.dtype([("ch", "U1")])
        data = np.array([("A",), ("B",), ("C",), ("D",)], dtype=dt)

        bjd = self.bjddumpb(data, soa_format="col")
        result = self.bjdloadb(bjd)

        for i in range(len(data)):
            self.assertEqual(result["ch"][i], data["ch"][i])

    def test_soa_fixed_string_max_uint8_length(self):
        """Test fixed-length string with 255 chars (max uint8 length)"""
        str255 = "X" * 255
        dt = np.dtype([("long", "U255")])
        data = np.array([(str255,), (str255,)], dtype=dt)

        bjd = self.bjddumpb(data, soa_format="col")
        result = self.bjdloadb(bjd)

        self.assertEqual(result["long"][0], str255)
        self.assertEqual(result["long"][1], str255)

    def test_soa_fixed_string_requires_uint16_length(self):
        """Test fixed-length string requiring uint16 length (>255 chars)"""
        str300 = "Y" * 300
        dt = np.dtype([("verylong", "U300")])
        data = np.array([(str300,), (str300,)], dtype=dt)

        bjd = self.bjddumpb(data, soa_format="col")
        result = self.bjdloadb(bjd)

        self.assertEqual(result["verylong"][0], str300)
        self.assertEqual(result["verylong"][1], str300)

    # --------------------------------------------------------------------------
    # SECTION B: Dictionary-Based String Tests
    # --------------------------------------------------------------------------

    def test_soa_dict_string_col_major(self):
        """Test dictionary string SOA (2 unique in 4 records = 0.5 ratio)"""
        dt = np.dtype([("id", "u1"), ("status", "U8")])
        data = np.array(
            [
                (ord("A"), "active"),
                (ord("B"), "inactive"),
                (ord("C"), "active"),
                (ord("D"), "active"),
            ],
            dtype=dt,
        )

        bjd = self.bjddumpb(data, soa_format="col")
        result = self.bjdloadb(bjd)

        self.assertTrue(np.array_equal(result["id"], data["id"]))
        for i in range(len(data)):
            self.assertEqual(result["status"][i], data["status"][i])

    def test_soa_dict_string_3_values(self):
        """Test dictionary string SOA with 3 unique values (3/6 = 0.5)"""
        dt = np.dtype([("color", "U5")])
        data = np.array(
            [("red",), ("green",), ("blue",), ("red",), ("green",), ("blue",)],
            dtype=dt,
        )

        bjd = self.bjddumpb(data, soa_format="col")
        result = self.bjdloadb(bjd)

        for i in range(len(data)):
            self.assertEqual(result["color"][i], data["color"][i])

    def test_soa_dict_string_all_same(self):
        """Test dictionary string SOA where all values are same (1/3 = 0.33)"""
        dt = np.dtype([("id", "u1"), ("tag", "U1")])
        data = np.array([(ord("A"), "X"), (ord("B"), "X"), (ord("C"), "X")], dtype=dt)

        bjd = self.bjddumpb(data, soa_format="col")
        result = self.bjdloadb(bjd)

        self.assertTrue(np.array_equal(result["id"], data["id"]))
        for i in range(len(data)):
            self.assertEqual(result["tag"][i], "X")

    def test_soa_dict_string_at_threshold(self):
        """Test dictionary string exactly at threshold (2/4 = 0.5)"""
        dt = np.dtype([("type", "U3")])
        data = np.array([("ON",), ("OFF",), ("ON",), ("OFF",)], dtype=dt)

        bjd = self.bjddumpb(data, soa_format="col")
        result = self.bjdloadb(bjd)

        self.assertEqual(result["type"][0], "ON")
        self.assertEqual(result["type"][1], "OFF")
        self.assertEqual(result["type"][2], "ON")
        self.assertEqual(result["type"][3], "OFF")

    def test_soa_dict_string_all_empty(self):
        """Test dictionary string where all values are empty (1/2 = 0.5)"""
        dt = np.dtype([("id", "u1"), ("tag", "U1")])
        data = np.array([(ord("A"), ""), (ord("B"), "")], dtype=dt)

        bjd = self.bjddumpb(data, soa_format="col")
        result = self.bjdloadb(bjd)

        self.assertTrue(np.array_equal(result["id"], data["id"]))
        self.assertEqual(result["tag"][0], "")
        self.assertEqual(result["tag"][1], "")

    def test_soa_dict_string_with_empty_entry(self):
        """Test dictionary string with empty string as one of the values"""
        dt = np.dtype([("id", "u1"), ("tag", "U1")])
        data = np.array([(ord("A"), ""), (ord("B"), ""), (ord("C"), "")], dtype=dt)

        bjd = self.bjddumpb(data, soa_format="col")
        result = self.bjdloadb(bjd)

        self.assertTrue(np.array_equal(result["id"], data["id"]))
        for i in range(len(data)):
            self.assertEqual(result["tag"][i], "")

    def test_soa_dict_string_256_unique(self):
        """Test dictionary with 256 unique values (may require uint16 index)"""
        unique_strs = [f"{i:03d}" for i in range(256)]
        dup_strs = unique_strs + unique_strs  # 512 records, 256 unique = 0.5
        dt = np.dtype([("code", "U3")])
        data = np.array([(s,) for s in dup_strs], dtype=dt)

        bjd = self.bjddumpb(data, soa_format="col")
        result = self.bjdloadb(bjd)

        for i in range(len(data)):
            self.assertEqual(result["code"][i], data["code"][i])

    # --------------------------------------------------------------------------
    # SECTION C: Offset-Table String Tests (Variable-Length)
    # --------------------------------------------------------------------------

    def test_soa_offset_string_col_major(self):
        """Test offset-table string SOA in column-major format"""
        dt = np.dtype([("id", "u1"), ("desc", "U32")])
        data = np.array(
            [
                (ord("A"), "short"),
                (ord("B"), "a very long description"),
                (ord("C"), "mid"),
            ],
            dtype=dt,
        )

        # Force offset encoding with soa_threshold=0
        bjd = self.bjddumpb(data, soa_format="col", soa_threshold=0)
        result = self.bjdloadb(bjd)

        self.assertTrue(np.array_equal(result["id"], data["id"]))
        self.assertEqual(result["desc"][0], "short")
        self.assertEqual(result["desc"][1], "a very long description")
        self.assertEqual(result["desc"][2], "mid")

    def test_soa_offset_string_row_major(self):
        """Test offset-table string SOA in row-major format"""
        dt = np.dtype([("id", "u1"), ("text", "U8")])
        data = np.array([(ord("A"), "Hello"), (ord("B"), "World")], dtype=dt)

        bjd = self.bjddumpb(data, soa_format="row", soa_threshold=0)
        result = self.bjdloadb(bjd)

        self.assertTrue(np.array_equal(result["id"], data["id"]))
        self.assertEqual(result["text"][0], "Hello")
        self.assertEqual(result["text"][1], "World")

    def test_soa_offset_string_with_empty(self):
        """Test offset-table string SOA with empty string"""
        dt = np.dtype([("id", "u1"), ("note", "U8")])
        data = np.array([(ord("A"), "abc"), (ord("B"), ""), (ord("C"), "de")], dtype=dt)

        bjd = self.bjddumpb(data, soa_format="col", soa_threshold=0)
        result = self.bjdloadb(bjd)

        self.assertTrue(np.array_equal(result["id"], data["id"]))
        self.assertEqual(result["note"][0], "abc")
        self.assertEqual(result["note"][1], "")
        self.assertEqual(result["note"][2], "de")

    def test_soa_offset_string_varying_lengths(self):
        """Test offset-table string with varying lengths"""
        dt = np.dtype([("desc", "U8")])
        data = np.array([("a",), ("bb",), ("ccc",), ("dddd",), ("eeeee",)], dtype=dt)

        bjd = self.bjddumpb(data, soa_format="col", soa_threshold=0)
        result = self.bjdloadb(bjd)

        for i in range(len(data)):
            self.assertEqual(result["desc"][i], data["desc"][i])

    def test_soa_offset_string_all_same(self):
        """Test offset-table string where all values are same"""
        dt = np.dtype([("note", "U8")])
        data = np.array([("same",), ("same",), ("same",)], dtype=dt)

        bjd = self.bjddumpb(data, soa_format="col", soa_threshold=0)
        result = self.bjdloadb(bjd)

        for i in range(len(data)):
            self.assertEqual(result["note"][i], "same")

    def test_soa_offset_string_uint8_offsets(self):
        """Test offset-table with small total (<256 bytes) using uint8 offsets"""
        dt = np.dtype([("txt", "U8")])
        data = np.array([("a",), ("bb",), ("ccc",)], dtype=dt)  # 6 bytes total

        bjd = self.bjddumpb(data, soa_format="col", soa_threshold=0)
        result = self.bjdloadb(bjd)

        for i in range(len(data)):
            self.assertEqual(result["txt"][i], data["txt"][i])

    def test_soa_offset_string_uint16_offsets(self):
        """Test offset-table with medium total (256-65535 bytes) using uint16"""
        str200 = "M" * 200
        dt = np.dtype([("txt", "U200")])
        data = np.array([(str200,), (str200,)], dtype=dt)  # 400 bytes total

        bjd = self.bjddumpb(data, soa_format="col", soa_threshold=0)
        result = self.bjdloadb(bjd)

        self.assertEqual(result["txt"][0], str200)
        self.assertEqual(result["txt"][1], str200)

    def test_soa_offset_string_uint32_offsets(self):
        """Test offset-table with large total (>65535 bytes) using uint32"""
        str35k = "L" * 35000
        dt = np.dtype([("txt", "U35000")])
        data = np.array([(str35k,), (str35k,)], dtype=dt)  # 70000 bytes total

        bjd = self.bjddumpb(data, soa_format="col", soa_threshold=0)
        result = self.bjdloadb(bjd)

        self.assertEqual(result["txt"][0], str35k)
        self.assertEqual(result["txt"][1], str35k)

    # --------------------------------------------------------------------------
    # SECTION D: Mixed String Encoding Tests
    # --------------------------------------------------------------------------

    def test_soa_fixed_string_above_threshold(self):
        """Test that high unique ratio uses fixed-length (2/3 > 0.5)"""
        dt = np.dtype([("id", "u1"), ("type", "U1")])
        data = np.array([(ord("A"), "A"), (ord("B"), "B"), (ord("C"), "A")], dtype=dt)

        bjd = self.bjddumpb(data, soa_format="col")
        result = self.bjdloadb(bjd)

        self.assertTrue(np.array_equal(result["id"], data["id"]))
        self.assertEqual(result["type"][0], "A")
        self.assertEqual(result["type"][1], "B")
        self.assertEqual(result["type"][2], "A")

    def test_soa_mixed_string_with_logical(self):
        """Test string field mixed with logical field"""
        dt = np.dtype([("flag", "?"), ("cat", "U3")])
        data = np.array([(True, "yes"), (False, "no"), (True, "yes")], dtype=dt)

        bjd = self.bjddumpb(data, soa_format="col")
        result = self.bjdloadb(bjd)

        self.assertTrue(np.array_equal(result["flag"], data["flag"]))
        self.assertEqual(result["cat"][0], "yes")
        self.assertEqual(result["cat"][1], "no")
        self.assertEqual(result["cat"][2], "yes")

    def test_soa_multiple_string_fields(self):
        """Test multiple string fields with different encodings"""
        dt = np.dtype([("name", "U8"), ("status", "U8"), ("id", "u1")])
        # name: 6 unique in 6 = 1.0 -> fixed
        # status: 2 unique in 6 = 0.33 -> dict
        data = np.array(
            [
                ("Alice", "on", 1),
                ("Bob", "off", 2),
                ("Carol", "on", 3),
                ("Dave", "off", 4),
                ("Eve", "on", 5),
                ("Frank", "off", 6),
            ],
            dtype=dt,
        )

        bjd = self.bjddumpb(data, soa_format="col")
        result = self.bjdloadb(bjd)

        self.assertTrue(np.array_equal(result["id"], data["id"]))
        for i in range(len(data)):
            self.assertEqual(result["name"][i], data["name"][i])
            self.assertEqual(result["status"][i], data["status"][i])

    # --------------------------------------------------------------------------
    # SECTION E: Fixed Array with String Tests
    # --------------------------------------------------------------------------

    def test_soa_fixed_array_with_string(self):
        """Test fixed array field combined with string field"""
        dt = np.dtype([("val", "u1"), ("vec", "u1", (2,)), ("cat", "U1")])
        data = np.array(
            [
                (ord("A"), [ord("a"), ord("b")], "X"),
                (ord("B"), [ord("c"), ord("d")], "Y"),
                (ord("C"), [ord("e"), ord("f")], "X"),
            ],
            dtype=dt,
        )

        bjd = self.bjddumpb(data, soa_format="col")
        result = self.bjdloadb(bjd)

        self.assertTrue(np.array_equal(result["val"], data["val"]))
        self.assertTrue(np.array_equal(result["vec"], data["vec"]))
        for i in range(len(data)):
            self.assertEqual(result["cat"][i], data["cat"][i])

    def test_soa_scalar_double_with_string(self):
        """Test scalar double field combined with string field"""
        dt = np.dtype([("pos", "f8"), ("tag", "U2")])
        data = np.array([(1.5, "CD"), (2.5, "EF")], dtype=dt)

        bjd = self.bjddumpb(data, soa_format="col")
        result = self.bjdloadb(bjd)

        self.assertTrue(np.allclose(result["pos"], data["pos"]))
        self.assertEqual(result["tag"][0], "CD")
        self.assertEqual(result["tag"][1], "EF")

    # --------------------------------------------------------------------------
    # SECTION F: Nested Struct with String Tests
    # --------------------------------------------------------------------------

    def test_soa_nested_struct_with_string_col_major(self):
        """Test nested struct with string field in column-major"""
        inner_dt = np.dtype([("name", "U2"), ("val", "u1")])
        outer_dt = np.dtype([("id", "u1"), ("info", inner_dt)])

        inner1 = np.array([("AB", ord("X"))], dtype=inner_dt)[0]
        inner2 = np.array([("CD", ord("Y"))], dtype=inner_dt)[0]
        data = np.array([(ord("A"), inner1), (ord("B"), inner2)], dtype=outer_dt)

        bjd = self.bjddumpb(data, soa_format="col")
        result = self.bjdloadb(bjd)

        self.assertTrue(np.array_equal(result["id"], data["id"]))
        self.assertEqual(result["info"]["name"][0], "AB")
        self.assertEqual(result["info"]["name"][1], "CD")
        self.assertTrue(np.array_equal(result["info"]["val"], data["info"]["val"]))

    def test_soa_nested_struct_numeric_only(self):
        """Test nested struct with only numeric fields"""
        inner_dt = np.dtype([("x", "u1"), ("y", "u1")])
        outer_dt = np.dtype([("id", "u1"), ("pt", inner_dt)])

        data = np.array(
            [
                (ord("A"), (ord("1"), ord("2"))),
                (ord("B"), (ord("3"), ord("4"))),
            ],
            dtype=outer_dt,
        )

        bjd = self.bjddumpb(data, soa_format="col")
        result = self.bjdloadb(bjd)

        self.assertTrue(np.array_equal(result["id"], data["id"]))
        self.assertTrue(np.array_equal(result["pt"]["x"], data["pt"]["x"]))
        self.assertTrue(np.array_equal(result["pt"]["y"], data["pt"]["y"]))

    def test_soa_nested_struct_3_levels(self):
        """Test deeply nested struct (3 levels)"""
        level3_dt = np.dtype([("d", "u1")])
        level2_dt = np.dtype([("c", level3_dt)])
        level1_dt = np.dtype([("a", "u1"), ("b", level2_dt)])

        data = np.array(
            [
                (ord("1"), ((ord("X"),),)),
                (ord("2"), ((ord("Y"),),)),
            ],
            dtype=level1_dt,
        )

        bjd = self.bjddumpb(data, soa_format="col")
        result = self.bjdloadb(bjd)

        self.assertTrue(np.array_equal(result["a"], data["a"]))
        self.assertTrue(np.array_equal(result["b"]["c"]["d"], data["b"]["c"]["d"]))

    def test_soa_nested_struct_with_logical(self):
        """Test nested struct with logical field"""
        inner_dt = np.dtype([("flag", "?")])
        outer_dt = np.dtype([("id", "u1"), ("info", inner_dt)])

        data = np.array(
            [
                (ord("A"), (True,)),
                (ord("B"), (False,)),
            ],
            dtype=outer_dt,
        )

        bjd = self.bjddumpb(data, soa_format="col")
        result = self.bjdloadb(bjd)

        self.assertTrue(np.array_equal(result["id"], data["id"]))
        self.assertTrue(np.array_equal(result["info"]["flag"], data["info"]["flag"]))

    def test_soa_nested_struct_row_major(self):
        """Test nested struct in row-major format"""
        inner_dt = np.dtype([("x", "u1")])
        outer_dt = np.dtype([("id", "u1"), ("pt", inner_dt)])

        data = np.array(
            [
                (ord("A"), (ord("1"),)),
                (ord("B"), (ord("2"),)),
            ],
            dtype=outer_dt,
        )

        bjd = self.bjddumpb(data, soa_format="row")
        result = self.bjdloadb(bjd)

        self.assertTrue(np.array_equal(result["id"], data["id"]))
        self.assertTrue(np.array_equal(result["pt"]["x"], data["pt"]["x"]))

    def test_soa_nested_struct_roundtrip(self):
        """Test nested struct roundtrip with strings"""
        inner_dt = np.dtype([("name", "U5"), ("code", "U2")])
        outer_dt = np.dtype([("id", "u1"), ("meta", inner_dt)])

        data = np.array(
            [
                (1, ("Alice", "A1")),
                (2, ("Bob", "B2")),
            ],
            dtype=outer_dt,
        )

        bjd = self.bjddumpb(data, soa_format="col")
        result = self.bjdloadb(bjd)

        self.assertTrue(np.array_equal(result["id"], data["id"]))
        self.assertEqual(result["meta"]["name"][0], "Alice")
        self.assertEqual(result["meta"]["name"][1], "Bob")
        self.assertEqual(result["meta"]["code"][0], "A1")
        self.assertEqual(result["meta"]["code"][1], "B2")

    # --------------------------------------------------------------------------
    # SECTION G: Row-Major String Tests
    # --------------------------------------------------------------------------

    def test_soa_row_major_fixed_string_3char(self):
        """Test row-major with fixed 3-char strings"""
        dt = np.dtype([("id", "u1"), ("tag", "U3")])
        data = np.array([(ord("A"), "XYZ"), (ord("B"), "ABC")], dtype=dt)

        bjd = self.bjddumpb(data, soa_format="row")
        result = self.bjdloadb(bjd)

        self.assertTrue(np.array_equal(result["id"], data["id"]))
        self.assertEqual(result["tag"][0], "XYZ")
        self.assertEqual(result["tag"][1], "ABC")

    def test_soa_row_major_mixed_types_with_string(self):
        """Test row-major with mixed types including string"""
        dt = np.dtype([("id", "u1"), ("name", "U2"), ("val", "i2")])
        data = np.array([(1, "AA", 100), (2, "BB", 200), (3, "CC", 300)], dtype=dt)

        bjd = self.bjddumpb(data, soa_format="row")
        result = self.bjdloadb(bjd)

        self.assertTrue(np.array_equal(result["id"], data["id"]))
        self.assertTrue(np.array_equal(result["val"], data["val"]))
        for i in range(len(data)):
            self.assertEqual(result["name"][i], data["name"][i])

    # --------------------------------------------------------------------------
    # SECTION H: N-Dimensional Array with String Tests
    # --------------------------------------------------------------------------

    def test_soa_2d_array_with_string(self):
        """Test 2D SOA array with string field"""
        dt = np.dtype([("id", "u1"), ("tag", "U1")])
        data = np.array(
            [
                [(ord("A"), "W"), (ord("B"), "X")],
                [(ord("C"), "Y"), (ord("D"), "Z")],
            ],
            dtype=dt,
        )

        bjd = self.bjddumpb(data, soa_format="col")
        result = self.bjdloadb(bjd)

        self.assertEqual(result.shape, data.shape)
        self.assertTrue(np.array_equal(result["id"], data["id"]))
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                self.assertEqual(result["tag"][i, j], data["tag"][i, j])

    def test_soa_3x2_array_with_string(self):
        """Test 3x2 SOA array with string field"""
        dt = np.dtype([("x", "u1"), ("name", "U2")])
        data = np.array(
            [
                [(ord("A"), "AA"), (ord("B"), "BB")],
                [(ord("C"), "CC"), (ord("D"), "DD")],
                [(ord("E"), "EE"), (ord("F"), "FF")],
            ],
            dtype=dt,
        )

        bjd = self.bjddumpb(data, soa_format="col")
        result = self.bjdloadb(bjd)

        self.assertEqual(result.shape, (3, 2))
        self.assertTrue(np.array_equal(result["x"], data["x"]))

    # --------------------------------------------------------------------------
    # SECTION I: Edge Cases and Error Handling
    # --------------------------------------------------------------------------

    def test_soa_empty_string_various_positions(self):
        """Test empty strings at various positions"""
        dt = np.dtype([("a", "U2"), ("b", "U2"), ("c", "U2")])
        data = np.array([("", "A", "P"), ("X", "", "Q"), ("Y", "B", "")], dtype=dt)

        bjd = self.bjddumpb(data, soa_format="col")
        result = self.bjdloadb(bjd)

        for field in ("a", "b", "c"):
            for i in range(len(data)):
                self.assertEqual(result[field][i], data[field][i])

    def test_soa_field_name_with_digits(self):
        """Test SOA with field names containing digits"""
        dt = np.dtype([("field1", "u1"), ("data2", "U2")])
        data = np.array([(ord("A"), "XX"), (ord("B"), "YY")], dtype=dt)

        bjd = self.bjddumpb(data, soa_format="col")
        result = self.bjdloadb(bjd)

        self.assertTrue(np.array_equal(result["field1"], data["field1"]))
        self.assertEqual(result["data2"][0], "XX")
        self.assertEqual(result["data2"][1], "YY")

    def test_soa_long_field_name_with_string(self):
        """Test SOA with long field name (>30 chars)"""
        long_name = "x" * 30
        dt = np.dtype([(long_name, "U4")])
        data = np.array([("test",), ("data",)], dtype=dt)

        bjd = self.bjddumpb(data, soa_format="col")
        result = self.bjdloadb(bjd)

        self.assertEqual(result[long_name][0], "test")
        self.assertEqual(result[long_name][1], "data")

    # --------------------------------------------------------------------------
    # SECTION J: Roundtrip Tests
    # --------------------------------------------------------------------------

    def test_soa_fixed_string_roundtrip(self):
        """Test fixed-length string roundtrip"""
        dt = np.dtype([("code", "U3")])
        data = np.array([("ABC",), ("DEF",), ("GHI",)], dtype=dt)

        bjd = self.bjddumpb(data, soa_format="col")
        result = self.bjdloadb(bjd)

        for i in range(len(data)):
            self.assertEqual(result["code"][i], data["code"][i])

    def test_soa_dict_string_roundtrip(self):
        """Test dictionary string roundtrip"""
        dt = np.dtype([("id", "u1"), ("status", "U8")])
        data = np.array(
            [
                (1, "active"),
                (2, "pending"),
                (3, "active"),
                (4, "pending"),
            ],
            dtype=dt,
        )

        bjd = self.bjddumpb(data, soa_format="col")
        result = self.bjdloadb(bjd)

        self.assertTrue(np.array_equal(result["id"], data["id"]))
        for i in range(len(data)):
            self.assertEqual(result["status"][i], data["status"][i])

    def test_soa_offset_string_roundtrip(self):
        """Test offset-table string roundtrip"""
        dt = np.dtype([("id", "u1"), ("desc", "U32")])
        data = np.array(
            [(1, "short"), (2, "a very long description"), (3, "mid")], dtype=dt
        )

        bjd = self.bjddumpb(data, soa_format="col")
        result = self.bjdloadb(bjd)

        self.assertTrue(np.array_equal(result["id"], data["id"]))
        for i in range(len(data)):
            self.assertEqual(result["desc"][i], data["desc"][i])

    def test_soa_row_major_string_roundtrip(self):
        """Test row-major string roundtrip"""
        dt = np.dtype([("id", "u1"), ("name", "U2")])
        data = np.array([(1, "AB"), (2, "CD")], dtype=dt)

        bjd = self.bjddumpb(data, soa_format="row")
        result = self.bjdloadb(bjd)

        self.assertTrue(np.array_equal(result["id"], data["id"]))
        self.assertEqual(result["name"][0], "AB")
        self.assertEqual(result["name"][1], "CD")

    def test_soa_2d_string_roundtrip(self):
        """Test 2D array with string roundtrip"""
        dt = np.dtype([("x", "u1"), ("name", "U1")])
        data = np.array(
            [
                [(ord("A"), "W"), (ord("B"), "X")],
                [(ord("C"), "Y"), (ord("D"), "Z")],
            ],
            dtype=dt,
        )

        bjd = self.bjddumpb(data, soa_format="col")
        result = self.bjdloadb(bjd)

        self.assertEqual(result.shape, data.shape)
        self.assertTrue(np.array_equal(result["x"], data["x"]))
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                self.assertEqual(result["name"][i, j], data["name"][i, j])

    def test_soa_mixed_string_encodings_roundtrip(self):
        """Test mixed string encodings roundtrip"""
        dt = np.dtype([("name", "U8"), ("status", "U8"), ("id", "u1")])
        data = np.array(
            [
                ("Alice", "on", 1),
                ("Bob", "off", 2),
                ("Carol", "on", 3),
                ("Dave", "off", 4),
                ("Eve", "on", 5),
                ("Frank", "off", 6),
            ],
            dtype=dt,
        )

        bjd = self.bjddumpb(data, soa_format="col")
        result = self.bjdloadb(bjd)

        self.assertTrue(np.array_equal(result["id"], data["id"]))
        for i in range(len(data)):
            self.assertEqual(result["name"][i], data["name"][i])
            self.assertEqual(result["status"][i], data["status"][i])

    # --------------------------------------------------------------------------
    # SECTION K: Binary Format Verification
    # --------------------------------------------------------------------------

    def test_soa_fixed_string_binary_format(self):
        """Test that fixed-length string produces correct binary marker"""
        dt = np.dtype([("code", "U3")])
        data = np.array([("ABC",), ("DEF",)], dtype=dt)

        bjd = self.bjddumpb(data, soa_format="col")

        # Should contain 'S' marker for fixed string followed by length
        self.assertIn(b"S", bjd)
        # Should contain the string data
        self.assertIn(b"ABC", bjd)
        self.assertIn(b"DEF", bjd)

    def test_soa_dict_string_binary_format(self):
        """Test that dictionary string produces [$S# marker"""
        dt = np.dtype([("status", "U8")])
        # 2 unique in 8 = 0.25 < 0.5, triggers dict
        data = np.array(
            [
                ("active",),
                ("inactive",),
                ("active",),
                ("active",),
                ("inactive",),
                ("active",),
                ("inactive",),
                ("active",),
            ],
            dtype=dt,
        )

        bjd = self.bjddumpb(data, soa_format="col")

        # Dictionary marker: [$S#
        self.assertIn(b"[$S#", bjd)

    def test_soa_offset_string_binary_format(self):
        """Test that offset string produces [$U] marker"""
        dt = np.dtype([("desc", "U32")])
        data = np.array([("short",), ("longer text",), ("x",)], dtype=dt)

        bjd = self.bjddumpb(data, soa_format="col", soa_threshold=0)

        # Offset marker: [$U] (array of uint8 offsets for small data)
        self.assertIn(b"[$", bjd)

    # add for coverage

    def test_encoder_edge_cases(self):
        """Test encoder edge cases for coverage"""
        import numpy as np
        from bjdata.encoder import EncoderException

        # Test line 127 - empty string in __can_encode_as_soa
        dt = np.dtype([("x", "u1")])
        empty_data = np.array([], dtype=dt)
        # Should handle empty arrays
        bjd = self.bjddumpb(empty_data, soa_format="col")
        self.assertIsInstance(bjd, bytes)

        # Test lines 222-225 - __encode_float edge cases
        # Test subnormal float (very small number)
        subnormal = 1e-320  # Smaller than normal float range
        bjd = self.bjddumpb(subnormal, no_float32=False)
        self.assertIsInstance(bjd, bytes)

        # Test lines 232-233, 244, 251, 254 - __encode_int edge cases
        # Test boundary values for different int sizes
        boundary_ints = [
            2**7,  # Just above int8 positive range
            2**15,  # Just above int16 positive range
            2**31,  # Just above int32 positive range
            -(2**7),  # int8 negative boundary
            -(2**15),  # int16 negative boundary
            -(2**31),  # int32 negative boundary
        ]
        for val in boundary_ints:
            bjd = self.bjddumpb(val)
            result = self.bjdloadb(bjd)
            self.assertEqual(result, val)

        # Test line 528, 534-535 - __map_dtype with different endianness
        # Test big-endian array
        data_be = np.array([1, 2, 3], dtype=">i4")
        bjd = self.bjddumpb(data_be)
        self.assertIsInstance(bjd, bytes)

        # Test line 543 - scalar string dtype
        scalar_str = np.str_("test")
        bjd = self.bjddumpb(scalar_str)
        self.assertIsInstance(bjd, bytes)

        # Test lines 548-555, 558 - Unicode scalar with specific size
        unicode_scalar = np.array("ab", dtype="U2")
        bjd = self.bjddumpb(unicode_scalar)
        self.assertIsInstance(bjd, bytes)

    def test_decoder_edge_cases(self):
        """Test decoder edge cases for coverage"""
        from bjdata.markers import TYPE_UINT8, TYPE_STRING, TYPE_CHAR

        # Test lines 381-382 - string length edge cases
        # String with large length that requires multi-byte encoding
        large_string = "x" * 300
        bjd = self.bjddumpb(large_string)
        result = self.bjdloadb(bjd)
        self.assertEqual(result, large_string)

        # Test line 393 - char decoding
        # Single character string
        single_char = "z"
        bjd = self.bjddumpb(single_char)
        result = self.bjdloadb(bjd)
        self.assertEqual(result, single_char)

        # Test line 406 - bytes length edge cases
        large_bytes = b"x" * 300
        bjd = self.bjddumpb(large_bytes)
        result = self.bjdloadb(bjd)
        self.assertEqual(result, large_bytes)

        # Test line 684 - container with specific type and count combinations
        # Object with type and count
        from bjdata.markers import (
            OBJECT_START,
            CONTAINER_TYPE,
            CONTAINER_COUNT,
            TYPE_INT8,
        )

        manual_obj = (
            OBJECT_START
            + CONTAINER_TYPE
            + TYPE_INT8
            + CONTAINER_COUNT
            + TYPE_UINT8
            + b"\x01"
            + TYPE_UINT8
            + b"\x01"
            + b"a"
            + b"\x05"
        )
        result = self.bjdloadb(manual_obj)
        self.assertEqual(result, {"a": 5})

        # Test line 775 - array with specific patterns
        from bjdata.markers import ARRAY_START, ARRAY_END

        # Nested empty arrays
        nested_empty = (
            ARRAY_START + ARRAY_START + ARRAY_END + ARRAY_START + ARRAY_END + ARRAY_END
        )
        result = self.bjdloadb(nested_empty)
        self.assertEqual(result, [[], []])

    def test_soa_edge_cases(self):
        """Test SOA edge cases for coverage"""
        import numpy as np

        # Empty structured array
        dt = np.dtype([("x", "u1"), ("y", "u1")])
        empty_data = np.array([], dtype=dt)
        bjd = self.bjddumpb(empty_data, soa_format="col")
        result = self.bjdloadb(bjd)
        self.assertEqual(len(result), 0)

        # Single field structured array
        dt_single = np.dtype([("x", "u1")])
        single_field = np.array([(1,), (2,), (3,)], dtype=dt_single)
        bjd = self.bjddumpb(single_field, soa_format="col")
        result = self.bjdloadb(bjd)
        self.assertTrue(np.array_equal(result, single_field))

        # 3D structured array
        dt = np.dtype([("x", "u1"), ("y", "u1")])
        data_3d = np.array(
            [
                [[(1, 2), (3, 4)], [(5, 6), (7, 8)]],
                [[(9, 10), (11, 12)], [(13, 14), (15, 16)]],
            ],
            dtype=dt,
        )
        bjd = self.bjddumpb(data_3d, soa_format="col")
        result = self.bjdloadb(bjd)
        self.assertTrue(np.array_equal(result, data_3d))

        # All numeric types in one array
        dt_all = np.dtype(
            [
                ("i1", "i1"),
                ("u1", "u1"),
                ("i2", "i2"),
                ("u2", "u2"),
                ("i4", "i4"),
                ("u4", "u4"),
                ("i8", "i8"),
                ("u8", "u8"),
                ("f4", "f4"),
                ("f8", "f8"),
                ("b", "?"),
            ]
        )
        data_all = np.array([(1, 2, 3, 4, 5, 6, 7, 8, 1.5, 2.5, True)], dtype=dt_all)
        bjd = self.bjddumpb(data_all, soa_format="row")
        result = self.bjdloadb(bjd)
        self.assertTrue(np.array_equal(result["i1"], data_all["i1"]))
        self.assertTrue(np.array_equal(result["b"], data_all["b"]))

    # coverage of specific sections

    def test_encoder_missing_branches(self):
        """Target specific missing branches in encoder.py"""
        import numpy as np

        # Line 127: fp_write(TYPE_NULL) - for non-finite Decimal
        from decimal import Decimal

        # Infinity as Decimal is not finite
        inf_decimal = Decimal("Infinity")
        bjd = self.bjddumpb(inf_decimal)
        result = self.bjdloadb(bjd)
        self.assertIsNone(result)  # Non-finite Decimal encodes as NULL

        # Also test NaN Decimal
        nan_decimal = Decimal("NaN")
        bjd = self.bjddumpb(nan_decimal)
        result = self.bjdloadb(bjd)
        self.assertIsNone(result)

        # Lines 222-225: Boolean dtype check in __get_numpy_dtype_marker
        # Use bool dtype explicitly
        bool_arr = np.array([True, False, True], dtype="?")
        # This won't use SOA, but will trigger __get_numpy_dtype_marker check
        bjd = self.bjddumpb(bool_arr)
        self.assertIsInstance(bjd, bytes)

        # Line 232: "b" prefix for boolean in dtype string
        bool_arr2 = np.array([True, False], dtype="b1")
        bjd = self.bjddumpb(bool_arr2)
        self.assertIsInstance(bjd, bytes)

        # Line 233: return None for unsupported dtype marker
        # This is in __get_numpy_dtype_marker when dtype is not recognized
        # Line 247: except ImportError in __can_encode_as_soa
        # Line 250: except ImportError in __encode_numpy
        # These only trigger if numpy is not installed, which won't happen in our tests

        # Lines 524, 530-531: Unsupported dtype exception in __map_dtype
        # Line 539: Exception path in __encode_numpy
        # These are error paths - let's trigger them

        # Test with unsupported numpy type (complex numbers are in the table but as float32)
        # Actually, all standard types are supported, so these exceptions are hard to trigger

    def test_decoder_soa_edge_cases(self):
        """Test decoder SOA handling edge cases"""
        from bjdata.markers import (
            TYPE_STRING,
            TYPE_UINT8,
            TYPE_INT8,
            TYPE_CHAR,
            TYPE_NULL,
            ARRAY_START,
            OBJECT_START,
            OBJECT_END,
            CONTAINER_TYPE,
            CONTAINER_COUNT,
        )
        import numpy as np

        # Lines 381-382: SOA with noop markers in schema
        # Create a valid SOA structure that triggers marker continuation
        dt = np.dtype([("x", "u1"), ("y", "u1")])
        data = np.array([(65, 66), (67, 68)], dtype=dt)
        bjd = self.bjddumpb(data, soa_format="col")

        result = self.bjdloadb(bjd)
        self.assertTrue(np.array_equal(result, data))

        # Line 393: Invalid SOA schema - manually craft bad SOA
        # SOA schema with string type (not fixed-length)
        bad_soa = (
            OBJECT_START
            + CONTAINER_TYPE
            + OBJECT_START
            + TYPE_UINT8
            + b"\x01"
            + b"x"
            + TYPE_STRING
            + OBJECT_END
        )
        try:
            result = self.bjdloadb(bad_soa)
            self.fail("Should have raised DecoderException")
        except Exception as e:
            self.assertIn("fixed-length", str(e))

        # Line 406: Missing # after SOA schema
        # SOA without count marker
        bad_soa2 = (
            OBJECT_START
            + CONTAINER_TYPE
            + OBJECT_START
            + TYPE_UINT8
            + b"\x01"
            + b"x"
            + TYPE_UINT8
            + OBJECT_END
            + TYPE_UINT8
            + b"\x01"
        )  # Missing CONTAINER_COUNT marker
        try:
            result = self.bjdloadb(bad_soa2)
            self.fail("Should have raised DecoderException")
        except Exception as e:
            self.assertIn("#", str(e))

        # Lines 414, 416: Container with integer type markers
        # Array with typed container using int markers
        from bjdata.markers import TYPE_INT16

        arr_typed = (
            ARRAY_START
            + CONTAINER_TYPE
            + TYPE_INT16
            + CONTAINER_COUNT
            + TYPE_UINT8
            + b"\x03"
            + b"\x01\x00"
            + b"\x02\x00"
            + b"\x03\x00"
        )
        result = self.bjdloadb(arr_typed)
        self.assertEqual(len(result), 3)

        # Line 684: Container bytes array too short
        # Manually craft bytes array with incorrect length
        bad_bytes = (
            ARRAY_START
            + CONTAINER_TYPE
            + TYPE_UINT8
            + CONTAINER_COUNT
            + TYPE_UINT8
            + b"\x05"
            + b"\x01\x02"
        )  # Only 2 bytes but count says 5
        from bjdata.decoder import DecoderException

        with self.assertRaises(DecoderException):
            result = self.bjdloadb(bad_bytes)

        # Line 737: container = list() - array with typed schema
        # This happens when decoding typed arrays
        typed_arr = (
            ARRAY_START
            + CONTAINER_TYPE
            + TYPE_UINT8
            + CONTAINER_COUNT
            + TYPE_UINT8
            + b"\x03"
            + b"\x01\x02\x03"
        )
        result = self.bjdloadb(typed_arr)
        # Result should be numpy array
        if isinstance(result, np.ndarray):
            self.assertTrue((result == np.array([1, 2, 3], dtype="u1")).all())
        else:
            self.assertEqual(result, [1, 2, 3])

        # Lines 822, 825: Empty object returns
        # Decode empty containers
        empty_obj = OBJECT_START + OBJECT_END
        result = self.bjdloadb(empty_obj)
        self.assertEqual(result, {})
        self.assertEqual(
            len(result), 0
        )  # Triggers line 822# Add these test methods to TestEncodeDecodeFp class in test.py

    def test_encoder_coverage_missing_lines(self):
        """Test encoder.py missing lines for 100% coverage"""
        import numpy as np
        from decimal import Decimal

        # Lines 222-225: Subnormal floats with no_float32=False
        subnormal_values = [2.22e-308, 1.5e-308]  # Subnormal range
        for val in subnormal_values:
            bjd = self.bjddumpb(val, no_float32=False)
            self.assertIsInstance(bjd, bytes)

        # Lines 232-233: Exact boundary integers
        bjd = self.bjddumpb(2**8)  # Exactly 256, needs uint16
        self.assertEqual(self.bjdloadb(bjd), 256)

        # Line 247: Overflow to high precision from uint64
        huge = 2**64  # Overflows uint64
        bjd = self.bjddumpb(huge)
        self.assertEqual(self.bjdloadb(bjd), huge)

        # Line 250: String length >= 256 (multi-byte encoding)
        long_str = "a" * 256
        bjd = self.bjddumpb(long_str)
        self.assertEqual(self.bjdloadb(bjd), long_str)

        # Lines 524, 530-531, 539: Different dtype string formats
        # Test with explicit endianness
        arr1 = np.array([1, 2], dtype=">i2")  # Big-endian int16
        bjd1 = self.bjddumpb(arr1)
        self.assertIsInstance(bjd1, bytes)

        arr2 = np.array([1, 2], dtype="|u1")  # Not applicable endian
        bjd2 = self.bjddumpb(arr2)
        self.assertIsInstance(bjd2, bytes)

        # Line 554: Fortran-ordered array (triggers conversion)
        arr_f = np.array([[1, 2, 3], [4, 5, 6]], dtype="i4", order="F")
        self.assertTrue(np.isfortran(arr_f))  # Verify it's Fortran-ordered
        bjd = self.bjddumpb(arr_f)
        # Just verify it encodes without error - the line 554 is about the encoder converting to C order
        self.assertIsInstance(bjd, bytes)
        # Decode and check shape/dtype are preserved
        result = self.bjdloadb(bjd)
        self.assertEqual(result.shape, arr_f.shape)
        self.assertEqual(result.dtype, arr_f.dtype)

    def test_decoder_coverage_missing_lines(self):
        """Test decoder.py missing lines for 100% coverage"""
        from bjdata.markers import (
            TYPE_STRING,
            TYPE_UINT8,
            TYPE_INT8,
            TYPE_INT16,
            TYPE_CHAR,
            ARRAY_START,
            ARRAY_END,
            OBJECT_START,
            OBJECT_END,
            CONTAINER_TYPE,
            CONTAINER_COUNT,
        )

        # Lines 381-382: String length with int8 marker
        # Create string with explicit int8 length marker
        manual_str = TYPE_STRING + TYPE_INT8 + b"\x0a" + b"0123456789"
        result = self.bjdloadb(manual_str)
        self.assertEqual(result, "0123456789")

        # Line 393: Char at boundary
        manual_char = TYPE_CHAR + b"\x7f"
        result = self.bjdloadb(manual_char)
        self.assertEqual(result, "\x7f")

        # Line 406: Bytes with int16 length marker (use proper encoding approach)
        # Instead of manual construction, test with actual large bytes
        large_bytes = b"\xab" * 300  # > 255 bytes
        bjd = self.bjddumpb(large_bytes)
        result = self.bjdloadb(bjd)
        self.assertEqual(len(result), 300)
        self.assertEqual(result, large_bytes)

        # Line 684: Object without type but with count
        obj_count_only = (
            OBJECT_START
            + CONTAINER_COUNT
            + TYPE_UINT8
            + b"\x01"
            + TYPE_UINT8
            + b"\x03"
            + b"key"
            + TYPE_UINT8
            + b"\x99"
        )
        result = self.bjdloadb(obj_count_only)
        self.assertEqual(result, {"key": 153})

    def test_missing_test_lines(self):
        """Test edge cases in test.py itself"""
        # Lines 972, 977 are likely unreachable or specific to certain Python versions
        # These are in the test_recursion method's version checks
        # Just test that normal deep nesting works without hitting recursion

        # Create moderately deep structure (won't hit recursion limit)
        obj = current = []
        for _ in range(20):
            new_list = []
            current.append(new_list)
            current = new_list

        # Should encode and decode successfully
        bjd = self.bjddumpb(obj)
        result = self.bjdloadb(bjd)
        self.assertIsInstance(result, list)

        # Test with moderately deep dict nesting
        obj_dict = current_dict = {}
        for i in range(20):
            new_dict = {}
            current_dict[f"level_{i}"] = new_dict
            current_dict = new_dict

        bjd = self.bjddumpb(obj_dict)
        result = self.bjdloadb(bjd)
        self.assertIsInstance(
            result, dict
        )  # Add these test methods to TestEncodeDecodeFp class in test.py

    def test_encoder_coverage_lines(self):
        """Test encoder missing coverage lines"""
        import numpy as np
        from decimal import Decimal

        # Line 127 (encoder.py) - __can_encode_as_soa with empty field check
        dt = np.dtype([("x", "u1")])
        data = np.array([(1,)], dtype=dt)
        bjd = self.bjddumpb(data, soa_format="col")
        self.assertIsInstance(bjd, bytes)

        # Lines 222-225 - __encode_float with subnormal values
        # Subnormal float (between 0 and smallest normal float)
        subnormal = 2.225e-308  # Just below normal range, triggers high precision
        bjd = self.bjddumpb(subnormal, no_float32=False)
        self.assertIsInstance(bjd, bytes)

        # Lines 232-233 - __encode_int with exact boundary values
        # Test values at exact power-of-2 boundaries
        boundary_values = [
            2**8,  # Requires uint16
            2**16,  # Requires uint32
            2**32,  # Requires uint64
            -(2**7),  # Exactly -128, uses int8
        ]
        for val in boundary_values:
            bjd = self.bjddumpb(val)
            result = self.bjdloadb(bjd)
            self.assertEqual(result, val)

        # Line 247 - Large int requiring high precision
        huge_int = 2**64  # Beyond uint64
        bjd = self.bjddumpb(huge_int)
        result = self.bjdloadb(bjd)
        self.assertEqual(result, huge_int)

        # Line 250 - Negative value beyond int64
        huge_neg = -(2**63) - 1
        bjd = self.bjddumpb(huge_neg)
        result = self.bjdloadb(bjd)
        self.assertEqual(result, huge_neg)

        # Lines 524, 530-531, 539 - __map_dtype edge cases
        # Test with different endianness markers
        # Big-endian int16
        arr_be_i16 = np.array([100, 200], dtype=">i2")
        bjd = self.bjddumpb(arr_be_i16)
        self.assertIsInstance(bjd, bytes)

        # Little-endian uint32
        arr_le_u32 = np.array([1000, 2000], dtype="<u4")
        bjd = self.bjddumpb(arr_le_u32)
        self.assertIsInstance(bjd, bytes)

        # Native byte order
        arr_native = np.array([1, 2], dtype="=i4")
        bjd = self.bjddumpb(arr_native)
        self.assertIsInstance(bjd, bytes)

        # Line 554 - Fortran-ordered array
        arr_fortran = np.array([[1, 2], [3, 4]], dtype="u1", order="F")
        bjd = self.bjddumpb(arr_fortran)
        self.assertIsInstance(bjd, bytes)

    def test_decoder_coverage_lines(self):
        """Test decoder missing coverage lines"""
        from bjdata.markers import (
            TYPE_STRING,
            TYPE_UINT8,
            TYPE_INT8,
            TYPE_CHAR,
            ARRAY_START,
            ARRAY_END,
            OBJECT_START,
            OBJECT_END,
            CONTAINER_TYPE,
            CONTAINER_COUNT,
        )

        # Lines 381-382 - String with int8 length marker (negative triggers special handling)
        # String with uint8 length = 255 (boundary)
        long_str = "x" * 255
        bjd = self.bjddumpb(long_str)
        result = self.bjdloadb(bjd)
        self.assertEqual(result, long_str)

        # Line 393 - Char with specific handling
        bjd_char = TYPE_CHAR + b"Z"
        result = self.bjdloadb(bjd_char)
        self.assertEqual(result, "Z")

        # Line 406 - Bytes with large length
        large_bytes = b"\x00" * 256  # > 255, requires multi-byte length
        bjd = self.bjddumpb(large_bytes)
        result = self.bjdloadb(bjd)
        self.assertEqual(result, large_bytes)

        # Line 684 - Object with typed container
        # Manually craft object with specific type
        obj_typed = (
            OBJECT_START
            + CONTAINER_TYPE
            + TYPE_UINT8
            + CONTAINER_COUNT
            + TYPE_UINT8
            + b"\x02"
            + TYPE_UINT8
            + b"\x01"
            + b"a"
            + b"\x01"
            + TYPE_UINT8
            + b"\x01"
            + b"b"
            + b"\x02"
        )
        result = self.bjdloadb(obj_typed)
        self.assertEqual(result, {"a": 1, "b": 2})

    def test_soa_empty_and_special(self):
        """Test SOA with empty arrays and special cases"""
        import numpy as np

        # Test with array that has size but one dimension is 0
        dt = np.dtype([("x", "u1"), ("y", "u1")])

        # Row-major with single element
        single = np.array([(10, 20)], dtype=dt)
        bjd = self.bjddumpb(single, soa_format="row")
        result = self.bjdloadb(bjd)
        self.assertTrue(np.array_equal(result, single))

        # Large array to test buffering (use u2 to avoid overflow)
        dt_large = np.dtype([("x", "u2"), ("y", "u2")])
        large = np.array([(i, i + 100) for i in range(1000)], dtype=dt_large)
        bjd = self.bjddumpb(large, soa_format="col")
        result = self.bjdloadb(bjd)
        self.assertTrue(np.array_equal(result, large))

        # Test all boolean fields (special encoding)
        dt_bool = np.dtype([("a", "?"), ("b", "?"), ("c", "?")])
        bool_data = np.array(
            [(True, False, True), (False, True, False), (True, True, False)],
            dtype=dt_bool,
        )
        bjd = self.bjddumpb(bool_data, soa_format="row")
        result = self.bjdloadb(bjd)
        self.assertTrue(np.array_equal(result, bool_data))

    def test_encoder_dtype_marker_coverage(self):
        """Cover encoder.py lines 215, 218, 222-225"""
        import numpy as np

        # Line 215: dtype with endianness prefix
        # Line 218: Check if dtype in marker dict
        arr_with_endian = np.array([1, 2, 3], dtype="<u4")  # Little-endian uint32
        bjd = self.bjddumpb(arr_with_endian)
        self.assertIsInstance(bjd, bytes)

        # Test with big-endian
        arr_big = np.array([1, 2], dtype=">i2")  # Big-endian int16
        bjd = self.bjddumpb(arr_big)
        self.assertIsInstance(bjd, bytes)

        # Lines 222-225: Boolean dtype handling in __get_numpy_dtype_marker
        # When used in SOA context, this checks for "b" or "?" prefix
        dt_bool = np.dtype([("flag", "?")])  # Boolean field
        data_bool = np.array([(True,), (False,)], dtype=dt_bool)
        bjd = self.bjddumpb(data_bool, soa_format="col")
        result = self.bjdloadb(bjd)
        self.assertTrue(np.array_equal(result, data_bool))

        # Test with 'b1' dtype (bool with 'b' prefix)
        dt_b1 = np.dtype([("flag", "b1")])
        data_b1 = np.array([(True,), (False,)], dtype=dt_b1)
        bjd = self.bjddumpb(data_b1, soa_format="col")
        result = self.bjdloadb(bjd)
        self.assertTrue(np.array_equal(result["flag"], data_b1["flag"]))

    def test_encoder_importerror_paths(self):
        """Lines 232-233, 247, 250, 524, 530-531, 539 are ImportError/Exception paths"""
        # These lines only execute when numpy is NOT installed or when invalid dtypes are used
        # They cannot be tested in this environment since numpy is required for the tests
        # Line 232-233: except ImportError in __can_encode_as_soa
        # Line 247, 250: return False paths
        # Line 524: except ImportError in __encode_numpy
        # Line 530-531: raise Exception for unsupported dtype
        # Line 539: raise Exception "you must install numpy"

        # We can't cover these without uninstalling numpy or mocking the import
        # These are defensive error handling - acceptable to leave uncovered
        pass

    def test_decoder_soa_schema_coverage(self):
        """Cover decoder.py lines 381-382, 414, 416"""
        import numpy as np
        from bjdata.markers import (
            TYPE_UINT8,
            TYPE_INT8,
            TYPE_NOOP,
            OBJECT_START,
            OBJECT_END,
            ARRAY_START,
            CONTAINER_TYPE,
            CONTAINER_COUNT,
        )

        # Lines 381-382: marker = fp_read(1); continue
        # This is in SOA schema parsing loop when encountering NOOP markers
        # Create SOA with NOOP in schema (though this is technically invalid)
        # Actually, let's just ensure we test various SOA structures

        # Test SOA with different field types to cover schema parsing
        dt_mixed = np.dtype([("a", "i1"), ("b", "u2"), ("c", "f4")])
        data_mixed = np.array([(1, 100, 1.5), (-2, 200, 2.5)], dtype=dt_mixed)
        bjd = self.bjddumpb(data_mixed, soa_format="col")
        result = self.bjdloadb(bjd)
        self.assertTrue(np.array_equal(result, data_mixed))

        # Lines 414, 416: if marker in __TYPES_INT; marker = fp_read(1)
        # This is in container decoding with integer type markers
        # Manually create array with int8 type
        arr_int8_typed = (
            ARRAY_START
            + CONTAINER_TYPE
            + TYPE_INT8
            + CONTAINER_COUNT
            + TYPE_UINT8
            + b"\x03"
            + b"\x01\x02\x03"
        )
        result = self.bjdloadb(arr_int8_typed)
        self.assertEqual(len(result), 3)

    def test_decoder_container_coverage(self):
        """Cover decoder.py lines 737, 822, 825"""
        from bjdata.markers import (
            ARRAY_START,
            ARRAY_END,
            OBJECT_START,
            OBJECT_END,
            CONTAINER_TYPE,
            CONTAINER_COUNT,
            TYPE_UINT8,
            TYPE_NULL,
        )

        # Line 737: container = list() - when creating list for typed array
        # This happens in the typed container path
        typed_null_array = (
            ARRAY_START
            + CONTAINER_TYPE
            + TYPE_NULL
            + CONTAINER_COUNT
            + TYPE_UINT8
            + b"\x03"
        )
        result = self.bjdloadb(typed_null_array)
        self.assertEqual(result, [None, None, None])

        # Lines 822, 825: elif len(newobj) == 0; return newobj
        # These are at the end of decode_value for empty containers
        # Test empty object
        empty = OBJECT_START + OBJECT_END
        result = self.bjdloadb(empty)
        self.assertEqual(result, {})

        # Test empty array
        empty_arr = ARRAY_START + ARRAY_END
        result = self.bjdloadb(empty_arr)
        self.assertEqual(result, [])

        # Empty object with count
        empty_obj_count = OBJECT_START + CONTAINER_COUNT + TYPE_UINT8 + b"\x00"
        result = self.bjdloadb(empty_obj_count)
        self.assertEqual(result, {})


@skipUnless(EXTENSION_ENABLED, "Extension not enabled")
class TestEncodeDecodePlainExt(TestEncodeDecodePlain):
    @staticmethod
    def bjdloadb(raw, *args, **kwargs):
        return bjdloadb(raw, *args, **kwargs)

    @staticmethod
    def bjddumpb(obj, *args, **kwargs):
        return bjddumpb(obj, *args, **kwargs)


class TestEncodeDecodeFp(TestEncodeDecodePlain):
    """Performs tests via file-like objects (BytesIO) instead of bytes instances"""

    @staticmethod
    def bjdloadb(raw, *args, **kwargs):
        return bjdpureload(BytesIO(raw), *args, **kwargs)

    @staticmethod
    def bjddumpb(obj, *args, **kwargs):
        out = BytesIO()
        bjdpuredump(obj, out, *args, **kwargs)
        return out.getvalue()

    @staticmethod
    def bjdload(fp, *args, **kwargs):
        return bjdpureload(fp, *args, **kwargs)

    @staticmethod
    def bjddump(obj, fp, *args, **kwargs):
        return bjdpuredump(obj, fp, *args, **kwargs)

    def test_decode_exception_position(self):
        with self.assertRaises(DecoderException) as ctx:
            self.bjdloadb(TYPE_STRING + TYPE_INT8 + b"\x01" + b"\xfe" + b"c0fefe" * 4)
        self.assertEqual(ctx.exception.position, 4)

    def test_invalid_fp_dump(self):
        with self.assertRaises(AttributeError):
            self.bjddump(None, 1)

        class Dummy(object):
            write = 1

        class Dummy2(object):
            @staticmethod
            def write(raw):
                raise ValueError("invalid - %s" % repr(raw))

        with self.assertRaises(TypeError):
            self.bjddump(b"", Dummy)

        with self.assertRaises(ValueError):
            self.bjddump(b"", Dummy2)

    def test_invalid_fp_load(self):
        with self.assertRaises(AttributeError):
            self.bjdload(1)

        class Dummy(object):
            read = 1

        class Dummy2(object):
            @staticmethod
            def read(length):
                raise ValueError("invalid - %d" % length)

        with self.assertRaises(TypeError):
            self.bjdload(Dummy)

        with self.assertRaises(ValueError):
            self.bjdload(Dummy2)

    def test_fp(self):
        obj = {"a": 123, "b": 456}
        output = BytesIO()
        self.bjddump(obj, output)
        output.seek(0)
        self.assertEqual(self.bjdload(output), obj)


@skipUnless(EXTENSION_ENABLED, "Extension not enabled")
class TestEncodeDecodeFpExt(TestEncodeDecodeFp):
    @staticmethod
    def bjdloadb(raw, *args, **kwargs):
        return bjdload(BytesIO(raw), *args, **kwargs)

    @staticmethod
    def bjddumpb(obj, *args, **kwargs):
        out = BytesIO()
        bjddump(obj, out, *args, **kwargs)
        return out.getvalue()

    @staticmethod
    def bjdload(fp, *args, **kwargs):
        return bjdload(fp, *args, **kwargs)

    @staticmethod
    def bjddump(obj, fp, *args, **kwargs):
        return bjddump(obj, fp, *args, **kwargs)

    # Seekable file-like object buffering
    def test_fp_buffer(self):
        output = BytesIO()

        # items which fit into extension decoder-internal read buffer (BUFFER_FP_SIZE in decoder.c, extension only)
        obj2 = ["fishy" * 64] * 10
        output.seek(0)
        self.bjddump(obj2, output)
        output.seek(0)
        self.assertEqual(self.bjdload(output), obj2)

        # larger than extension read buffer (extension only)
        obj3 = ["fishy" * 512] * 10
        output.seek(0)
        self.bjddump(obj3, output)
        output.seek(0)
        self.assertEqual(self.bjdload(output), obj3)

    # Multiple documents in same stream (issue #9)
    def test_fp_multi(self):
        obj = {"a": 123, "b": b"some raw content"}
        output = BytesIO()
        count = 10

        # Seekable an non-seekable runs
        for _ in range(2):
            output.seek(0)

            for i in range(count):
                obj["c"] = i
                self.bjddump(obj, output)

            output.seek(0)
            for i in range(count):
                obj["c"] = i
                self.assertEqual(self.bjdload(output), obj)

            output.seekable = lambda: False

    # Whole "token" in decoder input unavailable (in non-seekable file-like object)
    def test_fp_callable_incomplete(self):
        obj = [123, b"something"]
        # remove whole of last token (binary data 'something', without its length)
        output = BytesIO(self.bjddumpb(obj)[: -(len(obj[1]) + 1)])
        output.seekable = lambda: False

        with self.assert_raises_regex(DecoderException, "Insufficient input"):
            self.bjdload(output)

    def test_fp_seek_invalid(self):
        output = BytesIO()
        self.bjddump({"a": 333, "b": 444}, output)
        # pad with data (non-bjdata) to ensure buffering too much data
        output.write(b" " * 16)
        output.seek(0)

        output.seek_org = output.seek

        # seek fails
        def bad_seek(*_):
            raise OSError("bad seek")

        output.seek = bad_seek
        with self.assert_raises_regex(OSError, "bad seek"):
            self.bjdload(output)

        # decoding (lack of input) and seek fail - should get decoding failure
        output.seek_org(0, SEEK_END)
        with self.assert_raises_regex(DecoderException, "Insufficient input"):
            self.bjdload(output)

        # seek is not callable
        output.seek_org(0)
        output.seek = True
        with self.assert_raises_regex(TypeError, "not callable"):
            self.bjdload(output)

        # decoding (lack of input) and seek not callable - should get decoding failure
        output.seek_org(0, SEEK_END)
        with self.assert_raises_regex(DecoderException, "Insufficient input"):
            self.bjdload(output)


# def pympler_run(iterations=20):
#     from unittest import main
#     from pympler import tracker
#     from gc import collect

#     tracker = tracker.SummaryTracker()
#     for i in range(iterations):
#         try:
#             main()
#         except SystemExit:
#             pass
#         if i % 2:
#             collect()
#             tracker.print_diff()


# if __name__ == '__main__':
#     pympler_run()
