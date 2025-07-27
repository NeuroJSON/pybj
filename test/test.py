# Copyright (c) 2020-2023 Qianqian Fang <q.fang at neu.edu>. All rights reserved.
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


from sys import version_info, getrecursionlimit, setrecursionlimit
from functools import partial
from io import BytesIO, SEEK_END
from unittest import TestCase, skipUnless
from pprint import pformat
from decimal import Decimal
from struct import pack
from collections import OrderedDict

from bjdata import (dump as bjddump, dumpb as bjddumpb, load as bjdload, loadb as bjdloadb, EncoderException,
                    DecoderException, EXTENSION_ENABLED)
from bjdata.markers import (TYPE_NULL, TYPE_NOOP, TYPE_BOOL_TRUE, TYPE_BOOL_FALSE, TYPE_BYTE, TYPE_INT8, TYPE_UINT8, TYPE_INT16,
                            TYPE_INT32, TYPE_INT64, TYPE_UINT16, TYPE_UINT32, TYPE_UINT64, TYPE_FLOAT16, TYPE_FLOAT32, TYPE_FLOAT64,
                            TYPE_HIGH_PREC, TYPE_CHAR, TYPE_STRING, OBJECT_START, OBJECT_END, ARRAY_START, ARRAY_END,
                            CONTAINER_TYPE, CONTAINER_COUNT)
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
        return obj if isinstance(obj, unicode) else unicode(obj)  # noqa: F821 pylint: disable=undefined-variable
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
        return '\nInput:\n%s\nOutput (%d):\n%s' % (pformat(obj), len(encoded), encoded)

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

    def check_enc_dec(self, obj,
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
                      **kwargs):
        """Black-box test to check whether the provided object is the same once encoded and subsequently decoded."""
        encoded = self.bjddumpb(obj, **kwargs)

        if expected_type is not None:
            self.type_check(encoded[0], expected_type)
        if length is not None:
            assert_func = self.assertGreaterEqual if length_greater_or_equal else self.assertEqual
            assert_func(len(encoded), length, self.__format_in_out(obj, encoded))
        if approximate:
            self.assertTrue(self.numbers_close(self.bjdloadb(encoded, object_hook=object_hook,
                                                             object_pairs_hook=object_pairs_hook), obj),
                            msg=self.__format_in_out(obj, encoded))
        else:
            self.assertEqual(self.bjdloadb(encoded, object_hook=object_hook,
                                           object_pairs_hook=object_pairs_hook), obj,
                             self.__format_in_out(obj, encoded))

    def test_no_data(self):
        with self.assertRaises(DecoderException):
            self.bjdloadb(b'')

    def test_invalid_data(self):
        for invalid in (u('unicode'), 123):
            with self.assertRaises(TypeError):
                self.bjdloadb(invalid)

    def test_trailing_input(self):
        self.assertEqual(self.bjdloadb(TYPE_BOOL_TRUE * 10), True)

    def test_invalid_marker(self):
        with self.assertRaises(DecoderException) as ctx:
            self.bjdloadb(b'A')
        self.assertTrue(isinstance(ctx.exception.position, INTEGER_TYPES + (type(None),)))

    def test_bool(self):
        self.assertEqual(self.bjddumpb(True), TYPE_BOOL_TRUE)
        self.assertEqual(self.bjddumpb(False), TYPE_BOOL_FALSE)
        self.check_enc_dec(True, 1)
        self.check_enc_dec(False, 1)

    def test_null(self):
        self.assertEqual(self.bjddumpb(None), TYPE_NULL)
        self.check_enc_dec(None, 1)

    def test_char(self):
        self.assertEqual(self.bjddumpb(u('a')), TYPE_CHAR + 'a'.encode('utf-8'))
        # no char, char invalid utf-8
        for suffix in (b'', b'\xfe'):
            with self.assertRaises(DecoderException):
                self.bjdloadb(TYPE_CHAR + suffix)
        for char in (u('a'), u('\0'), u('~')):
            self.check_enc_dec(char, 2)

    def test_string(self):
        self.assertEqual(self.bjddumpb(u('ab')), TYPE_STRING + TYPE_UINT8 + b'\x02' + 'ab'.encode('utf-8'))
        self.check_enc_dec(u(''), 3)
        # invalid string size, string too short, string invalid utf-8
        for suffix in (b'\x81', b'\x01', b'\x01' + b'\xfe'):
            with self.assertRaises(DecoderException):
                self.bjdloadb(TYPE_STRING + TYPE_INT8 + suffix)
        # Note: In Python 2 plain str type is encoded as byte array
        for string in ('some ascii', u(r'\u00a9 with extended\u2122'), u('long string') * 100):
            self.check_enc_dec(string, 4, length_greater_or_equal=True)

    def test_int(self):
        self.assertEqual(self.bjddumpb(Decimal(-1.5)),
                         TYPE_HIGH_PREC + TYPE_UINT8 + b'\x04' + '-1.5'.encode('utf-8'))
        # insufficient length
        with self.assertRaises(DecoderException):
            self.bjdloadb(TYPE_INT16 + b'\x01')

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
                (TYPE_HIGH_PREC, 9999999999999999999999999999999999999, 40)):
            self.check_enc_dec(value, total_size, expected_type=type_)

        self.assertEqual((self.bjdloadb(b'[$U#U\x08\x01\x02\x03\x04\x05\x06\x07\x08', uint8_bytes=True)
            == b'\x01\x02\x03\x04\x05\x06\x07\x08'), True)

        self.assertEqual((self.bjdloadb(b'[$B#U\x08\x01\x02\x03\x04\x05\x06\x07\x08')
            == b'\x01\x02\x03\x04\x05\x06\x07\x08'), True)

        self.assertEqual((self.bjdloadb(b'[$u#U\x04\x01\x02\x03\x04\x05\x06\x07\x08')
            ==ndarray([513, 1027, 1541, 2055], np.uint16)).all(), True)

        self.assertEqual((self.bjdloadb(b'[$m#U\x02\x01\x02\x03\x04\x05\x06\x07\x08')
            ==ndarray([67305985, 134678021], np.uint32)).all(), True)

        self.assertEqual((self.bjdloadb(b'[$M#U\x01\x01\x02\x03\x04\x05\x06\x07\x08')
            ==ndarray([578437695752307201], np.uint64)).all(), True)

        self.assertEqual((self.bjdloadb(b'[$i#U\x08\xf1\xf2\xf3\xf4\xf5\xf6\xf7\xf8')
            ==ndarray([-15, -14, -13, -12, -11, -10,  -9,  -8], np.int8)).all(), True)

        self.assertEqual((self.bjdloadb(b'[$I#U\x04\xf1\xf2\xf3\xf4\xf5\xf6\xf7\xf8')
            ==ndarray([-3343, -2829, -2315, -1801], np.int16)).all(), True)

        self.assertEqual((self.bjdloadb(b'[$l#U\x02\xf1\xf2\xf3\xf4\xf5\xf6\xf7\xf8')
            ==ndarray([-185339151, -117967115], np.int32)).all(), True)

        self.assertEqual((self.bjdloadb(b'[$L#U\x01\xf1\xf2\xf3\xf4\xf5\xf6\xf7\xf8')
            ==ndarray([-506664896818842895], np.int64)).all(), True)

    def test_high_precision(self):
        self.assertEqual(self.bjddumpb(Decimal(-1.5)),
                         TYPE_HIGH_PREC + TYPE_UINT8 + b'\x04' + '-1.5'.encode('utf-8'))
        # insufficient length, invalid utf-8, invalid decimal value
        for suffix in (b'n', b'\xfe\xfe', b'na'):
            with self.assertRaises(DecoderException):
                self.bjdloadb(TYPE_HIGH_PREC + TYPE_UINT8 + b'\x02' + suffix)

        self.check_enc_dec('1.8e315')
        for value in (
                '0.0',
                '2.5',
                '10e30',
                '-1.2345e67890'):
            # minimum length because: marker + length marker + length + value
            self.check_enc_dec(Decimal(value), 4, length_greater_or_equal=True)
        # cannot compare equality, so test separately
        for value in ('nan', '-inf', 'inf'):
            self.assertEqual(str(self.bjdloadb(self.bjddumpb(float(value)))), value)

    def test_float(self):
        # insufficient length
        for float_type in (TYPE_FLOAT32, TYPE_FLOAT64):
            with self.assertRaises(DecoderException):
                self.bjdloadb(float_type + b'\x01')

        self.check_enc_dec(0.0, 5, expected_type=TYPE_FLOAT32)

        for type_, value, total_size in (
                (TYPE_FLOAT32, 1.18e-37, 5),
                (TYPE_FLOAT32, 3.4e37, 5),
                (TYPE_FLOAT64, 2.23e-308, 9),
                (TYPE_FLOAT64, 12345.44e40, 9),
                (TYPE_FLOAT64, 1.8e307, 9)):
            self.check_enc_dec(value,
                               total_size,
                               approximate=True,
                               expected_type=type_,
                               no_float32=False)
            # using only float64 (default)
            self.check_enc_dec(value,
                               9 if type_ == TYPE_FLOAT32 else total_size,
                               approximate=True,
                               expected_type=(TYPE_FLOAT64 if type_ == TYPE_FLOAT32 else type_))
        for value in ('nan', '-inf', 'inf'):
            for no_float32 in (True, False):
                self.assertEqual(str(self.bjdloadb(self.bjddumpb(float(value), no_float32=no_float32))), value)

        # value which results in high_prec usage
        for no_float32 in (True, False):
            self.check_enc_dec(2.22e-308, 4, expected_type=TYPE_HIGH_PREC, length_greater_or_equal=True,
                               no_float32=no_float32)

    def test_array(self):
        # invalid length
        with self.assertRaises(DecoderException):
            self.bjdloadb(ARRAY_START + CONTAINER_COUNT + self.bjddumpb(-5))
        # unencodable type within
        with self.assertRaises(EncoderException):
            self.bjddumpb([type(None)])
        for sequence in list, tuple:
            self.assertEqual(self.bjddumpb(sequence()), ARRAY_START + ARRAY_END)
        self.assertEqual(self.bjddumpb((None,), container_count=True), (ARRAY_START + CONTAINER_COUNT + TYPE_UINT8 +
                                                                        b'\x01' + TYPE_NULL))
        obj = [123,
               1.25,
               43121609.5543,
               12345.44e40,
               Decimal('10e15'),
               'a',
               'here is a string',
               None,
               True,
               False,
               [[1, 2], 3, [4, 5, 6], 7],
               {'a dict': 456}]
        for opts in ({'container_count': False}, {'container_count': True}):
            self.check_enc_dec(obj, **opts)

    def test_bytes(self):
        # insufficient length
        with self.assertRaises(DecoderException):
            self.bjdloadb(ARRAY_START + CONTAINER_TYPE + TYPE_BYTE + CONTAINER_COUNT + TYPE_UINT8 + b'\x02' + b'\x01')
        for cast in (bytes, bytearray):
            self.check_enc_dec(cast(b''))
            self.check_enc_dec(cast(b'\x01' * 4))
            self.assertEqual((self.bjdloadb(self.bjddumpb(cast(b'\x04' * 4)), no_bytes=True) == ndarray([4] * 4, npint8)).all(), True)
            self.check_enc_dec(cast(b'largebinary' * 100))

    def test_nd_array(self):
        raw_start = (ARRAY_START + CONTAINER_TYPE + TYPE_INT8 + CONTAINER_COUNT + ARRAY_START + \
                    CONTAINER_TYPE + TYPE_INT8 + CONTAINER_COUNT + TYPE_UINT8 + b'\x02' + \
                    b'\x03' + b'\x02' + b'\x01'+ b'\x02'+ b'\x03'+ b'\x04'+ b'\x05'+ b'\x06')
        self.assertEqual((self.bjdloadb(raw_start)==ndarray([[1,2],[3,4],[5,6]], npint8)).all(), True)

        self.assertEqual((self.bjdloadb(self.bjddumpb(np.uint8(9)))  == np.uint8(9)), True)
        self.assertEqual((self.bjdloadb(self.bjddumpb(np.int8(9)))   == np.int8(9)), True)
        self.assertEqual((self.bjdloadb(self.bjddumpb(np.uint16(6))) == np.uint16(6)), True)
        self.assertEqual((self.bjdloadb(self.bjddumpb(np.int16(6)))  == np.int16(6)), True)
        self.assertEqual((self.bjdloadb(self.bjddumpb(np.uint32(6))) == np.uint32(6)), True)
        self.assertEqual((self.bjdloadb(self.bjddumpb(np.int32(6)))  == np.int32(6)), True)
        self.assertEqual((self.bjdloadb(self.bjddumpb(np.uint64(6))) == np.uint64(6)), True)
        self.assertEqual((self.bjdloadb(self.bjddumpb(np.int64(5)))  == np.int64(5)), True)

        self.assertEqual((self.bjdloadb(self.bjddumpb(np.eye(2, dtype=np.uint8))) == np.eye(2, dtype=np.uint8)).all(), True)
        self.assertEqual((self.bjdloadb(self.bjddumpb(np.eye(2, dtype=np.int16))) == np.eye(2, dtype=np.int16)).all(), True)

        self.assertEqual((self.bjdloadb(self.bjddumpb(np.float16(2.2)))  == 16486), True)
        self.assertEqual((self.bjdloadb(self.bjddumpb(np.float32(2.2)))  == np.float32(2.2)), True)

        self.assertEqual((self.bjdloadb(self.bjddumpb(np.array([1.3,-0.5,0.7,1000,11], dtype=np.float32))) == np.array([1.3,-0.5,0.7,1000,11], dtype=np.float32)).all(), True)
        self.assertEqual((self.bjdloadb(self.bjddumpb(np.array([1,2,3,4], dtype=np.uint16))) == np.array([1,2,3,4], dtype=np.uint16)).all(), True)
        self.assertEqual((self.bjdloadb(self.bjddumpb(np.array([1,2,3,4], dtype=np.uint8))) == np.array([1,2,3,4], dtype=np.uint8)).all(), True)
        self.assertEqual((self.bjdloadb(self.bjddumpb(np.array([], dtype=np.int8))) == np.array([], dtype=np.int8)).all(), True)
        self.assertEqual((self.bjdloadb(self.bjddumpb(np.array([-1,-2,-3,-4], dtype=np.float32))) == np.array([-1,-2,-3,-4], dtype=np.float32)).all(), True)
        self.assertEqual((self.bjdloadb(self.bjddumpb(np.array([[-1,-2,5],[-3,-4,-6]], dtype=np.float64))) == np.array([[-1,-2,5],[-3,-4,-6]], dtype=np.float64)).all(), True)

        raw_start = (ARRAY_START + CONTAINER_TYPE + TYPE_INT8 + CONTAINER_COUNT + ARRAY_START + \
                    TYPE_UINT8 + b'\x03' + TYPE_UINT16 + b'\x02' + b'\x00' + ARRAY_END + \
                    b'\x01'+ b'\x02'+ b'\x03'+ b'\x04'+ b'\x05'+ b'\x06')
        self.assertEqual((self.bjdloadb(raw_start)==ndarray([[1,2],[3,4],[5,6]], npint8)).all(), True)

    def test_array_fixed(self):
        raw_start = ARRAY_START + CONTAINER_TYPE + TYPE_INT8 + CONTAINER_COUNT + TYPE_UINT8
        self.assertEqual(self.bjdloadb(raw_start + b'\x00'), [])

        # fixed types + count
        for bjd_type, py_obj in ((TYPE_NULL, None), (TYPE_BOOL_TRUE, True), (TYPE_BOOL_FALSE, False)):
            self.assertEqual(
                self.bjdloadb(ARRAY_START + CONTAINER_TYPE + bjd_type + CONTAINER_COUNT + TYPE_UINT8 + b'\x05'),
                [py_obj] * 5
            )
        self.assertEqual((self.bjdloadb(raw_start + b'\x03' + (b'\x01' * 3))==ndarray([1, 1, 1], dtype=npint8)).all(), True)

        # invalid type
        with self.assertRaises(DecoderException):
            self.bjdloadb(ARRAY_START + CONTAINER_TYPE + b'\x01')

        # type without count
        with self.assertRaises(DecoderException):
            self.bjdloadb(ARRAY_START + CONTAINER_TYPE + TYPE_INT8 + b'\x01')

        # count without type
        self.assertEqual(self.bjdloadb(ARRAY_START + CONTAINER_COUNT + TYPE_UINT8 + b'\x02' + TYPE_BOOL_FALSE +
                                       TYPE_BOOL_TRUE),
                         [False, True])

        # nested
        self.assertEqual(self.bjdloadb(ARRAY_START + CONTAINER_TYPE + ARRAY_START + CONTAINER_COUNT + TYPE_UINT8 +
                                       b'\x03' + ARRAY_END + CONTAINER_COUNT + TYPE_UINT8 + b'\x01' + TYPE_BOOL_TRUE +
                                       TYPE_BOOL_FALSE + TYPE_BOOL_TRUE + ARRAY_END),
                         [[], [True], [False, True]])

    def test_array_noop(self):
        # only supported without type
        self.assertEqual(self.bjdloadb(ARRAY_START +
                                       TYPE_NOOP +
                                       TYPE_UINT8 + b'\x01' +
                                       TYPE_NOOP +
                                       TYPE_UINT8 + b'\x02' +
                                       TYPE_NOOP +
                                       ARRAY_END), [1, 2])
        self.assertEqual(self.bjdloadb(ARRAY_START + CONTAINER_COUNT + TYPE_UINT8 + b'\x01' +
                                       TYPE_NOOP +
                                       TYPE_UINT8 + b'\x01'), [1])

    def test_object_invalid(self):
        # negative length
        with self.assertRaises(DecoderException):
            self.bjdloadb(OBJECT_START + CONTAINER_COUNT + self.bjddumpb(-1))

        with self.assertRaises(EncoderException):
            self.bjddumpb({123: 'non-string key'})

        with self.assertRaises(EncoderException):
            self.bjddumpb({'fish': type(list)})

        # invalid key size type
        with self.assertRaises(DecoderException):
            self.bjdloadb(OBJECT_START + TYPE_NULL)

        # invalid key size, key too short, key invalid utf-8, no value
        for suffix in (b'\x81', b'\x01', b'\x01' + b'\xfe', b'\x0101'):
            with self.assertRaises(DecoderException):
                self.bjdloadb(OBJECT_START + TYPE_INT8 + suffix)

        # invalid items() method
        class BadDict(dict):
            def items(self):
                return super(BadDict, self).keys()

        with self.assertRaises(ValueError):
            self.bjddumpb(BadDict({'a': 1, 'b': 2}))

    def test_object(self):
        # custom hook
        with self.assertRaises(TypeError):
            self.bjdloadb(self.bjddumpb({}), object_pairs_hook=int)
        # same as not specifying a custom class
        self.bjdloadb(self.bjddumpb({}), object_pairs_hook=None)

        for hook in (None, OrderedDict):
            check_enc_dec = partial(self.check_enc_dec, object_pairs_hook=hook)

            self.assertEqual(self.bjddumpb({}), OBJECT_START + OBJECT_END)
            self.assertEqual(self.bjddumpb({'a': None}, container_count=True),
                             (OBJECT_START + CONTAINER_COUNT + TYPE_UINT8 + b'\x01' + TYPE_UINT8 + b'\x01' +
                              'a'.encode('utf-8') + TYPE_NULL))
            check_enc_dec({})
            check_enc_dec({'longkey1' * 65: 1})
            check_enc_dec({'longkey2' * 4096: 1})

            obj = {'int': 123,
                   'longint': 9223372036854775807,
                   'float': 1.25,
                   'hp': Decimal('10e15'),
                   'char': 'a',
                   'str': 'here is a string',
                   'unicode': u(r'\u00a9 with extended\u2122'),
                   '': 'empty key',
                   u(r'\u00a9 with extended\u2122'): 'unicode-key',
                   'null': None,
                   'true': True,
                   'false': False,
                   'array': [1, 2, 3],
                   'bytes_array': b'1234',
                   'object': {'another one': 456, 'yet another': {'abc': True}}}
            for opts in ({'container_count': False}, {'container_count': True}):
                check_enc_dec(obj, **opts)

        # dictionary key sorting
        obj1 = OrderedDict.fromkeys('abcdefghijkl')
        obj2 = OrderedDict.fromkeys('abcdefghijkl'[::-1])
        self.assertNotEqual(self.bjddumpb(obj1), self.bjddumpb(obj2))
        self.assertEqual(self.bjddumpb(obj1, sort_keys=True), self.bjddumpb(obj2, sort_keys=True))

        self.assertEqual(self.bjdloadb(self.bjddumpb(obj1), object_pairs_hook=OrderedDict), obj1)

    def test_object_fixed(self):
        raw_start = OBJECT_START + CONTAINER_TYPE + TYPE_INT8 + CONTAINER_COUNT + TYPE_UINT8

        for hook in (None, OrderedDict):
            loadb = partial(self.bjdloadb, object_pairs_hook=hook)

            self.assertEqual(loadb(raw_start + b'\x00'), {})
            self.assertEqual(loadb(raw_start + b'\x03' + (TYPE_UINT8 + b'\x02' + b'aa' + b'\x01' +
                                                          TYPE_UINT8 + b'\x02' + b'bb' + b'\x02' +
                                                          TYPE_UINT8 + b'\x02' + b'cc' + b'\x03')),
                             {'aa': 1, 'bb': 2, 'cc': 3})

            # count only
            self.assertEqual(loadb(OBJECT_START + CONTAINER_COUNT + TYPE_UINT8 + b'\x02' +
                                   TYPE_UINT8 + b'\x02' + b'aa' + TYPE_NULL + TYPE_UINT8 + b'\x02' + b'bb' + TYPE_NULL),
                             {'aa': None, 'bb': None})

            # fixed type + count
            self.assertEqual(loadb(OBJECT_START + CONTAINER_TYPE + TYPE_NULL + CONTAINER_COUNT + TYPE_UINT8 + b'\x02' +
                                   TYPE_UINT8 + b'\x02' + b'aa' + TYPE_UINT8 + b'\x02' + b'bb'),
                             {'aa': None, 'bb': None})

            # fixed type + count (bytes)
            self.assertEqual(loadb(OBJECT_START + CONTAINER_TYPE + TYPE_UINT8 + CONTAINER_COUNT + TYPE_UINT8 + b'\x02' +
                                   TYPE_UINT8 + b'\x02' + b'aa' + b'\x04' + TYPE_UINT8 + b'\x02' + b'bb' + b'\x05'),
                             {'aa': 4, 'bb': 5})

    def test_object_noop(self):
        # only supported without type
        for hook in (None, OrderedDict):
            loadb = partial(self.bjdloadb, object_pairs_hook=hook)
            self.assertEqual(loadb(OBJECT_START +
                                   TYPE_NOOP +
                                   TYPE_UINT8 + b'\x01' + 'a'.encode('utf-8') + TYPE_NULL +
                                   TYPE_NOOP +
                                   TYPE_UINT8 + b'\x01' + 'b'.encode('utf-8') + TYPE_BOOL_TRUE +
                                   OBJECT_END), {'a': None, 'b': True})
            self.assertEqual(loadb(OBJECT_START + CONTAINER_COUNT + TYPE_UINT8 + b'\x01' +
                                   TYPE_NOOP +
                                   TYPE_UINT8 + b'\x01' + 'a'.encode('utf-8') + TYPE_NULL), {'a': None})

    def test_intern_object_keys(self):
        encoded = self.bjddumpb({'asdasd': 1, 'qwdwqd': 2})
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
        mapping = {'a': 1, 'b': 2}
        mapping['c'] = mapping

        for container in (sequence, mapping):
            with self.assertRaises(ValueError):
                self.bjddumpb(container)

        # Refering to the same container multiple times is valid however
        sequence = [1, 2, 3]
        mapping = {'a': 1, 'b': 2}
        self.check_enc_dec([sequence, mapping, sequence, mapping])

    def test_unencodable(self):
        with self.assertRaises(EncoderException):
            self.bjddumpb(type(None))

    def test_decoder_fuzz(self):
        for start, end, fmt in ((0, pow(2, 8), '>B'), (pow(2, 8), pow(2, 16), '>H'), (pow(2, 16), pow(2, 18), '>I')):
            for i in range(start, end):
                try:
                    self.bjdloadb(pack(fmt, i))
                except DecoderException:
                    pass
                except Exception as ex:  # pragma: no cover  pylint: disable=broad-except
                    self.fail('Unexpected failure: %s' % ex)

    def assert_raises_regex(self, *args, **kwargs):
        # pylint: disable=deprecated-method,no-member
        return (self.assertRaisesRegexp if PY2 else self.assertRaisesRegex)(*args, **kwargs)

    def test_recursion(self):
        old_limit = getrecursionlimit()
        setrecursionlimit(200)
        try:
            obj = current = []
            for _ in range(getrecursionlimit() * 2):
                new_list = []
                current.append(new_list)
                current = new_list

            with self.assert_raises_regex(RuntimeError, 'recursion'):
                self.bjddumpb(obj)

            raw = ARRAY_START * (getrecursionlimit() * 2)
            with self.assert_raises_regex(RuntimeError, 'recursion'):
                self.bjdloadb(raw)
        finally:
            setrecursionlimit(old_limit)

    def test_encode_default(self):
        def default(obj):
            if isinstance(obj, set):
                return sorted(obj)
            raise EncoderException('__test__marker__')

        dumpb_default = partial(self.bjddumpb, default=default)
        # Top-level custom type
        obj1 = {1, 2, 3}
        obj2 = default(obj1)
        # Custom type within sequence or mapping
        obj3 = OrderedDict(sorted({'a': 1, 'b': obj1, 'c': [2, obj1]}.items()))
        obj4 = OrderedDict(sorted({'a': 1, 'b': obj2, 'c': [2, obj2]}.items()))

        with self.assert_raises_regex(EncoderException, 'Cannot encode item'):
            self.bjddumpb(obj1)
        # explicit None should behave the same as no default
        with self.assert_raises_regex(EncoderException, 'Cannot encode item'):
            self.bjddumpb(obj1, default=None)

        with self.assert_raises_regex(EncoderException, '__test__marker__'):
            dumpb_default(self)

        self.assertEqual(dumpb_default(obj1), self.bjddumpb(obj2))
        self.assertEqual(dumpb_default(obj3), self.bjddumpb(obj4))

    def test_decode_object_hook(self):
        with self.assertRaises(TypeError):
            self.check_enc_dec({'a': 1, 'b': 2}, object_hook=int)

        def default(obj):
            if isinstance(obj, set):
                return {'__set__': list(obj)}
            raise EncoderException('__test__marker__')

        def object_hook(obj):
            if '__set__' in obj:
                return set(obj['__set__'])
            return obj

        self.check_enc_dec({'a': 1, 'b': {2, 3, 4}}, object_hook=object_hook, default=default)

        class UnHandled(object):
            pass

        with self.assertRaises(EncoderException):
            self.check_enc_dec({'a': 1, 'b': UnHandled()}, object_hook=object_hook, default=default)


@skipUnless(EXTENSION_ENABLED, 'Extension not enabled')
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
            self.bjdloadb(TYPE_STRING + TYPE_INT8 + b'\x01' + b'\xfe' + b'c0fefe' * 4)
        self.assertEqual(ctx.exception.position, 4)

    def test_invalid_fp_dump(self):
        with self.assertRaises(AttributeError):
            self.bjddump(None, 1)

        class Dummy(object):
            write = 1

        class Dummy2(object):
            @staticmethod
            def write(raw):
                raise ValueError('invalid - %s' % repr(raw))

        with self.assertRaises(TypeError):
            self.bjddump(b'', Dummy)

        with self.assertRaises(ValueError):
            self.bjddump(b'', Dummy2)

    def test_invalid_fp_load(self):
        with self.assertRaises(AttributeError):
            self.bjdload(1)

        class Dummy(object):
            read = 1

        class Dummy2(object):

            @staticmethod
            def read(length):
                raise ValueError('invalid - %d' % length)

        with self.assertRaises(TypeError):
            self.bjdload(Dummy)

        with self.assertRaises(ValueError):
            self.bjdload(Dummy2)

    def test_fp(self):
        obj = {'a': 123, 'b': 456}
        output = BytesIO()
        self.bjddump(obj, output)
        output.seek(0)
        self.assertEqual(self.bjdload(output), obj)


@skipUnless(EXTENSION_ENABLED, 'Extension not enabled')
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
        obj2 = ['fishy' * 64] * 10
        output.seek(0)
        self.bjddump(obj2, output)
        output.seek(0)
        self.assertEqual(self.bjdload(output), obj2)

        # larger than extension read buffer (extension only)
        obj3 = ['fishy' * 512] * 10
        output.seek(0)
        self.bjddump(obj3, output)
        output.seek(0)
        self.assertEqual(self.bjdload(output), obj3)

    # Multiple documents in same stream (issue #9)
    def test_fp_multi(self):
        obj = {'a': 123, 'b': b'some raw content'}
        output = BytesIO()
        count = 10

        # Seekable an non-seekable runs
        for _ in range(2):
            output.seek(0)

            for i in range(count):
                obj['c'] = i
                self.bjddump(obj, output)

            output.seek(0)
            for i in range(count):
                obj['c'] = i
                self.assertEqual(self.bjdload(output), obj)

            output.seekable = lambda: False

    # Whole "token" in decoder input unavailable (in non-seekable file-like object)
    def test_fp_callable_incomplete(self):
        obj = [123, b'something']
        # remove whole of last token (binary data 'something', without its length)
        output = BytesIO(self.bjddumpb(obj)[:-(len(obj[1]) + 1)])
        output.seekable = lambda: False

        with self.assert_raises_regex(DecoderException, 'Insufficient input'):
            self.bjdload(output)

    def test_fp_seek_invalid(self):
        output = BytesIO()
        self.bjddump({'a': 333, 'b': 444}, output)
        # pad with data (non-bjdata) to ensure buffering too much data
        output.write(b' ' * 16)
        output.seek(0)

        output.seek_org = output.seek

        # seek fails
        def bad_seek(*_):
            raise OSError('bad seek')

        output.seek = bad_seek
        with self.assert_raises_regex(OSError, 'bad seek'):
            self.bjdload(output)

        # decoding (lack of input) and seek fail - should get decoding failure
        output.seek_org(0, SEEK_END)
        with self.assert_raises_regex(DecoderException, 'Insufficient input'):
            self.bjdload(output)

        # seek is not callable
        output.seek_org(0)
        output.seek = True
        with self.assert_raises_regex(TypeError, 'not callable'):
            self.bjdload(output)

        # decoding (lack of input) and seek not callable - should get decoding failure
        output.seek_org(0, SEEK_END)
        with self.assert_raises_regex(DecoderException, 'Insufficient input'):
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
