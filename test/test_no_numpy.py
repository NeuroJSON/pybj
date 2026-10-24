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

"""numpy is optional: bjdata, C extension included, works without it."""

import os
import subprocess
import sys
from unittest import TestCase, skipUnless

from bjdata import EXTENSION_ENABLED

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Runs in a fresh interpreter that cannot import numpy, whether or not it is installed
SCRIPT = r"""
import sys
sys.modules["numpy"] = None
sys.path.insert(0, ROOT)

import bjdata
from bjdata import DecoderException, EncoderException

if EXT:
    assert bjdata.EXTENSION_ENABLED
    from bjdata import dumpb, loadb
else:
    from bjdata.encoder import dumpb
    from bjdata.decoder import loadb

obj = {"a": [1, -2, 3.5, None, True, "x", b"\x00\x01"], "b": {"c": []}}
assert loadb(dumpb(obj)) == obj

# Packed array of int32
assert loadb(b"[$l#i\x03" + b"\x01\x00\x00\x00\x02\x00\x00\x00\xfd\xff\xff\xff") == [1, 2, -3]
# ND-array of uint8, 2x3
assert loadb(b"[$U#[i\x02i\x03]\x01\x02\x03\x04\x05\x06") == [[1, 2, 3], [4, 5, 6]]
# ND-array of int16, 2x1x2
assert loadb(b"[$I#[i\x02i\x01i\x02]\x01\x00\x02\x00\x03\x00\x04\x00") == [[[1, 2]], [[3, 4]]]
# Bytes still decode to bytes
assert loadb(b"[$B#i\x02\x01\x02") == b"\x01\x02"

# Structured (SOA) arrays need numpy
try:
    loadb(b"[${i\x01xU}#i\x01\x05")
except DecoderException as ex:
    assert "numpy" in str(ex), ex
else:
    raise AssertionError("SOA decoded without numpy")

try:
    dumpb(object())
except EncoderException:
    pass
else:
    raise AssertionError("encoded an object()")

assert sys.modules["numpy"] is None
"""


class TestNoNumpy(TestCase):
    @staticmethod
    def run_without_numpy(ext):
        script = "ROOT = %r\nEXT = %r\n%s" % (ROOT, ext, SCRIPT)
        return subprocess.run(
            [sys.executable, "-c", script], capture_output=True, text=True
        )

    def check(self, ext):
        result = self.run_without_numpy(ext)
        self.assertEqual(result.returncode, 0, result.stderr)
        # A numpy import failing inside the extension would print here
        self.assertEqual(result.stderr, "")

    def test_pure(self):
        self.check(False)

    @skipUnless(EXTENSION_ENABLED, "Extension not enabled")
    def test_extension(self):
        self.check(True)
