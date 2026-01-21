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


"""BJData (Draft 4) and UBJSON (Draft 12) implementation

Example usage:

# To encode
encoded = bjdata.dumpb({'a':123,'b':12.3,'c':[1,2,3,[4,5],'test']})

# To decode
decoded = bjdata.loadb(encoded)

To use a file-like object as input/output, use dump() & load() methods instead.
"""

try:
    from _bjdata import dump, dumpb, load, loadb

    EXTENSION_ENABLED = True
except ImportError:  # pragma: no cover
    from .encoder import dump, dumpb
    from .decoder import load, loadb

    EXTENSION_ENABLED = False

from .encoder import EncoderException
from .decoder import DecoderException

__version__ = "0.6.6"

__all__ = (
    "EXTENSION_ENABLED",
    "dump",
    "dumpb",
    "EncoderException",
    "load",
    "loadb",
    "DecoderException",
)
