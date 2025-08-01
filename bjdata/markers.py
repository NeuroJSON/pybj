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


"""BJData and UBJSON marker defitions"""

# Value types
TYPE_NONE = b"\x00"  # Used internally only, not part of bjdata specification
TYPE_NULL = b"Z"
TYPE_NOOP = b"N"
TYPE_BOOL_TRUE = b"T"
TYPE_BOOL_FALSE = b"F"
TYPE_BYTE = b"B"
TYPE_INT8 = b"i"
TYPE_UINT8 = b"U"
TYPE_INT16 = b"I"
TYPE_UINT16 = b"u"
TYPE_INT32 = b"l"
TYPE_UINT32 = b"m"
TYPE_INT64 = b"L"
TYPE_UINT64 = b"M"
TYPE_FLOAT16 = b"h"
TYPE_FLOAT32 = b"d"
TYPE_FLOAT64 = b"D"
TYPE_HIGH_PREC = b"H"
TYPE_CHAR = b"C"
TYPE_STRING = b"S"

# Container delimiters
OBJECT_START = b"{"
OBJECT_END = b"}"
ARRAY_START = b"["
ARRAY_END = b"]"

# Optional container parameters
CONTAINER_TYPE = b"$"
CONTAINER_COUNT = b"#"
