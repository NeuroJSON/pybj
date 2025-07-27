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


"""Converts between json & bjdata"""

from __future__ import print_function
from sys import argv, stderr, stdout, stdin, exit  # pylint: disable=redefined-builtin
from json import load as jload, dump as jdump
from collections import OrderedDict

from .compat import STDIN_RAW, STDOUT_RAW
from . import dump as bjdump, load as bjload, EncoderException, DecoderException


def __error(*args, **kwargs):
    print(*args, file=stderr, **kwargs)


def from_json(in_stream, out_stream):
    try:
        obj = jload(in_stream, object_pairs_hook=OrderedDict)
    except ValueError as ex:
        __error("Failed to decode json: %s" % ex)
        return 8
    try:
        bjdump(obj, out_stream, sort_keys=False)
    except EncoderException as ex:
        __error("Failed to encode to bjdata: %s" % ex)
        return 16
    return 0


def to_json(in_stream, out_stream):
    try:
        obj = bjload(in_stream, intern_object_keys=True, object_pairs_hook=OrderedDict)
    except DecoderException as ex:
        __error("Failed to decode bjdata: %s" % ex)
        return 8
    try:
        jdump(obj, out_stream, sort_keys=False, separators=(",", ":"))
    except TypeError as ex:
        __error("Failed to encode to json: %s" % ex)
        return 16
    return 0


__ACTION = frozenset(("fromjson", "tojson"))


def main():
    if not (3 <= len(argv) <= 4 and argv[1] in __ACTION):
        print(
            """USAGE: bjdata (fromjson|tojson) (INFILE|-) [OUTFILE]

Converts an objects between json and bjdata formats. Input is read from INFILE
unless set to '-', in which case stdin is used. If OUTFILE is not
specified, output goes to stdout.""",
            file=stderr,
        )
        return 1

    do_from_json = argv[1] == "fromjson"
    in_file = out_file = None
    try:
        # input
        if argv[2] == "-":
            in_stream = stdin if do_from_json else STDIN_RAW
        else:
            try:
                in_stream = in_file = open(argv[2], "r" if do_from_json else "rb")
            except IOError as ex:
                __error("Failed to open input file for reading: %s" % ex)
                return 2
        # output
        if len(argv) == 3:
            out_stream = STDOUT_RAW if do_from_json else stdout
        else:
            try:
                out_stream = out_file = open(argv[3], "wb" if do_from_json else "w")
            except IOError as ex:
                __error("Failed to open output file for writing: %s" % ex)
                return 4

        return (from_json if do_from_json else to_json)(in_stream, out_stream)
    except IOError as ex:
        __error("I/O failure: %s" % ex)
    finally:
        if in_file:
            in_file.close()
        if out_file:
            out_file.close()


if __name__ == "__main__":
    exit(main())
