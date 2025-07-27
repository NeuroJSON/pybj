/*
 * Copyright (c) 2020-2023 Qianqian Fang <q.fang at neu.edu>. All rights reserved.
 * Copyright (c) 2016-2019 Iotic Labs Ltd. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     https://github.com/NeuroJSON/pybj/blob/master/LICENSE
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#if defined (__cplusplus)
extern "C" {
#endif

#include <Python.h>

/******************************************************************************/

typedef struct {
    PyObject *object_hook;
    PyObject *object_pairs_hook;
    // don't convert BYTE arrays to bytes instances (and keep as an array of individual integers)
    int no_bytes;
    int uint8_bytes;
    int intern_object_keys;
    int islittle;
} _bjdata_decoder_prefs_t;

typedef struct _bjdata_decoder_buffer_t {
    // either supports buffer interface or is callable returning bytes
    PyObject *input;
    // NULL unless input supports seeking in which case expecting callable with signature of io.IOBase.seek()
    PyObject *seek;
    // function used to read data from this buffer with (depending on whether fixed, callable or seekable)
    const char* (*read_func)(struct _bjdata_decoder_buffer_t *buffer, Py_ssize_t *len, char *dst_buffer);
    // buffer protocol access to raw bytes of input
    Py_buffer view;
    // whether view will need to be released
    int view_set;
    // current position in view
    Py_ssize_t pos;
    // total bytes supplied to user (same as pos in case where callable not used)
    Py_ssize_t total_read;
    // temporary destination buffer if required read larger than currently available input
    char *tmp_dst;
    _bjdata_decoder_prefs_t prefs;
} _bjdata_decoder_buffer_t;

/******************************************************************************/

extern _bjdata_decoder_buffer_t* _bjdata_decoder_buffer_create(_bjdata_decoder_prefs_t* prefs,
                                                               PyObject *input, PyObject *seek);
extern int _bjdata_decoder_buffer_free(_bjdata_decoder_buffer_t **buffer);
extern int _bjdata_decoder_init(void);
// note: marker argument only used internally - supply NULL
extern PyObject* _bjdata_decode_value(_bjdata_decoder_buffer_t *buffer, char *given_marker);
extern void _bjdata_decoder_cleanup(void);

#if defined (__cplusplus)
}
#endif
