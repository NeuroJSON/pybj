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
    PyObject *default_func;
    int container_count;
    int sort_keys;
    int no_float32;
    int uint8_bytes;
    int islittle;
} _bjdata_encoder_prefs_t;

typedef struct {
    // holds PyBytes instance (buffer)
    PyObject *obj;
    // raw access to obj, size & position
    char* raw;
    size_t len;
    size_t pos;
    // if not NULL, full buffer will be written to this method
    PyObject *fp_write;
    // PySet of sequences and mappings for detecting a circular reference
    PyObject *markers;
    _bjdata_encoder_prefs_t prefs;
} _bjdata_encoder_buffer_t;

/******************************************************************************/

extern _bjdata_encoder_buffer_t* _bjdata_encoder_buffer_create(_bjdata_encoder_prefs_t* prefs, PyObject *fp_write);
extern void _bjdata_encoder_buffer_free(_bjdata_encoder_buffer_t **buffer);
extern PyObject* _bjdata_encoder_buffer_finalise(_bjdata_encoder_buffer_t *buffer);
extern int _bjdata_encode_value(PyObject *obj, _bjdata_encoder_buffer_t *buffer);
extern int _bjdata_encoder_init(void);
extern void _bjdata_encoder_cleanup(void);

#if defined (__cplusplus)
}
#endif
