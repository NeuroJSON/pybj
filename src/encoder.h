/*
 * Copyright (c) 2020-2025 Qianqian Fang <q.fang at neu.edu>. All rights reserved.
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
    PyObject* default_func;
    int container_count;
    int sort_keys;
    int no_float32;
    int islittle;
    int uint8_bytes;
    int soa_format;  /* SOA encoding format for structured arrays */
    double soa_threshold;  // -1=auto, 0=force offset, 0.0-1.0=dict ratio
} _bjdata_encoder_prefs_t;

typedef struct {
    // holds PyBytes instance (buffer)
    PyObject* obj;
    // raw access to obj, size & position
    char* raw;
    size_t len;
    size_t pos;
    // if not NULL, full buffer will be written to this method
    PyObject* fp_write;
    // PySet of sequences and mappings for detecting a circular reference
    PyObject* markers;
    _bjdata_encoder_prefs_t prefs;
} _bjdata_encoder_buffer_t;

/* String field analysis result */
typedef struct {
    int encoding;           /* SOA_STRING_FIXED/DICT/OFFSET */
    Py_ssize_t fixed_len;   /* For FIXED: max UTF-8 byte length */
    PyObject* dict_list;    /* For DICT: list of unique strings */
    Py_ssize_t dict_count;  /* For DICT: number of unique values */
    int index_size;         /* For DICT/OFFSET: 1, 2, or 4 bytes */
    char index_marker;      /* For DICT/OFFSET: TYPE_UINT8/16/32 */
    Py_ssize_t total_len;   /* Total UTF-8 bytes across all strings */
} _string_field_info_t;

/******************************************************************************/

/*
 * Create an encoder buffer.
 *
 * @param prefs     Encoder preferences/options including SOA format
 * @param fp_write  Optional write function (NULL for in-memory buffer)
 * @return          New encoder buffer, or NULL on error (exception set)
 */
extern _bjdata_encoder_buffer_t* _bjdata_encoder_buffer_create(_bjdata_encoder_prefs_t* prefs, PyObject* fp_write);

/*
 * Free an encoder buffer.
 *
 * @param buffer  Pointer to buffer pointer (will be set to NULL)
 */
extern void _bjdata_encoder_buffer_free(_bjdata_encoder_buffer_t** buffer);

/*
 * Finalize encoding and return result.
 *
 * @param buffer  Encoder buffer
 * @return        Bytes object (if no fp_write) or None (if fp_write), NULL on error
 */
extern PyObject* _bjdata_encoder_buffer_finalise(_bjdata_encoder_buffer_t* buffer);

/*
 * Encode a Python object as BJData.
 *
 * This function handles all Python types including:
 * - Primitives: None, bool, int, float, Decimal, str, bytes
 * - Containers: list, tuple, dict
 * - NumPy arrays: scalars, ndarrays, structured arrays
 *
 * When soa_format is enabled (not SOA_FORMAT_NONE) and the object is a
 * numpy structured array (record array), it will be encoded using the
 * BJData Draft 4 SOA format:
 *
 * - SOA_FORMAT_COL: Column-major storage (all values of field1, then field2, etc.)
 *   Written as: {${schema}#count <field1_data><field2_data>...}
 *
 * - SOA_FORMAT_ROW: Row-major storage (interleaved records)
 *   Written as: [${schema}#count <record1><record2>...]
 *
 * @param obj     Python object to encode
 * @param buffer  Encoder buffer
 * @return        0 on success, non-zero on error (exception set)
 */
extern int _bjdata_encode_value(PyObject* obj, _bjdata_encoder_buffer_t* buffer);

/*
 * Initialize the encoder module (import required Python modules).
 *
 * @return  0 on success, 1 on failure (exception set)
 */
extern int _bjdata_encoder_init(void);

/*
 * Clean up the encoder module (release Python references).
 */
extern void _bjdata_encoder_cleanup(void);

#if defined (__cplusplus)
}
#endif