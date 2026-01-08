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
    PyObject* object_hook;
    PyObject* object_pairs_hook;
    // don't convert BYTE arrays to bytes instances (and keep as an array of individual integers)
    int no_bytes;
    int intern_object_keys;
    int islittle;
    int uint8_bytes;
} _bjdata_decoder_prefs_t;

typedef struct _bjdata_decoder_buffer_t {
    // either supports buffer interface or is callable returning bytes
    PyObject* input;
    // NULL unless input supports seeking in which case expecting callable with signature of io.IOBase.seek()
    PyObject* seek;
    // function used to read data from this buffer with (depending on whether fixed, callable or seekable)
    const char* (*read_func)(struct _bjdata_decoder_buffer_t* buffer, Py_ssize_t* len, char* dst_buffer);
    // buffer protocol access to raw bytes of input
    Py_buffer view;
    // whether view will need to be released
    int view_set;
    // current position in view
    Py_ssize_t pos;
    // total bytes supplied to user (same as pos in case where callable not used)
    Py_ssize_t total_read;
    // temporary destination buffer if required read larger than currently available input
    char* tmp_dst;
    _bjdata_decoder_prefs_t prefs;
} _bjdata_decoder_buffer_t;

/******************************************************************************/


/* SOA schema field structure - represents a single field in an SOA schema */
typedef struct _soa_field_t {
    char* name;
    Py_ssize_t name_len;
    char type_marker;
    int dtype_num;               /* -1 for strings, -2 for nested struct */
    Py_ssize_t itemsize;

    /* Sub-array support */
    Py_ssize_t num_elem;         /* Number of elements (>1 for sub-arrays) */

    /* String encoding support */
    int str_encoding;            /* SOA_STRING_FIXED/DICT/OFFSET, or -1 if not string */
    Py_ssize_t str_fixed_len;    /* Fixed string length */
    PyObject* str_dict;          /* Dictionary list for DICT encoding */
    Py_ssize_t str_dict_count;   /* Number of dictionary entries */
    int str_index_size;          /* Index size in bytes */

    /* Nested struct support (when dtype_num == -2) */
    struct _soa_field_t* nested_fields;  /* Array of nested field descriptors */
    Py_ssize_t nested_count;             /* Number of nested fields */
} _soa_field_t;

/* SOA schema structure */
typedef struct {
    _soa_field_t* fields;
    Py_ssize_t num_fields;
} _soa_schema_t;

/******************************************************************************/

/*
 * Create a decoder buffer from input.
 *
 * @param prefs  Decoder preferences/options
 * @param input  Input source (must support buffer interface or be callable)
 * @param seek   Optional seek function for buffered reading (can be NULL)
 * @return       New decoder buffer, or NULL on error (exception set)
 */
extern _bjdata_decoder_buffer_t* _bjdata_decoder_buffer_create(_bjdata_decoder_prefs_t* prefs,
        PyObject* input, PyObject* seek);

/*
 * Free a decoder buffer and clean up resources.
 *
 * @param buffer  Pointer to buffer pointer (will be set to NULL)
 * @return        Non-zero if cleanup failed and no other exception was set
 */
extern int _bjdata_decoder_buffer_free(_bjdata_decoder_buffer_t** buffer);

/*
 * Initialize the decoder module (import required Python modules).
 *
 * @return  0 on success, 1 on failure (exception set)
 */
extern int _bjdata_decoder_init(void);

/*
 * Decode a single BJData value from the buffer.
 *
 * This function handles all BJData types including:
 * - Primitives: null, bool, integers, floats, char, string, high-precision
 * - Containers: arrays and objects
 * - Optimized formats: typed/counted arrays, ND-arrays
 * - SOA (Structure of Arrays) format for structured data
 *
 * SOA format is automatically detected when the container type marker ($)
 * is followed by an object start marker ({). The schema is decoded from
 * the object, and the payload is read as a numpy structured array.
 *
 * For arrays ([), SOA data is read in row-major (AOS/interleaved) order.
 * For objects ({), SOA data is read in column-major (true SOA) order.
 *
 * @param buffer        Decoder buffer to read from
 * @param given_marker  Optional pre-read type marker (NULL to read from buffer)
 * @return              Decoded Python object, or NULL on error (exception set)
 */
extern PyObject* _bjdata_decode_value(_bjdata_decoder_buffer_t* buffer, char* given_marker);

/*
 * Clean up the decoder module (release Python references).
 */
extern void _bjdata_decoder_cleanup(void);

#if defined (__cplusplus)
}
#endif