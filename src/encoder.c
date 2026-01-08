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

#include <Python.h>
#include <bytesobject.h>
#include <string.h>

#define NO_IMPORT_ARRAY

#include "numpyapi.h"
#include "common.h"
#include "markers.h"
#include "encoder.h"
#include "python_funcs.h"

/******************************************************************************/

static char bytes_array_prefix[] = {ARRAY_START, CONTAINER_TYPE, TYPE_BYTE, CONTAINER_COUNT};

#define POWER_TWO(x) ((long long) 1 << (x))

#if defined(_MSC_VER) && !defined(fpclassify)
    #define USE__FPCLASS
#endif

// initial encoder buffer size (when not supplied with fp)
#define BUFFER_INITIAL_SIZE 64
// encoder buffer size when using fp (i.e. minimum number of bytes to buffer before writing out)
#define BUFFER_FP_SIZE 256

static PyObject* EncoderException = NULL;
static PyTypeObject* PyDec_Type = NULL;
#define PyDec_Check(v) PyObject_TypeCheck(v, PyDec_Type)

/******************************************************************************/

static int _encoder_buffer_write(_bjdata_encoder_buffer_t* buffer, const char* const chunk, size_t chunk_len);

#define RECURSE_AND_BAIL_ON_NONZERO(action, recurse_msg) {\
        int ret;\
        BAIL_ON_NONZERO(Py_EnterRecursiveCall(recurse_msg));\
        ret = (action);\
        Py_LeaveRecursiveCall();\
        BAIL_ON_NONZERO(ret);\
    }

#define WRITE_OR_BAIL(str, len) BAIL_ON_NONZERO(_encoder_buffer_write(buffer, (str), len))
#define WRITE_CHAR_OR_BAIL(c) {\
        char ctmp = (c);\
        WRITE_OR_BAIL(&ctmp, 1);\
    }

/* These functions return non-zero on failure (an exception will have been set). Note that no type checking is performed
 * where a Python type is mentioned in the function name!
 */
static int _encode_PyBytes(PyObject* obj, _bjdata_encoder_buffer_t* buffer);
static int _encode_PyObject_as_PyDecimal(PyObject* obj, _bjdata_encoder_buffer_t* buffer);
static int _encode_PyDecimal(PyObject* obj, _bjdata_encoder_buffer_t* buffer);
static int _encode_PyUnicode(PyObject* obj, _bjdata_encoder_buffer_t* buffer);
static int _encode_PyFloat(PyObject* obj, _bjdata_encoder_buffer_t* buffer);
static int _encode_PyLong(PyObject* obj, _bjdata_encoder_buffer_t* buffer);
static int _encode_longlong(long long num, _bjdata_encoder_buffer_t* buffer);
#if PY_MAJOR_VERSION < 3
    static int _encode_PyInt(PyObject* obj, _bjdata_encoder_buffer_t* buffer);
#endif
static int _encode_PySequence(PyObject* obj, _bjdata_encoder_buffer_t* buffer);
static int _encode_mapping_key(PyObject* obj, _bjdata_encoder_buffer_t* buffer);
static int _encode_PyMapping(PyObject* obj, _bjdata_encoder_buffer_t* buffer);
static int _encode_NDarray(PyObject* obj, _bjdata_encoder_buffer_t* buffer);
static int _encode_soa(PyArrayObject* arr, _bjdata_encoder_buffer_t* buffer, int is_row_major);

/* Helper functions for SOA encoding */
static int _write_field_schema_recursive(PyArray_Descr* fd, _bjdata_encoder_buffer_t* buffer);
static int _write_index_value(Py_ssize_t idx, int size, _bjdata_encoder_buffer_t* buffer);
static int _analyze_string_field(PyArrayObject* flat, PyObject* field_name,
                                 Py_ssize_t field_index, npy_intp count,
                                 double threshold, _string_field_info_t* info);

/* Unified lookup table for numpy type to BJData marker */
static const int numpytypes[][2] = {
    {NPY_BOOL,       TYPE_UINT8},
    {NPY_BYTE,       TYPE_INT8},
    {NPY_INT8,       TYPE_INT8},
    {NPY_SHORT,      TYPE_INT16},
    {NPY_INT16,      TYPE_INT16},
    {NPY_INT,        TYPE_INT32},
    {NPY_INT32,      TYPE_INT32},
    {NPY_LONGLONG,   TYPE_INT64},
    {NPY_INT64,      TYPE_INT64},
    {NPY_UINT8,      TYPE_UINT8},
    {NPY_UBYTE,      TYPE_UINT8},
    {NPY_USHORT,     TYPE_UINT16},
    {NPY_UINT16,     TYPE_UINT16},
    {NPY_UINT,       TYPE_UINT32},
    {NPY_UINT32,     TYPE_UINT32},
    {NPY_ULONGLONG,  TYPE_UINT64},
    {NPY_UINT64,     TYPE_UINT64},
    {NPY_HALF,       TYPE_FLOAT16},
    {NPY_FLOAT16,    TYPE_FLOAT16},
    {NPY_FLOAT,      TYPE_FLOAT32},
    {NPY_FLOAT32,    TYPE_FLOAT32},
    {NPY_DOUBLE,     TYPE_FLOAT64},
    {NPY_FLOAT64,    TYPE_FLOAT64},
    {NPY_CFLOAT,     TYPE_FLOAT32},
    {NPY_COMPLEX64,  TYPE_FLOAT32},
    {NPY_CDOUBLE,    TYPE_FLOAT64},
    {NPY_COMPLEX128, TYPE_FLOAT64},
    {NPY_STRING,     TYPE_STRING},
    {NPY_UNICODE,    TYPE_STRING}
};

/******************************************************************************/

/* fp_write, if not NULL, must be a callable which accepts a single bytes argument. On failure will set exception.
 * Currently only increases reference count for fp_write parameter.
 */
_bjdata_encoder_buffer_t* _bjdata_encoder_buffer_create(_bjdata_encoder_prefs_t* prefs, PyObject* fp_write) {
    _bjdata_encoder_buffer_t* buffer;

    if (NULL == (buffer = calloc(1, sizeof(_bjdata_encoder_buffer_t)))) {
        PyErr_NoMemory();
        return NULL;
    }

    buffer->len = (NULL != fp_write) ? BUFFER_FP_SIZE : BUFFER_INITIAL_SIZE;
    BAIL_ON_NULL(buffer->obj = PyBytes_FromStringAndSize(NULL, buffer->len));
    buffer->raw = PyBytes_AS_STRING(buffer->obj);
    buffer->pos = 0;

    BAIL_ON_NULL(buffer->markers = PySet_New(NULL));

    buffer->prefs = *prefs;
    buffer->fp_write = fp_write;
    Py_XINCREF(fp_write);

    // treat Py_None as no default_func being supplied
    if (Py_None == buffer->prefs.default_func) {
        buffer->prefs.default_func = NULL;
    }

    return buffer;

bail:
    _bjdata_encoder_buffer_free(&buffer);
    return NULL;
}

void _bjdata_encoder_buffer_free(_bjdata_encoder_buffer_t** buffer) {
    if (NULL != buffer && NULL != *buffer) {
        Py_XDECREF((*buffer)->obj);
        Py_XDECREF((*buffer)->fp_write);
        Py_XDECREF((*buffer)->markers);
        free(*buffer);
        *buffer = NULL;
    }
}

// Note: Sets python exception on failure and returns non-zero
static int _encoder_buffer_write(_bjdata_encoder_buffer_t* buffer, const char* const chunk, size_t chunk_len) {
    size_t new_len;
    PyObject* fp_write_ret;

    if (0 == chunk_len) {
        return 0;
    }

    // no write method, use buffer only
    if (NULL == buffer->fp_write) {
        // increase buffer size if too small
        if (chunk_len > (buffer->len - buffer->pos)) {
            for (new_len = buffer->len; new_len < (buffer->pos + chunk_len); new_len *= 2);

            BAIL_ON_NONZERO(_PyBytes_Resize(&buffer->obj, new_len));
            buffer->raw = PyBytes_AS_STRING(buffer->obj);
            buffer->len = new_len;
        }

        memcpy(&(buffer->raw[buffer->pos]), chunk, sizeof(char) * chunk_len);
        buffer->pos += chunk_len;

    } else {
        // increase buffer to fit all first
        if (chunk_len > (buffer->len - buffer->pos)) {
            BAIL_ON_NONZERO(_PyBytes_Resize(&buffer->obj, (buffer->pos + chunk_len)));
            buffer->raw = PyBytes_AS_STRING(buffer->obj);
            buffer->len = buffer->pos + chunk_len;
        }

        memcpy(&(buffer->raw[buffer->pos]), chunk, sizeof(char) * chunk_len);
        buffer->pos += chunk_len;

        // flush buffer to write method
        if (buffer->pos >= buffer->len) {
            BAIL_ON_NULL(fp_write_ret = PyObject_CallFunctionObjArgs(buffer->fp_write, buffer->obj, NULL));
            Py_DECREF(fp_write_ret);
            Py_DECREF(buffer->obj);
            buffer->len = BUFFER_FP_SIZE;
            BAIL_ON_NULL(buffer->obj = PyBytes_FromStringAndSize(NULL, buffer->len));
            buffer->raw = PyBytes_AS_STRING(buffer->obj);
            buffer->pos = 0;
        }
    }

    return 0;

bail:
    return 1;
}

// Flushes remaining bytes to writer and returns None or returns final bytes object (when no writer specified).
// Does NOT free passed in buffer struct.
PyObject* _bjdata_encoder_buffer_finalise(_bjdata_encoder_buffer_t* buffer) {
    PyObject* fp_write_ret;

    // shrink buffer to fit
    if (buffer->pos < buffer->len) {
        BAIL_ON_NONZERO(_PyBytes_Resize(&buffer->obj, buffer->pos));
        buffer->len = buffer->pos;
    }

    if (NULL == buffer->fp_write) {
        Py_INCREF(buffer->obj);
        return buffer->obj;
    } else {
        if (buffer->pos > 0) {
            BAIL_ON_NULL(fp_write_ret = PyObject_CallFunctionObjArgs(buffer->fp_write, buffer->obj, NULL));
            Py_DECREF(fp_write_ret);
        }

        Py_RETURN_NONE;
    }

bail:
    return NULL;
}

/******************************************************************************/

static int _encode_PyBytes(PyObject* obj, _bjdata_encoder_buffer_t* buffer) {
    const char* raw;
    Py_ssize_t len;

    raw = PyBytes_AS_STRING(obj);
    len = PyBytes_GET_SIZE(obj);

    WRITE_OR_BAIL(bytes_array_prefix, sizeof(bytes_array_prefix));
    BAIL_ON_NONZERO(_encode_longlong(len, buffer));
    WRITE_OR_BAIL(raw, len);
    // no ARRAY_END since length was specified

    return 0;

bail:
    return 1;
}

static int _encode_PyByteArray(PyObject* obj, _bjdata_encoder_buffer_t* buffer) {
    const char* raw;
    Py_ssize_t len;

    raw = PyByteArray_AS_STRING(obj);
    len = PyByteArray_GET_SIZE(obj);

    WRITE_OR_BAIL(bytes_array_prefix, sizeof(bytes_array_prefix));
    BAIL_ON_NONZERO(_encode_longlong(len, buffer));
    WRITE_OR_BAIL(raw, len);
    // no ARRAY_END since length was specified

    return 0;

bail:
    return 1;
}

/******************************************************************************/

/* Unified marker lookup - used by both regular arrays and SOA */
static int _lookup_marker(npy_intp numpytypeid) {
    int i, len = (sizeof(numpytypes) >> 3);

    for (i = 0; i < len; i++) {
        if (numpytypeid == (npy_intp)numpytypes[i][0]) {
            return numpytypes[i][1];
        }
    }

    return -1;
}

/* Get SOA-specific type marker (excludes string types, includes bool) */
static int _get_soa_type_marker(int dtype_num) {
    /* Boolean needs special handling in SOA */
    if (dtype_num == NPY_BOOL) {
        return TYPE_BOOL_TRUE;
    }

    /* For other types, use the unified lookup */
    int marker = _lookup_marker(dtype_num);

    /* Filter out string types - they need special handling in SOA */
    if (marker == TYPE_STRING) {
        return -1;
    }

    return marker;
}

/* Check if a dtype is string/unicode */
static inline int _is_string_type(int type_num) {
    return (type_num == NPY_UNICODE || type_num == NPY_STRING);
}

/* Recursively check if a dtype is supported for SOA encoding */
static int _is_soa_compatible_dtype(PyArray_Descr* fd) {
    int type_num = DESCR_TYPE_NUM(fd);

    /* String types are supported */
    if (_is_string_type(type_num)) {
        return 1;
    }

    /* Numeric types (including bool) */
    if (_get_soa_type_marker(type_num) >= 0) {
        return 1;
    }

    if (type_num == NPY_VOID) {
        /* Check for sub-array first */
        PyObject* subdtype = PyObject_GetAttrString((PyObject*)fd, "subdtype");

        if (subdtype && subdtype != Py_None && PyTuple_Check(subdtype) && PyTuple_GET_SIZE(subdtype) >= 2) {
            PyArray_Descr* base = (PyArray_Descr*)PyTuple_GET_ITEM(subdtype, 0);
            int result = _is_soa_compatible_dtype(base);
            Py_DECREF(subdtype);
            return result;
        }

        Py_XDECREF(subdtype);
        PyErr_Clear();

        /* Check for nested struct */
        PyObject* names = PyObject_GetAttrString((PyObject*)fd, "names");

        if (names && names != Py_None && PyTuple_Check(names) && PyTuple_GET_SIZE(names) > 0) {
            PyObject* fields = PyObject_GetAttrString((PyObject*)fd, "fields");

            if (!fields || !PyMapping_Check(fields)) {
                Py_DECREF(names);
                Py_XDECREF(fields);
                PyErr_Clear();
                return 0;
            }

            int result = 1;
            Py_ssize_t num_names = PyTuple_GET_SIZE(names);

            for (Py_ssize_t i = 0; i < num_names && result; i++) {
                PyObject* fname = PyTuple_GET_ITEM(names, i);
                PyObject* info = PyObject_GetItem(fields, fname);

                if (!info || !PyTuple_Check(info) || PyTuple_GET_SIZE(info) < 1) {
                    Py_XDECREF(info);
                    result = 0;
                    break;
                }

                PyArray_Descr* nested_fd = (PyArray_Descr*)PyTuple_GET_ITEM(info, 0);
                result = _is_soa_compatible_dtype(nested_fd);
                Py_DECREF(info);
            }

            Py_DECREF(fields);
            Py_DECREF(names);
            return result;
        }

        Py_XDECREF(names);
        PyErr_Clear();
    }

    return 0;
}

/* Check if numpy array is a structured array suitable for SOA encoding */
static int _can_encode_as_soa(PyArrayObject* arr) {
    PyArray_Descr* dtype = PyArray_DESCR(arr);
    PyObject* names = PyObject_GetAttrString((PyObject*)dtype, "names");

    if (!names || names == Py_None || !PyTuple_Check(names)) {
        Py_XDECREF(names);
        PyErr_Clear();
        return 0;
    }

    Py_ssize_t num_names = PyTuple_GET_SIZE(names);

    if (num_names == 0) {
        Py_DECREF(names);
        return 0;
    }

    PyObject* fields = PyObject_GetAttrString((PyObject*)dtype, "fields");

    if (!fields || !PyMapping_Check(fields)) {
        Py_DECREF(names);
        Py_XDECREF(fields);
        PyErr_Clear();
        return 0;
    }

    int result = 1;

    for (Py_ssize_t i = 0; i < num_names && result; i++) {
        PyObject* fname = PyTuple_GET_ITEM(names, i);
        PyObject* info = PyObject_GetItem(fields, fname);

        if (!info || !PyTuple_Check(info) || PyTuple_GET_SIZE(info) < 1) {
            Py_XDECREF(info);
            result = 0;
            break;
        }

        PyArray_Descr* fd = (PyArray_Descr*)PyTuple_GET_ITEM(info, 0);
        result = _is_soa_compatible_dtype(fd);
        Py_DECREF(info);
    }

    Py_DECREF(fields);
    Py_DECREF(names);
    return result;
}

/* Helper: encode a field name (UTF-8 string with length prefix) */
static int _encode_field_name(PyObject* name, _bjdata_encoder_buffer_t* buffer) {
    PyObject* encoded = PyUnicode_AsEncodedString(name, "utf-8", NULL);

    if (!encoded) {
        return 1;
    }

    int ret = _encode_longlong(PyBytes_GET_SIZE(encoded), buffer);

    if (ret == 0) {
        WRITE_OR_BAIL(PyBytes_AS_STRING(encoded), PyBytes_GET_SIZE(encoded));
    }

    Py_DECREF(encoded);
    return ret;

bail:
    Py_DECREF(encoded);
    return 1;
}

/* Write schema for a field recursively (handles nested structs) */
static int _write_field_schema_recursive(PyArray_Descr* fd, _bjdata_encoder_buffer_t* buffer) {
    int type_num = DESCR_TYPE_NUM(fd);
    Py_ssize_t i;

    /* Handle NPY_VOID: could be sub-array or nested struct */
    if (type_num == NPY_VOID) {
        /* Check for sub-array first */
        PyObject* subdtype = PyObject_GetAttrString((PyObject*)fd, "subdtype");

        if (subdtype && subdtype != Py_None && PyTuple_Check(subdtype) && PyTuple_GET_SIZE(subdtype) >= 2) {
            PyArray_Descr* base_dtype = (PyArray_Descr*)PyTuple_GET_ITEM(subdtype, 0);
            PyObject* shape = PyTuple_GET_ITEM(subdtype, 1);

            Py_ssize_t num_elem = 1;

            if (PyTuple_Check(shape)) {
                for (i = 0; i < PyTuple_GET_SIZE(shape); i++) {
                    num_elem *= PyLong_AsLong(PyTuple_GET_ITEM(shape, i));
                }
            }

            int base_type = DESCR_TYPE_NUM(base_dtype);
            Py_DECREF(subdtype);

            /* Write sub-array schema: [TTT...] */
            WRITE_CHAR_OR_BAIL(ARRAY_START);

            if (base_type == NPY_BOOL) {
                for (i = 0; i < num_elem; i++) {
                    WRITE_CHAR_OR_BAIL(TYPE_BOOL_TRUE);
                }
            } else if (_is_string_type(base_type)) {
                for (i = 0; i < num_elem; i++) {
                    WRITE_CHAR_OR_BAIL(TYPE_CHAR);
                }
            } else {
                int marker = _get_soa_type_marker(base_type);

                if (marker < 0) {
                    PyErr_Format(PyExc_ValueError, "Unsupported sub-array element type: %d", base_type);
                    goto bail;
                }

                for (i = 0; i < num_elem; i++) {
                    WRITE_CHAR_OR_BAIL((char)marker);
                }
            }

            WRITE_CHAR_OR_BAIL(ARRAY_END);
            return 0;
        }

        Py_XDECREF(subdtype);
        PyErr_Clear();

        /* Check for nested struct */
        PyObject* names = PyObject_GetAttrString((PyObject*)fd, "names");

        if (names && names != Py_None && PyTuple_Check(names) && PyTuple_GET_SIZE(names) > 0) {
            PyObject* fields = PyObject_GetAttrString((PyObject*)fd, "fields");

            if (fields && PyMapping_Check(fields)) {
                WRITE_CHAR_OR_BAIL(OBJECT_START);

                Py_ssize_t nf = PyTuple_GET_SIZE(names);

                for (i = 0; i < nf; i++) {
                    PyObject* nm = PyTuple_GET_ITEM(names, i);

                    BAIL_ON_NONZERO(_encode_field_name(nm, buffer));

                    PyObject* info = PyObject_GetItem(fields, nm);

                    if (!info) {
                        Py_DECREF(fields);
                        Py_DECREF(names);
                        goto bail;
                    }

                    PyArray_Descr* nested_fd = (PyArray_Descr*)PyTuple_GET_ITEM(info, 0);
                    int ret = _write_field_schema_recursive(nested_fd, buffer);
                    Py_DECREF(info);

                    if (ret != 0) {
                        Py_DECREF(fields);
                        Py_DECREF(names);
                        goto bail;
                    }
                }

                WRITE_CHAR_OR_BAIL(OBJECT_END);
                Py_DECREF(fields);
                Py_DECREF(names);
                return 0;
            }

            Py_XDECREF(fields);
        }

        Py_XDECREF(names);
        PyErr_Clear();

        PyErr_SetString(PyExc_ValueError, "Unsupported void type in SOA schema");
        goto bail;
    }

    /* String types - write NUMPY BYTE SIZE (not character count) */
    if (_is_string_type(type_num)) {
        WRITE_CHAR_OR_BAIL(TYPE_STRING);
        BAIL_ON_NONZERO(_encode_longlong(DESCR_ELSIZE(fd), buffer));
        return 0;
    }

    /* Boolean */
    if (type_num == NPY_BOOL) {
        WRITE_CHAR_OR_BAIL(TYPE_BOOL_TRUE);
        return 0;
    }

    /* Numeric types */
    int marker = _get_soa_type_marker(type_num);

    if (marker < 0) {
        PyErr_Format(PyExc_ValueError, "Unsupported SOA field type: %d", type_num);
        goto bail;
    }

    WRITE_CHAR_OR_BAIL((char)marker);
    return 0;

bail:
    return 1;
}

/* Helper to write index value with specified byte size */
static int _write_index_value(Py_ssize_t idx, int size, _bjdata_encoder_buffer_t* buffer) {
    char buf[8];
    int le = buffer->prefs.islittle;

    if (size == 1) {
        buf[0] = (char)idx;
        WRITE_OR_BAIL(buf, 1);
    } else if (size == 2) {
        buf[le ? 0 : 1] = idx & 0xFF;
        buf[le ? 1 : 0] = (idx >> 8) & 0xFF;
        WRITE_OR_BAIL(buf, 2);
    } else {
        for (int i = 0; i < 4; i++) {
            buf[le ? i : 3 - i] = (idx >> (8 * i)) & 0xFF;
        }

        WRITE_OR_BAIL(buf, 4);
    }

    return 0;

bail:
    return 1;
}

/* Determine the smallest index type for a given count */
static inline void _get_index_type(Py_ssize_t count, int* size, char* marker) {
    if (count <= 255) {
        *size = 1;
        *marker = TYPE_UINT8;
    } else if (count <= 65535) {
        *size = 2;
        *marker = TYPE_UINT16;
    } else {
        *size = 4;
        *marker = TYPE_UINT32;
    }
}

/* Analyze string field to determine best encoding */
static int _analyze_string_field(PyArrayObject* flat, PyObject* field_name,
                                 Py_ssize_t field_index, npy_intp count,
                                 double threshold, _string_field_info_t* info) {
    PyObject* unique_set = NULL;
    PyObject* unique_list = NULL;
    Py_ssize_t max_len = 0;
    Py_ssize_t total_len = 0;
    npy_intp j;

    /* Initialize */
    memset(info, 0, sizeof(_string_field_info_t));
    info->encoding = SOA_STRING_FIXED;
    info->fixed_len = 1;

    if (count == 0) {
        return 0;
    }

    BAIL_ON_NULL(unique_set = PySet_New(NULL));

    /* First pass: collect unique values and compute lengths */
    for (j = 0; j < count; j++) {
        PyObject* rec = PyArray_GETITEM(flat, PyArray_GETPTR1(flat, j));

        if (!rec) {
            goto bail;
        }

        PyObject* val = NULL;

        if (PyTuple_Check(rec)) {
            val = PyTuple_GET_ITEM(rec, field_index);
            Py_INCREF(val);
        } else {
            val = PyObject_GetItem(rec, field_name);
        }

        Py_DECREF(rec);

        if (!val) {
            goto bail;
        }

        /* Get UTF-8 length */
        PyObject* utf8 = PyUnicode_AsEncodedString(val, "utf-8", NULL);

        if (!utf8) {
            Py_DECREF(val);
            goto bail;
        }

        Py_ssize_t len = PyBytes_GET_SIZE(utf8);
        max_len = (len > max_len) ? len : max_len;
        total_len += len;
        Py_DECREF(utf8);

        PySet_Add(unique_set, val);
        Py_DECREF(val);
    }

    Py_ssize_t num_unique = PySet_GET_SIZE(unique_set);
    info->total_len = total_len;
    info->fixed_len = max_len > 0 ? max_len : 1;

    /* Force offset if threshold is exactly 0 */
    if (threshold == 0.0) {
        info->encoding = SOA_STRING_OFFSET;
        _get_index_type(total_len, &info->index_size, &info->index_marker);
        Py_DECREF(unique_set);
        return 0;
    }

    /* Calculate costs */
    double thresh = (threshold > 0) ? threshold : 0.3;
    Py_ssize_t fixed_cost = max_len * count;

    /* Dict cost: indices + dictionary overhead */
    int idx_size;
    char idx_marker;
    _get_index_type(num_unique, &idx_size, &idx_marker);

    /* Calculate dict strings total length */
    Py_ssize_t dict_strings_total = 0;
    unique_list = PySequence_List(unique_set);

    if (unique_list) {
        for (Py_ssize_t i = 0; i < num_unique; i++) {
            PyObject* item = PyList_GET_ITEM(unique_list, i);
            PyObject* enc = PyUnicode_AsEncodedString(item, "utf-8", NULL);

            if (enc) {
                dict_strings_total += PyBytes_GET_SIZE(enc);
                Py_DECREF(enc);
            }
        }
    }

    Py_ssize_t dict_cost = idx_size * count + dict_strings_total + num_unique * 2;

    /* Offset cost */
    int off_size;
    char off_marker;
    _get_index_type(total_len, &off_size, &off_marker);
    Py_ssize_t offset_cost = idx_size * count + (count + 1) * off_size + total_len;

    /* Choose encoding */
    if (num_unique <= (Py_ssize_t)(count * thresh) && dict_cost < fixed_cost && dict_cost < offset_cost) {
        info->encoding = SOA_STRING_DICT;
        info->dict_list = unique_list;
        unique_list = NULL;
        info->dict_count = num_unique;
        info->index_size = idx_size;
        info->index_marker = idx_marker;
    } else if (max_len > 32 && offset_cost < fixed_cost) {
        info->encoding = SOA_STRING_OFFSET;
        info->index_size = off_size;
        info->index_marker = off_marker;
    } else {
        info->encoding = SOA_STRING_FIXED;
    }

    Py_XDECREF(unique_list);
    Py_DECREF(unique_set);
    return 0;

bail:
    Py_XDECREF(unique_list);
    Py_XDECREF(unique_set);
    return -1;
}

/* Write string schema based on encoding type */
static int _write_string_schema(_string_field_info_t* info, _bjdata_encoder_buffer_t* buffer) {
    Py_ssize_t i;

    if (info->encoding == SOA_STRING_FIXED) {
        WRITE_CHAR_OR_BAIL(TYPE_STRING);
        BAIL_ON_NONZERO(_encode_longlong(info->fixed_len, buffer));

    } else if (info->encoding == SOA_STRING_DICT) {
        char hdr[] = {ARRAY_START, CONTAINER_TYPE, TYPE_STRING, CONTAINER_COUNT};
        WRITE_OR_BAIL(hdr, 4);
        BAIL_ON_NONZERO(_encode_longlong(info->dict_count, buffer));

        for (i = 0; i < info->dict_count; i++) {
            PyObject* item = PyList_GET_ITEM(info->dict_list, i);
            PyObject* enc = PyUnicode_AsEncodedString(item, "utf-8", NULL);

            if (!enc) {
                goto bail;
            }

            BAIL_ON_NONZERO(_encode_longlong(PyBytes_GET_SIZE(enc), buffer));
            WRITE_OR_BAIL(PyBytes_AS_STRING(enc), PyBytes_GET_SIZE(enc));
            Py_DECREF(enc);
        }

    } else {  /* SOA_STRING_OFFSET */
        char hdr[] = {ARRAY_START, CONTAINER_TYPE, info->index_marker, ARRAY_END};
        WRITE_OR_BAIL(hdr, 4);
    }

    return 0;

bail:
    return 1;
}

/* Write string value based on encoding */
static int _write_string_value(PyObject* str_val, _string_field_info_t* info,
                               Py_ssize_t record_index, _bjdata_encoder_buffer_t* buffer) {
    PyObject* utf8 = PyUnicode_AsEncodedString(str_val, "utf-8", NULL);

    if (!utf8) {
        return 1;
    }

    int ret = 0;

    if (info->encoding == SOA_STRING_FIXED) {
        Py_ssize_t len = PyBytes_GET_SIZE(utf8);
        Py_ssize_t write_len = (len < info->fixed_len) ? len : info->fixed_len;

        if (_encoder_buffer_write(buffer, PyBytes_AS_STRING(utf8), write_len) != 0) {
            ret = 1;
        } else {
            /* Pad with zeros */
            for (Py_ssize_t p = len; p < info->fixed_len && ret == 0; p++) {
                char zero = '\0';

                if (_encoder_buffer_write(buffer, &zero, 1) != 0) {
                    ret = 1;
                }
            }
        }

    } else if (info->encoding == SOA_STRING_DICT) {
        Py_ssize_t idx = PySequence_Index(info->dict_list, str_val);

        if (idx < 0) {
            PyErr_Clear();
            idx = 0;
        }

        ret = _write_index_value(idx, info->index_size, buffer);

    } else {  /* SOA_STRING_OFFSET */
        ret = _write_index_value(record_index, info->index_size, buffer);
    }

    Py_DECREF(utf8);
    return ret;
}

/* Get field value from a record (handles both tuple and object access) */
static PyObject* _get_field_value(PyObject* rec, PyObject* field_name, Py_ssize_t field_index) {
    if (PyTuple_Check(rec)) {
        PyObject* val = PyTuple_GET_ITEM(rec, field_index);
        Py_INCREF(val);
        return val;
    }

    return PyObject_GetItem(rec, field_name);
}

/* Encode numpy structured array as SOA format */
static int _encode_soa(PyArrayObject* arr, _bjdata_encoder_buffer_t* buffer, int is_row_major) {
    PyArray_Descr* dtype = PyArray_DESCR(arr);
    PyObject* names = NULL, *fields = NULL;
    PyArrayObject* flat = NULL;
    npy_intp count, i, j, nf;
    int ndim;
    double threshold;

    /* Per-field info */
    Py_ssize_t* field_offset = NULL;
    Py_ssize_t* field_itemsize = NULL;
    int* field_type = NULL;  /* 0=numeric, 1=bool, 2=string */
    _string_field_info_t* str_info = NULL;
    PyObject** str_values = NULL;

    BAIL_ON_NULL(names = PyObject_GetAttrString((PyObject*)dtype, "names"));
    BAIL_ON_NULL(fields = PyObject_GetAttrString((PyObject*)dtype, "fields"));

    if (!PyMapping_Check(fields)) {
        PyErr_SetString(PyExc_ValueError, "dtype.fields is not a mapping");
        goto bail;
    }

    nf = PyTuple_GET_SIZE(names);
    count = PyArray_SIZE(arr);
    ndim = PyArray_NDIM(arr);
    threshold = (buffer->prefs.soa_threshold >= 0) ? buffer->prefs.soa_threshold : 0.3;

    BAIL_ON_NULL(flat = (PyArrayObject*)PyArray_Flatten(arr, NPY_CORDER));

    /* Allocate per-field arrays */
    BAIL_ON_NULL(field_offset = calloc(nf, sizeof(Py_ssize_t)));
    BAIL_ON_NULL(field_itemsize = calloc(nf, sizeof(Py_ssize_t)));
    BAIL_ON_NULL(field_type = calloc(nf, sizeof(int)));
    BAIL_ON_NULL(str_info = calloc(nf, sizeof(_string_field_info_t)));
    BAIL_ON_NULL(str_values = calloc(nf, sizeof(PyObject*)));

    /* Analyze fields */
    for (i = 0; i < nf; i++) {
        PyObject* fname = PyTuple_GET_ITEM(names, i);
        PyObject* info = PyObject_GetItem(fields, fname);

        if (!info) {
            goto bail;
        }

        PyArray_Descr* fd = (PyArray_Descr*)PyTuple_GET_ITEM(info, 0);
        field_offset[i] = PyLong_AsSsize_t(PyTuple_GET_ITEM(info, 1));
        field_itemsize[i] = DESCR_ELSIZE(fd);

        int type_num = DESCR_TYPE_NUM(fd);

        if (type_num == NPY_BOOL) {
            field_type[i] = 1;
        } else if (_is_string_type(type_num)) {
            field_type[i] = 2;

            if (_analyze_string_field(flat, fname, i, count, threshold, &str_info[i]) < 0) {
                Py_DECREF(info);
                goto bail;
            }

            /* Cache string values for offset encoding */
            if (str_info[i].encoding == SOA_STRING_OFFSET) {
                str_values[i] = PyList_New(count);

                if (!str_values[i]) {
                    Py_DECREF(info);
                    goto bail;
                }

                for (j = 0; j < count; j++) {
                    PyObject* rec = PyArray_GETITEM(flat, PyArray_GETPTR1(flat, j));

                    if (!rec) {
                        Py_DECREF(info);
                        goto bail;
                    }

                    PyObject* val = _get_field_value(rec, fname, i);
                    Py_DECREF(rec);

                    if (!val) {
                        Py_DECREF(info);
                        goto bail;
                    }

                    PyList_SET_ITEM(str_values[i], j, val);
                }
            }
        } else {
            field_type[i] = 0;
        }

        Py_DECREF(info);
    }

    /* Write header */
    WRITE_CHAR_OR_BAIL(is_row_major ? ARRAY_START : OBJECT_START);
    WRITE_CHAR_OR_BAIL(CONTAINER_TYPE);
    WRITE_CHAR_OR_BAIL(OBJECT_START);

    /* Write schema */
    for (i = 0; i < nf; i++) {
        PyObject* nm = PyTuple_GET_ITEM(names, i);
        BAIL_ON_NONZERO(_encode_field_name(nm, buffer));

        if (field_type[i] == 2) {
            BAIL_ON_NONZERO(_write_string_schema(&str_info[i], buffer));
        } else {
            PyObject* info = PyObject_GetItem(fields, nm);

            if (!info) {
                goto bail;
            }

            PyArray_Descr* fd = (PyArray_Descr*)PyTuple_GET_ITEM(info, 0);
            int ret = _write_field_schema_recursive(fd, buffer);
            Py_DECREF(info);

            if (ret != 0) {
                goto bail;
            }
        }
    }

    WRITE_CHAR_OR_BAIL(OBJECT_END);

    /* Write count */
    WRITE_CHAR_OR_BAIL(CONTAINER_COUNT);

    if (ndim > 1) {
        WRITE_CHAR_OR_BAIL(ARRAY_START);

        for (i = 0; i < ndim; i++) {
            BAIL_ON_NONZERO(_encode_longlong(PyArray_DIMS(arr)[i], buffer));
        }

        WRITE_CHAR_OR_BAIL(ARRAY_END);
    } else {
        BAIL_ON_NONZERO(_encode_longlong(count, buffer));
    }

    /* Write payload */
#define WRITE_FIELD(fi, ri) do { \
        if (field_type[fi] == 1) { \
            void* ptr = (char*)PyArray_GETPTR1(flat, ri) + field_offset[fi]; \
            WRITE_CHAR_OR_BAIL(*((npy_bool*)ptr) ? TYPE_BOOL_TRUE : TYPE_BOOL_FALSE); \
        } else if (field_type[fi] == 2) { \
            PyObject* fname = PyTuple_GET_ITEM(names, fi); \
            PyObject* rec = PyArray_GETITEM(flat, PyArray_GETPTR1(flat, ri)); \
            if (!rec) goto bail; \
            PyObject* val = _get_field_value(rec, fname, fi); \
            Py_DECREF(rec); \
            if (!val) goto bail; \
            int ret = _write_string_value(val, &str_info[fi], ri, buffer); \
            Py_DECREF(val); \
            if (ret != 0) goto bail; \
        } else { \
            void* ptr = (char*)PyArray_GETPTR1(flat, ri) + field_offset[fi]; \
            WRITE_OR_BAIL((char*)ptr, field_itemsize[fi]); \
        } \
    } while(0)

    if (is_row_major) {
        for (j = 0; j < count; j++)
            for (i = 0; i < nf; i++) {
                WRITE_FIELD(i, j);
            }
    } else {
        for (i = 0; i < nf; i++)
            for (j = 0; j < count; j++) {
                WRITE_FIELD(i, j);
            }
    }

#undef WRITE_FIELD

    /* Write offset tables and string buffers for OFFSET encoding */
    for (i = 0; i < nf; i++) {
        if (field_type[i] != 2 || str_info[i].encoding != SOA_STRING_OFFSET) {
            continue;
        }

        /* Calculate and write offset table */
        Py_ssize_t offset = 0;
        BAIL_ON_NONZERO(_write_index_value(0, str_info[i].index_size, buffer));

        for (j = 0; j < count; j++) {
            PyObject* val = PyList_GET_ITEM(str_values[i], j);
            PyObject* utf8 = PyUnicode_AsEncodedString(val, "utf-8", NULL);

            if (!utf8) {
                goto bail;
            }

            offset += PyBytes_GET_SIZE(utf8);
            Py_DECREF(utf8);
            BAIL_ON_NONZERO(_write_index_value(offset, str_info[i].index_size, buffer));
        }

        /* Write string buffer */
        for (j = 0; j < count; j++) {
            PyObject* val = PyList_GET_ITEM(str_values[i], j);
            PyObject* utf8 = PyUnicode_AsEncodedString(val, "utf-8", NULL);

            if (!utf8) {
                goto bail;
            }

            WRITE_OR_BAIL(PyBytes_AS_STRING(utf8), PyBytes_GET_SIZE(utf8));
            Py_DECREF(utf8);
        }
    }

    /* Cleanup */
    for (i = 0; i < nf; i++) {
        Py_XDECREF(str_info[i].dict_list);
        Py_XDECREF(str_values[i]);
    }

    free(field_offset);
    free(field_itemsize);
    free(field_type);
    free(str_info);
    free(str_values);
    Py_DECREF(names);
    Py_DECREF(fields);
    Py_DECREF((PyObject*)flat);
    return 0;

bail:

    if (str_info) {
        for (i = 0; i < nf; i++) {
            Py_XDECREF(str_info[i].dict_list);
        }

        free(str_info);
    }

    if (str_values) {
        for (i = 0; i < nf; i++) {
            Py_XDECREF(str_values[i]);
        }

        free(str_values);
    }

    free(field_offset);
    free(field_itemsize);
    free(field_type);
    Py_XDECREF(names);
    Py_XDECREF(fields);
    Py_XDECREF((PyObject*)flat);
    return 1;
}

static int _encode_NDarray(PyObject* obj, _bjdata_encoder_buffer_t* buffer) {
    PyArrayObject* arr;
    Py_INCREF(obj);
    arr = (PyArrayObject*)PyArray_EnsureArray(obj);

    if (arr == NULL) {
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError, "PyArray_EnsureArray failed");
        }

        return 1;
    }

    int type = PyArray_TYPE(arr);

    /* Check if this is a structured array */
    if (type == NPY_VOID) {
        /* Try SOA encoding for structured arrays */
        if (_can_encode_as_soa(arr)) {
            int is_row_major = (buffer->prefs.soa_format == SOA_FORMAT_ROW);
            int result = _encode_soa(arr, buffer, is_row_major);
            Py_DECREF(arr);
            return result;
        }

        /* Structured array not suitable for SOA - convert to Python list of dicts */
        PyObject* list = PyArray_ToList((PyArrayObject*)arr);
        Py_DECREF(arr);

        if (!list) {
            return 1;
        }

        int result = _encode_PySequence(list, buffer);
        Py_DECREF(list);
        return result;
    }

    /* Regular (non-structured) array encoding */
    int ndim = PyArray_NDIM(arr);
    npy_intp bytes = PyArray_ITEMSIZE(arr);

    int marker = _lookup_marker(type);

    if (marker < 0) {
        if (!PyErr_Occurred()) {
            PyErr_Format(PyExc_ValueError, "Unsupported array type: %d", type);
        }

        Py_DECREF(arr);
        return 1;
    }

    if (ndim == 0) { /*scalar*/
        WRITE_CHAR_OR_BAIL((char)marker);

        if (marker == TYPE_STRING) {
            _encode_longlong(bytes, buffer);
        }

        WRITE_OR_BAIL(PyArray_BYTES(arr), bytes);
        Py_DECREF(arr);
        return 0;
    }

    npy_intp* dims = PyArray_DIMS(arr);
    npy_intp total = PyArray_SIZE(arr);

    WRITE_CHAR_OR_BAIL(ARRAY_START);
    WRITE_CHAR_OR_BAIL(CONTAINER_TYPE);

    if (marker == TYPE_STRING) {
        WRITE_CHAR_OR_BAIL(TYPE_CHAR);
    } else {
        WRITE_CHAR_OR_BAIL((char)marker);
    }

    WRITE_CHAR_OR_BAIL(CONTAINER_COUNT);

    WRITE_CHAR_OR_BAIL(ARRAY_START);

    for (int i = 0 ; i < ndim; i++) {
        _encode_longlong(dims[i], buffer);
    }

    if (type == NPY_UNICODE) {
        _encode_longlong(4, buffer);
    }

    WRITE_CHAR_OR_BAIL(ARRAY_END);

    WRITE_OR_BAIL(PyArray_BYTES(arr), bytes * total);
    Py_DECREF(arr);
    // no ARRAY_END since length was specified

    return 0;

bail:

    if (!PyErr_Occurred()) {
        PyErr_SetString(PyExc_RuntimeError, "NDarray encoding failed");
    }

    Py_DECREF(arr);
    return 1;
}

/******************************************************************************/

static int _encode_PyObject_as_PyDecimal(PyObject* obj, _bjdata_encoder_buffer_t* buffer) {
    PyObject* decimal = NULL;

    // Decimal class has no public C API
    BAIL_ON_NULL(decimal =  PyObject_CallFunctionObjArgs((PyObject*)PyDec_Type, obj, NULL));
    BAIL_ON_NONZERO(_encode_PyDecimal(decimal, buffer));
    Py_DECREF(decimal);
    return 0;

bail:
    Py_XDECREF(decimal);
    return 1;
}

static int _encode_PyDecimal(PyObject* obj, _bjdata_encoder_buffer_t* buffer) {
    PyObject* is_finite;
    PyObject* str = NULL;
    PyObject* encoded = NULL;
    const char* raw;
    Py_ssize_t len;

    // Decimal class has no public C API
    BAIL_ON_NULL(is_finite = PyObject_CallMethod(obj, "is_finite", NULL));

    if (Py_True == is_finite) {
#if PY_MAJOR_VERSION >= 3
        BAIL_ON_NULL(str = PyObject_Str(obj));
#else
        BAIL_ON_NULL(str = PyObject_Unicode(obj));
#endif
        BAIL_ON_NULL(encoded = PyUnicode_AsEncodedString(str, "utf-8", NULL));
        raw = PyBytes_AS_STRING(encoded);
        len = PyBytes_GET_SIZE(encoded);

        WRITE_CHAR_OR_BAIL(TYPE_HIGH_PREC);
        BAIL_ON_NONZERO(_encode_longlong(len, buffer));
        WRITE_OR_BAIL(raw, len);
        Py_DECREF(str);
        Py_DECREF(encoded);
    } else {
        WRITE_CHAR_OR_BAIL(TYPE_NULL);
    }

    Py_DECREF(is_finite);
    return 0;

bail:
    Py_XDECREF(is_finite);
    Py_XDECREF(str);
    Py_XDECREF(encoded);
    return 1;
}

/******************************************************************************/

static int _encode_PyUnicode(PyObject* obj, _bjdata_encoder_buffer_t* buffer) {
    PyObject* str;
    const char* raw;
    Py_ssize_t len;

    BAIL_ON_NULL(str = PyUnicode_AsEncodedString(obj, "utf-8", NULL));
    raw = PyBytes_AS_STRING(str);
    len = PyBytes_GET_SIZE(str);

    if (1 == len) {
        WRITE_CHAR_OR_BAIL(TYPE_CHAR);
    } else {
        WRITE_CHAR_OR_BAIL(TYPE_STRING);
        BAIL_ON_NONZERO(_encode_longlong(len, buffer));
    }

    WRITE_OR_BAIL(raw, len);
    Py_DECREF(str);
    return 0;

bail:
    Py_XDECREF(str);
    return 1;
}

/******************************************************************************/

static int _encode_PyFloat(PyObject* obj, _bjdata_encoder_buffer_t* buffer) {
    char numtmp[9]; // holds type char + float32/64
    double abs;
    double num = PyFloat_AsDouble(obj);

    if (-1.0 == num && PyErr_Occurred()) {
        goto bail;
    }

#ifndef USE__BJDATA

#ifdef USE__FPCLASS

    switch (_fpclass(num)) {
        case _FPCLASS_SNAN:
        case _FPCLASS_QNAN:
        case _FPCLASS_NINF:
        case _FPCLASS_PINF:
#else
    switch (fpclassify(num)) {
        case FP_NAN:
        case FP_INFINITE:
#endif
            WRITE_CHAR_OR_BAIL(TYPE_NULL);
            return 0;
#ifdef USE__FPCLASS

        case _FPCLASS_NZ:
        case _FPCLASS_PZ:
#else
        case FP_ZERO:
#endif
            BAIL_ON_NONZERO(_pyfuncs_ubj_PyFloat_Pack4(num, (unsigned char*)&numtmp[1], buffer->prefs.islittle));
            numtmp[0] = TYPE_FLOAT32;
            WRITE_OR_BAIL(numtmp, 5);
            return 0;
#ifdef USE__FPCLASS

        case _FPCLASS_ND:
        case _FPCLASS_PD:
#else
        case FP_SUBNORMAL:
#endif
            BAIL_ON_NONZERO(_encode_PyObject_as_PyDecimal(obj, buffer));
            return 0;
    }


#else /*USE__BJDATA*/


#ifdef USE__FPCLASS

    switch (_fpclass(num)) {
#else

    switch (fpclassify(num)) {
#endif

#ifdef USE__FPCLASS

        case _FPCLASS_NZ:
        case _FPCLASS_PZ:
#else
        case FP_ZERO:
#endif
            BAIL_ON_NONZERO(_pyfuncs_ubj_PyFloat_Pack4(num, (unsigned char*)&numtmp[1], buffer->prefs.islittle));
            numtmp[0] = TYPE_FLOAT32;
            WRITE_OR_BAIL(numtmp, 5);
            return 0;
#ifdef USE__FPCLASS

        case _FPCLASS_ND:
        case _FPCLASS_PD:
#else
        case FP_SUBNORMAL:
#endif
            BAIL_ON_NONZERO(_encode_PyObject_as_PyDecimal(obj, buffer));
            return 0;
    }

#endif

    abs = fabs(num);

    if (!buffer->prefs.no_float32 && 1.18e-38 <= abs && 3.4e38 >= abs) {
        BAIL_ON_NONZERO(_pyfuncs_ubj_PyFloat_Pack4(num, (unsigned char*)&numtmp[1], buffer->prefs.islittle));
        numtmp[0] = TYPE_FLOAT32;
        WRITE_OR_BAIL(numtmp, 5);
    } else {
        BAIL_ON_NONZERO(_pyfuncs_ubj_PyFloat_Pack8(num, (unsigned char*)&numtmp[1], buffer->prefs.islittle));
        numtmp[0] = TYPE_FLOAT64;
        WRITE_OR_BAIL(numtmp, 9);
    }

    return 0;

bail:
    return 1;
}

/******************************************************************************/

#define WRITE_TYPE_AND_INT8_OR_BAIL(c1, c2) {\
        numtmp[0] = c1;\
        numtmp[1] = (char)c2;\
        WRITE_OR_BAIL(numtmp, 2);\
    }
#define WRITE_INT_INTO_NUMTMP(num, size) {\
        /* numtmp also stores type, so need one larger*/\
        if(!islittle){\
            unsigned char i = size + 1;\
            do {\
                numtmp[--i] = (char)num;\
                num >>= 8;\
            } while (i > 1);\
        }else{\
            unsigned char i = 1;\
            do {\
                numtmp[i++] = (char)num;\
                num >>= 8;\
            } while (i < size + 1);\
        }\
    }
#define WRITE_INT16_OR_BAIL(num) {\
        WRITE_INT_INTO_NUMTMP(num, 2);\
        numtmp[0] = TYPE_INT16;\
        WRITE_OR_BAIL(numtmp, 3);\
    }
#define WRITE_INT32_OR_BAIL(num) {\
        WRITE_INT_INTO_NUMTMP(num, 4);\
        numtmp[0] = TYPE_INT32;\
        WRITE_OR_BAIL(numtmp, 5);\
    }
#define WRITE_INT64_OR_BAIL(num) {\
        WRITE_INT_INTO_NUMTMP(num, 8);\
        numtmp[0] = TYPE_INT64;\
        WRITE_OR_BAIL(numtmp, 9);\
    }

#ifdef USE__BJDATA

#define WRITE_UINT16_OR_BAIL(num) {\
        WRITE_INT_INTO_NUMTMP(num, 2);\
        numtmp[0] = TYPE_UINT16;\
        WRITE_OR_BAIL(numtmp, 3);\
    }
#define WRITE_UINT32_OR_BAIL(num) {\
        WRITE_INT_INTO_NUMTMP(num, 4);\
        numtmp[0] = TYPE_UINT32;\
        WRITE_OR_BAIL(numtmp, 5);\
    }
#define WRITE_UINT64_OR_BAIL(num) {\
        WRITE_INT_INTO_NUMTMP(num, 8);\
        numtmp[0] = TYPE_UINT64;\
        WRITE_OR_BAIL(numtmp, 9);\
    }

#endif


static int _encode_longlong(long long num, _bjdata_encoder_buffer_t* buffer) {
    char numtmp[9]; // large enough to hold type + maximum integer (INT64)
    int islittle = (buffer->prefs.islittle);

#ifdef USE__BJDATA

    if (num >= 0) {
        if (num < POWER_TWO(8)) {
            WRITE_TYPE_AND_INT8_OR_BAIL(TYPE_UINT8, num);
        } else if (num < POWER_TWO(16)) {
            WRITE_UINT16_OR_BAIL(num);
        } else if (num < POWER_TWO(32)) {
            WRITE_UINT32_OR_BAIL(num);
        } else {
            WRITE_UINT64_OR_BAIL(num);
        }

#else

    if (num >= 0) {
        if (num < POWER_TWO(8)) {
            WRITE_TYPE_AND_INT8_OR_BAIL(TYPE_UINT8, num);
        } else if (num < POWER_TWO(15)) {
            WRITE_INT16_OR_BAIL(num);
        } else if (num < POWER_TWO(31)) {
            WRITE_INT32_OR_BAIL(num);
        } else {
            WRITE_INT64_OR_BAIL(num);
        }

#endif
    } else if (num >= -(POWER_TWO(7))) {
        WRITE_TYPE_AND_INT8_OR_BAIL(TYPE_INT8, num);
    } else if (num >= -(POWER_TWO(15))) {
        WRITE_INT16_OR_BAIL(num);
    } else if (num >= -(POWER_TWO(31))) {
        WRITE_INT32_OR_BAIL(num);
    } else {
        WRITE_INT64_OR_BAIL(num);
    }

    return 0;

bail:
    return 1;
}

static int _encode_PyLong(PyObject* obj, _bjdata_encoder_buffer_t* buffer) {
    int overflow;
    long long num = PyLong_AsLongLongAndOverflow(obj, &overflow);

    if (overflow) {
        char numtmp[9]; // large enough to hold type + maximum integer (INT64)
        unsigned long long unum = PyLong_AsUnsignedLongLong(obj);
        int islittle = (buffer->prefs.islittle);

        if (PyErr_Occurred()) {
            PyErr_Clear();
            BAIL_ON_NONZERO(_encode_PyObject_as_PyDecimal(obj, buffer));
        } else {
            WRITE_UINT64_OR_BAIL(unum);
        }

        return 0;
    } else if (num == -1 && PyErr_Occurred()) {
        // unexpected as PyLong should fit if not overflowing
        goto bail;
    } else {
        return _encode_longlong(num, buffer);
    }

bail:
    return 1;
}

#if PY_MAJOR_VERSION < 3
static int _encode_PyInt(PyObject* obj, _bjdata_encoder_buffer_t* buffer) {
    long num = PyInt_AsLong(obj);

    if (num == -1 && PyErr_Occurred()) {
        // unexpected as PyInt should fit into long
        return 1;
    } else {
        return _encode_longlong(num, buffer);
    }
}
#endif

/******************************************************************************/

static int _encode_PySequence(PyObject* obj, _bjdata_encoder_buffer_t* buffer) {
    PyObject* ident;        // id of sequence (for checking circular reference)
    PyObject* seq = NULL;   // converted sequence (via PySequence_Fast)
    Py_ssize_t len;
    Py_ssize_t i;
    int seen;

    // circular reference check
    BAIL_ON_NULL(ident = PyLong_FromVoidPtr(obj));

    if ((seen = PySet_Contains(buffer->markers, ident))) {
        if (-1 != seen) {
            PyErr_SetString(PyExc_ValueError, "Circular reference detected");
        }

        goto bail;
    }

    BAIL_ON_NONZERO(PySet_Add(buffer->markers, ident));

    BAIL_ON_NULL(seq = PySequence_Fast(obj, "_encode_PySequence expects sequence"));
    len = PySequence_Fast_GET_SIZE(seq);

    WRITE_CHAR_OR_BAIL(ARRAY_START);

    if (buffer->prefs.container_count) {
        WRITE_CHAR_OR_BAIL(CONTAINER_COUNT);
        BAIL_ON_NONZERO(_encode_longlong(len, buffer));
    }

    for (i = 0; i < len; i++) {
        BAIL_ON_NONZERO(_bjdata_encode_value(PySequence_Fast_GET_ITEM(seq, i), buffer));
    }

    if (!buffer->prefs.container_count) {
        WRITE_CHAR_OR_BAIL(ARRAY_END);
    }

    if (-1 == PySet_Discard(buffer->markers, ident)) {
        goto bail;
    }

    Py_DECREF(ident);
    Py_DECREF(seq);
    return 0;

bail:
    Py_XDECREF(ident);
    Py_XDECREF(seq);
    return 1;
}

/******************************************************************************/

static int _encode_mapping_key(PyObject* obj, _bjdata_encoder_buffer_t* buffer) {
    PyObject* str = NULL;
    const char* raw;
    Py_ssize_t len;

    if (PyUnicode_Check(obj)) {
        BAIL_ON_NULL(str = PyUnicode_AsEncodedString(obj, "utf-8", NULL));
    }

#if PY_MAJOR_VERSION < 3
    else if (PyString_Check(obj)) {
        BAIL_ON_NULL(str = PyString_AsEncodedObject(obj, "utf-8", NULL));
    }

#endif
    else {
        PyErr_SetString(EncoderException, "Mapping keys can only be strings");
        goto bail;
    }

    raw = PyBytes_AS_STRING(str);
    len = PyBytes_GET_SIZE(str);
    BAIL_ON_NONZERO(_encode_longlong(len, buffer));
    WRITE_OR_BAIL(raw, len);
    Py_DECREF(str);
    return 0;

bail:
    Py_XDECREF(str);
    return 1;
}

static int _encode_PyMapping(PyObject* obj, _bjdata_encoder_buffer_t* buffer) {
    PyObject* ident; // id of sequence (for checking circular reference)
    PyObject* items = NULL;
    PyObject* iter = NULL;
    PyObject* item = NULL;
    int seen;

    // circular reference check
    BAIL_ON_NULL(ident = PyLong_FromVoidPtr(obj));

    if ((seen = PySet_Contains(buffer->markers, ident))) {
        if (-1 != seen) {
            PyErr_SetString(PyExc_ValueError, "Circular reference detected");
        }

        goto bail;
    }

    BAIL_ON_NONZERO(PySet_Add(buffer->markers, ident));

    BAIL_ON_NULL(items = PyMapping_Items(obj));

    if (buffer->prefs.sort_keys) {
        BAIL_ON_NONZERO(PyList_Sort(items));
    }

    WRITE_CHAR_OR_BAIL(OBJECT_START);

    if (buffer->prefs.container_count) {
        WRITE_CHAR_OR_BAIL(CONTAINER_COUNT);
        _encode_longlong(PyList_GET_SIZE(items), buffer);
    }

    BAIL_ON_NULL(iter = PyObject_GetIter(items));

    while (NULL != (item = PyIter_Next(iter))) {
        if (!PyTuple_Check(item) || 2 != PyTuple_GET_SIZE(item)) {
            PyErr_SetString(PyExc_ValueError, "items must return 2-tuples");
            goto bail;
        }

        BAIL_ON_NONZERO(_encode_mapping_key(PyTuple_GET_ITEM(item, 0), buffer));
        BAIL_ON_NONZERO(_bjdata_encode_value(PyTuple_GET_ITEM(item, 1), buffer));
        Py_CLEAR(item);
    }

    // for PyIter_Next
    if (PyErr_Occurred()) {
        goto bail;
    }

    if (!buffer->prefs.container_count) {
        WRITE_CHAR_OR_BAIL(OBJECT_END);
    }

    if (-1 == PySet_Discard(buffer->markers, ident)) {
        goto bail;
    }

    Py_DECREF(iter);
    Py_DECREF(items);
    Py_DECREF(ident);
    return 0;

bail:
    Py_XDECREF(item);
    Py_XDECREF(iter);
    Py_XDECREF(items);
    Py_XDECREF(ident);
    return 1;
}

/******************************************************************************/

int _bjdata_encode_value(PyObject* obj, _bjdata_encoder_buffer_t* buffer) {
    PyObject* newobj = NULL; // result of default call (when encoding unsupported types)

    if (Py_None == obj) {
        WRITE_CHAR_OR_BAIL(TYPE_NULL);
    } else if (Py_True == obj) {
        WRITE_CHAR_OR_BAIL(TYPE_BOOL_TRUE);
    } else if (Py_False == obj) {
        WRITE_CHAR_OR_BAIL(TYPE_BOOL_FALSE);
    } else if (PyUnicode_Check(obj)) {
        BAIL_ON_NONZERO(_encode_PyUnicode(obj, buffer));
#if PY_MAJOR_VERSION < 3
    } else if (PyInt_Check(obj) && Py_TYPE(obj) != NULL && strstr(Py_TYPE(obj)->tp_name, "numpy") == NULL) {
        BAIL_ON_NONZERO(_encode_PyInt(obj, buffer));
#endif
    } else if (PyLong_Check(obj)) {
        BAIL_ON_NONZERO(_encode_PyLong(obj, buffer));
    } else if (PyFloat_Check(obj)) {
        BAIL_ON_NONZERO(_encode_PyFloat(obj, buffer));
    } else if (PyDec_Check(obj)) {
        BAIL_ON_NONZERO(_encode_PyDecimal(obj, buffer));
    } else if (PyBytes_Check(obj)) {
        BAIL_ON_NONZERO(_encode_PyBytes(obj, buffer));
    } else if (PyByteArray_Check(obj)) {
        BAIL_ON_NONZERO(_encode_PyByteArray(obj, buffer));
    } else if (PyArray_CheckAnyScalar(obj)) {
        RECURSE_AND_BAIL_ON_NONZERO(_encode_NDarray(obj, buffer), " while encoding a Numpy scalar");
    } else if (PySequence_Check(obj)) {
        if (PyArray_CheckExact(obj)) {
            RECURSE_AND_BAIL_ON_NONZERO(_encode_NDarray(obj, buffer), " while encoding a Numpy ndarray");
        } else {
            RECURSE_AND_BAIL_ON_NONZERO(_encode_PySequence(obj, buffer), " while encoding an array");
        }

        // order important since Mapping could also be Sequence
    } else if (PyMapping_Check(obj)
               // Unfortunately PyMapping_Check is no longer enough, see https://bugs.python.org/issue5945
#if PY_MAJOR_VERSION >= 3
               && PyObject_HasAttrString(obj, "items")
#endif
              ) {
        RECURSE_AND_BAIL_ON_NONZERO(_encode_PyMapping(obj, buffer), " while encoding an object");
    } else if (NULL == obj) {
        PyErr_SetString(PyExc_RuntimeError, "Internal error - _bjdata_encode_value got NULL obj");
        goto bail;
    } else if (NULL != buffer->prefs.default_func) {
        BAIL_ON_NULL(newobj = PyObject_CallFunctionObjArgs(buffer->prefs.default_func, obj, NULL));
        RECURSE_AND_BAIL_ON_NONZERO(_bjdata_encode_value(newobj, buffer), " while encoding with default function");
        Py_DECREF(newobj);
    } else {
        PyErr_Format(EncoderException, "Cannot encode item of type %s", obj->ob_type->tp_name);
        goto bail;
    }

    return 0;

bail:
    Py_XDECREF(newobj);
    return 1;
}

int _bjdata_encoder_init(void) {
    PyObject* tmp_module = NULL;
    PyObject* tmp_obj = NULL;

    // try to determine floating point format / endianess
    _pyfuncs_ubj_detect_formats();

    // allow encoder to access EncoderException & Decimal class
    BAIL_ON_NULL(tmp_module = PyImport_ImportModule("bjdata.encoder"));
    BAIL_ON_NULL(EncoderException = PyObject_GetAttrString(tmp_module, "EncoderException"));
    Py_CLEAR(tmp_module);

    BAIL_ON_NULL(tmp_module = PyImport_ImportModule("decimal"));
    BAIL_ON_NULL(tmp_obj = PyObject_GetAttrString(tmp_module, "Decimal"));

    if (!PyType_Check(tmp_obj)) {
        PyErr_SetString(PyExc_ImportError, "decimal.Decimal type import failure");
        goto bail;
    }

    PyDec_Type = (PyTypeObject*) tmp_obj;
    Py_CLEAR(tmp_module);

    return 0;

bail:
    Py_CLEAR(EncoderException);
    Py_CLEAR(PyDec_Type);
    Py_XDECREF(tmp_obj);
    Py_XDECREF(tmp_module);
    return 1;
}


void _bjdata_encoder_cleanup(void) {
    Py_CLEAR(EncoderException);
    Py_CLEAR(PyDec_Type);
}