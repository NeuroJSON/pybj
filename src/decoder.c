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

#define NO_IMPORT_ARRAY

#include "numpyapi.h"
#include "common.h"
#include "markers.h"
#include "decoder.h"
#include "python_funcs.h"

/******************************************************************************/

#define RECURSE_AND_RETURN_OR_BAIL(action, recurse_msg) {\
        PyObject *ret;\
        BAIL_ON_NONZERO(Py_EnterRecursiveCall(recurse_msg));\
        ret = (action);\
        Py_LeaveRecursiveCall();\
        return ret;\
    }

#define RAISE_DECODER_EXCEPTION(msg) {\
        PyObject *num = NULL, *str = NULL, *tuple = NULL;\
        if ((num = PyLong_FromSize_t(buffer->total_read)) &&\
                (str = PyUnicode_FromString(msg)) &&\
                (tuple = PyTuple_Pack(2, str, num))) {\
            PyErr_SetObject(DecoderException, tuple);\
        } else {\
            PyErr_Format(DecoderException, "%s (at byte [%zd])", msg, buffer->total_read);\
        }\
        Py_XDECREF(tuple);\
        Py_XDECREF(num);\
        Py_XDECREF(str);\
        goto bail;\
    }

#define ACTION_READ_ERROR(stmt, len, item_str) {\
        if (NULL == (stmt)) {\
            if (read > 0) {\
                goto bail;\
            } else if ((len > 0) || (read < len)) {\
                RAISE_DECODER_EXCEPTION(("Insufficient input (" item_str ")"));\
            }\
        } else if (read < len) {\
            RAISE_DECODER_EXCEPTION(("Insufficient (partial) input (" item_str ")"));\
        }\
    }

#define READ_VIA_FUNC(buffer, readptr, dst) \
    buffer->read_func(buffer, readptr, dst)

#define READ_INTO_OR_BAIL(len, dst_buffer, item_str) {\
        Py_ssize_t read = len;\
        ACTION_READ_ERROR(READ_VIA_FUNC(buffer, &read, dst_buffer), len, item_str);\
    }

#define READ_OR_BAIL(len, dst_buffer, item_str) {\
        Py_ssize_t read = len;\
        ACTION_READ_ERROR((dst_buffer = READ_VIA_FUNC(buffer, &read, NULL)), len, item_str);\
    }

#define READ_OR_BAIL_CAST(len, dst_buffer, cast, item_str) {\
        Py_ssize_t read = len;\
        ACTION_READ_ERROR((dst_buffer = cast READ_VIA_FUNC(buffer, &read, NULL)), len, item_str);\
    }

#define READ_CHAR_OR_BAIL(dst_char, item_str) {\
        const char* tmp;\
        READ_OR_BAIL(1, tmp, item_str);\
        dst_char = tmp[0];\
    }

#define DECODE_UNICODE_OR_BAIL(dst_obj, raw, length, item_str) {\
        if (NULL == ((dst_obj) = PyUnicode_FromStringAndSize(raw, length))) {\
            RAISE_DECODER_EXCEPTION(("Failed to decode utf8: " item_str));\
        }\
    }

#define DECODE_LENGTH_OR_BAIL(length) BAIL_ON_NEGATIVE((length) = _decode_int_non_negative(buffer, NULL))

#define DECODE_LENGTH_OR_BAIL_MARKER(length, marker) \
    BAIL_ON_NEGATIVE((length) = _decode_int_non_negative(buffer, &(marker)))

// decoder buffer size when using fp
#define BUFFER_FP_SIZE 256
// io.SEEK_CUR constant
#define IO_SEEK_CUR 1

static PyObject* DecoderException = NULL;
static PyTypeObject* PyDec_Type = NULL;
#define PyDec_Check(v) PyObject_TypeCheck(v, PyDec_Type)

/******************************************************************************/

typedef struct {
    char marker;      // next marker after container parameters
    int counting;     // indicates whether container has count specified
    long long count;  // number of elements in container
    char type;        // type of container values, if typed, otherwise TYPE_NONE
    int invalid;      // indicates the parameter specification is invalid
    int is_soa;       // indicates this is an SOA format container
    _soa_schema_t* soa_schema;
} _container_params_t;

/* Buffer read functions */
static const char* _decoder_buffer_read_fixed(_bjdata_decoder_buffer_t* buffer, Py_ssize_t* len, char* dst_buffer);
static const char* _decoder_buffer_read_callable(_bjdata_decoder_buffer_t* buffer, Py_ssize_t* len, char* dst_buffer);
static const char* _decoder_buffer_read_buffered(_bjdata_decoder_buffer_t* buffer, Py_ssize_t* len, char* dst_buffer);

/* Decode functions */
static PyObject* _decode_int8(_bjdata_decoder_buffer_t* buffer);
static PyObject* _decode_int16_32(_bjdata_decoder_buffer_t* buffer, Py_ssize_t size);
static PyObject* _decode_uint16_32(_bjdata_decoder_buffer_t* buffer, Py_ssize_t size);
static PyObject* _decode_int64(_bjdata_decoder_buffer_t* buffer);
static PyObject* _decode_uint64(_bjdata_decoder_buffer_t* buffer);
static PyObject* _decode_float32(_bjdata_decoder_buffer_t* buffer);
static PyObject* _decode_float64(_bjdata_decoder_buffer_t* buffer);
static PyObject* _decode_high_prec(_bjdata_decoder_buffer_t* buffer);
static long long _decode_int_non_negative(_bjdata_decoder_buffer_t* buffer, char* given_marker);
static PyObject* _decode_char(_bjdata_decoder_buffer_t* buffer);
static PyObject* _decode_string(_bjdata_decoder_buffer_t* buffer);
static _container_params_t _get_container_params(_bjdata_decoder_buffer_t* buffer, int in_mapping, unsigned int* ndim, long long** dims);
static PyObject* _decode_array(_bjdata_decoder_buffer_t* buffer);
static PyObject* _decode_object_with_pairs_hook(_bjdata_decoder_buffer_t* buffer);
static PyObject* _decode_object(_bjdata_decoder_buffer_t* buffer);

/* SOA support functions */
static void _free_soa_schema(_soa_schema_t* schema);
static int _get_type_info(char type, int* bytelen);
static PyObject* _decode_soa_payload(_bjdata_decoder_buffer_t* buffer, _soa_schema_t* schema, int is_row_major);
static int _decode_soa_field_type(_bjdata_decoder_buffer_t* buffer, _soa_field_t* f);
static _soa_schema_t* _decode_soa_schema(_bjdata_decoder_buffer_t* buffer);

/******************************************************************************/

_bjdata_decoder_buffer_t* _bjdata_decoder_buffer_create(_bjdata_decoder_prefs_t* prefs, PyObject* input,
        PyObject* seek) {
    _bjdata_decoder_buffer_t* buffer;

    if (NULL == (buffer = calloc(1, sizeof(_bjdata_decoder_buffer_t)))) {
        PyErr_NoMemory();
        return NULL;
    }

    buffer->prefs = *prefs;
    buffer->input = input;
    Py_XINCREF(input);

    if (PyObject_CheckBuffer(input)) {
        BAIL_ON_NONZERO(PyObject_GetBuffer(input, &buffer->view, PyBUF_SIMPLE));
        buffer->read_func = _decoder_buffer_read_fixed;
        buffer->view_set = 1;
    } else if (PyCallable_Check(input)) {
        if (NULL == seek) {
            buffer->read_func = _decoder_buffer_read_callable;
        } else {
            buffer->read_func = _decoder_buffer_read_buffered;
            buffer->seek = seek;
            Py_INCREF(seek);
        }
    } else {
        PyErr_SetString(PyExc_TypeError, "Input neither support buffer interface nor is callable");
        goto bail;
    }

    if (Py_None == buffer->prefs.object_hook) {
        buffer->prefs.object_hook = NULL;
    }

    if (Py_None == buffer->prefs.object_pairs_hook) {
        buffer->prefs.object_pairs_hook = NULL;
    }

    return buffer;

bail:
    _bjdata_decoder_buffer_free(&buffer);
    return NULL;
}

int _bjdata_decoder_buffer_free(_bjdata_decoder_buffer_t** buffer) {
    int failed = 0;

    if (NULL != buffer && NULL != *buffer) {
        if ((*buffer)->view_set) {
            if (NULL != (*buffer)->seek && (*buffer)->view.len > (*buffer)->pos) {
                PyObject* type, *value, *traceback, *seek_result;

                PyErr_Fetch(&type, &value, &traceback);
                seek_result = PyObject_CallFunction((*buffer)->seek, "nn",
                                                    ((*buffer)->pos - (*buffer)->view.len), IO_SEEK_CUR);
                Py_XDECREF(seek_result);

                if (NULL != type) {
                    PyErr_Restore(type, value, traceback);
                } else if (NULL == seek_result) {
                    failed = 1;
                }
            }

            PyBuffer_Release(&((*buffer)->view));
            (*buffer)->view_set = 0;
        }

        if (NULL != (*buffer)->tmp_dst) {
            free((*buffer)->tmp_dst);
            (*buffer)->tmp_dst = NULL;
        }

        Py_CLEAR((*buffer)->input);
        Py_CLEAR((*buffer)->seek);
        free(*buffer);
        *buffer = NULL;
    }

    return failed;
}

static const char* _decoder_buffer_read_fixed(_bjdata_decoder_buffer_t* buffer, Py_ssize_t* len, char* dst_buffer) {
    Py_ssize_t old_pos;

    if (0 == *len) {
        return NULL;
    }

    if (buffer->total_read < buffer->view.len) {
        *len = MIN(*len, (buffer->view.len - buffer->total_read));
        old_pos = buffer->total_read;
        buffer->total_read += *len;

        if (NULL != dst_buffer) {
            return memcpy(dst_buffer, &((char*)buffer->view.buf)[old_pos], *len);
        } else {
            return &((char*)buffer->view.buf)[old_pos];
        }
    } else {
        *len = 0;
        return NULL;
    }
}

static const char* _decoder_buffer_read_callable(_bjdata_decoder_buffer_t* buffer, Py_ssize_t* len, char* dst_buffer) {
    PyObject* read_result = NULL;

    if (0 == *len) {
        return NULL;
    }

    if (buffer->view_set) {
        PyBuffer_Release(&buffer->view);
        buffer->view_set = 0;
    }

    BAIL_ON_NULL(read_result = PyObject_CallFunction(buffer->input, "n", *len));
    BAIL_ON_NONZERO(PyObject_GetBuffer(read_result, &buffer->view, PyBUF_SIMPLE));
    buffer->view_set = 1;
    Py_CLEAR(read_result);

    if (0 == buffer->view.len) {
        *len = 0;
        return NULL;
    }

    *len = buffer->view.len;
    buffer->total_read += *len;

    if (NULL != dst_buffer) {
        return memcpy(dst_buffer, buffer->view.buf, *len);
    } else {
        return buffer->view.buf;
    }

bail:
    *len = 1;
    Py_XDECREF(read_result);
    return NULL;
}

static const char* _decoder_buffer_read_buffered(_bjdata_decoder_buffer_t* buffer, Py_ssize_t* len, char* dst_buffer) {
    Py_ssize_t old_pos;
    char* tmp_dst;
    Py_ssize_t remaining_old = 0;
    PyObject* read_result = NULL;

    if (0 == *len) {
        return NULL;
    }

    if (NULL != buffer->tmp_dst) {
        free(buffer->tmp_dst);
        buffer->tmp_dst = NULL;
    }

    if (!buffer->view_set || *len > (buffer->view.len - buffer->pos)) {
        if (NULL == dst_buffer) {
            if (NULL == (tmp_dst = buffer->tmp_dst = malloc(sizeof(char) * (size_t) * len))) {
                PyErr_NoMemory();
                goto bail;
            }
        } else {
            tmp_dst = dst_buffer;
        }

        if (buffer->view_set) {
            remaining_old = buffer->view.len - buffer->pos;

            if (remaining_old > 0) {
                memcpy(tmp_dst, &((char*)buffer->view.buf)[buffer->pos], remaining_old);
                buffer->pos = buffer->view.len;
                buffer->total_read += remaining_old;
            }

            PyBuffer_Release(&buffer->view);
            buffer->view_set = 0;
            buffer->pos = 0;
        }

        BAIL_ON_NULL(read_result = PyObject_CallFunction(buffer->input, "n",
                                   MAX(BUFFER_FP_SIZE, (*len - remaining_old))));
        BAIL_ON_NONZERO(PyObject_GetBuffer(read_result, &buffer->view, PyBUF_SIMPLE));
        buffer->view_set = 1;
        Py_CLEAR(read_result);

        if (0 == remaining_old && buffer->view.len == 0) {
            *len = 0;
            return NULL;
        }

        *len = MIN(*len, (buffer->view.len - buffer->pos) + remaining_old);
        buffer->pos = *len - remaining_old;
        buffer->total_read += buffer->pos;
        memcpy(&tmp_dst[remaining_old], (char*)buffer->view.buf, buffer->pos);
        return tmp_dst;

    } else {
        old_pos = buffer->pos;
        buffer->pos += *len;
        buffer->total_read += *len;

        if (NULL != dst_buffer) {
            return memcpy(dst_buffer, &((char*)buffer->view.buf)[old_pos], *len);
        } else {
            return &((char*)buffer->view.buf)[old_pos];
        }
    }

bail:
    *len = 1;
    Py_XDECREF(read_result);
    return NULL;
}

/******************************************************************************/

/* Helper to create Python int from signed/unsigned byte */
static inline PyObject* _pylong_from_byte(char value, int is_signed) {
    long lval = is_signed ? (long)(signed char)value : (long)(unsigned char)value;
#if PY_MAJOR_VERSION < 3
    return PyInt_FromLong(lval);
#else
    return PyLong_FromLong(lval);
#endif
}

static PyObject* _decode_byte(_bjdata_decoder_buffer_t* buffer) {
    char value;
    READ_CHAR_OR_BAIL(value, "byte");
    return _pylong_from_byte(value, 0);
bail:
    return NULL;
}

static PyObject* _decode_int8(_bjdata_decoder_buffer_t* buffer) {
    char value;
    READ_CHAR_OR_BAIL(value, "int8");
    return _pylong_from_byte(value, 1);
bail:
    return NULL;
}

static PyObject* _decode_uint8(_bjdata_decoder_buffer_t* buffer) {
    char value;
    READ_CHAR_OR_BAIL(value, "uint8");
    return _pylong_from_byte(value, 0);
bail:
    return NULL;
}

static PyObject* _decode_uint16_32(_bjdata_decoder_buffer_t* buffer, Py_ssize_t size) {
    const unsigned char* raw;
    unsigned long value = 0;
    Py_ssize_t i;

    READ_OR_BAIL_CAST(size, raw, (const unsigned char*), "uint16/32");

    if (buffer->prefs.islittle) {
        unsigned char* buf = (unsigned char*)&value;

        for (i = 0; i < size; i++) {
            buf[i] = *raw++;
        }
    } else {
        for (i = size; i > 0; i--) {
            value = (value << 8) | *raw++;
        }
    }

#if PY_MAJOR_VERSION < 3
    return PyInt_FromLong(value);
#else
    return PyLong_FromUnsignedLong(value);
#endif

bail:
    return NULL;
}

static PyObject* _decode_int16_32(_bjdata_decoder_buffer_t* buffer, Py_ssize_t size) {
    const unsigned char* raw;
    long value = 0;
    Py_ssize_t i;

    READ_OR_BAIL_CAST(size, raw, (const unsigned char*), "int16/32");

    if (buffer->prefs.islittle) {
        unsigned char* buf = (unsigned char*)&value;

        for (i = 0; i < size; i++) {
            buf[i] = *raw++;
        }
    } else {
        for (i = size; i > 0; i--) {
            value = (value << 8) | *raw++;
        }
    }

    // extend signed bit
    if (SIZEOF_LONG > size) {
        value |= -(value & (1L << ((8 * size) - 1)));
    }

#if PY_MAJOR_VERSION < 3
    return PyInt_FromLong(value);
#else
    return PyLong_FromLong(value);
#endif

bail:
    return NULL;
}

static PyObject* _decode_uint64(_bjdata_decoder_buffer_t* buffer) {
    const unsigned char* raw;
    unsigned long long value = 0;
    const Py_ssize_t size = 8;
    Py_ssize_t i;

    READ_OR_BAIL_CAST(8, raw, (const unsigned char*), "uint64");

    if (buffer->prefs.islittle) {
        unsigned char* buf = (unsigned char*)&value;

        for (i = 0; i < size; i++) {
            buf[i] = *raw++;
        }
    } else {
        for (i = size; i > 0; i--) {
            value = (value << 8) | *raw++;
        }
    }

    if (value <= ULONG_MAX) {
        return PyLong_FromUnsignedLong(Py_SAFE_DOWNCAST(value, unsigned long long, unsigned long));
    }

    return PyLong_FromUnsignedLongLong(value);

bail:
    return NULL;
}

static PyObject* _decode_int64(_bjdata_decoder_buffer_t* buffer) {
    const unsigned char* raw;
    long long value = 0L;
    const Py_ssize_t size = 8;
    Py_ssize_t i;

    READ_OR_BAIL_CAST(8, raw, (const unsigned char*), "int64");

    if (buffer->prefs.islittle) {
        unsigned char* buf = (unsigned char*)&value;

        for (i = 0; i < size; i++) {
            buf[i] = *raw++;
        }
    } else {
        for (i = size; i > 0; i--) {
            value = (value << 8) | *raw++;
        }
    }

    // extend signed bit
    if (SIZEOF_LONG_LONG > 8) {
        value |= -(value & ((long long)1 << ((8 * size) - 1)));
    }

    if (value >= LONG_MIN && value <= LONG_MAX) {
        return PyLong_FromLong(Py_SAFE_DOWNCAST(value, long long, long));
    }

    return PyLong_FromLongLong(value);

bail:
    return NULL;
}

static long long _decode_int_non_negative(_bjdata_decoder_buffer_t* buffer, char* given_marker) {
    char marker;
    PyObject* int_obj = NULL;
    long long value;

    if (NULL == given_marker) {
        READ_CHAR_OR_BAIL(marker, "Length marker");
    } else {
        marker = *given_marker;
    }

    switch (marker) {
        case TYPE_BYTE:
            BAIL_ON_NULL(int_obj = _decode_byte(buffer));
            break;

        case TYPE_UINT8:
            BAIL_ON_NULL(int_obj = _decode_uint8(buffer));
            break;

        case TYPE_INT8:
            BAIL_ON_NULL(int_obj = _decode_int8(buffer));
            break;

        case TYPE_UINT16:
            BAIL_ON_NULL(int_obj = _decode_uint16_32(buffer, 2));
            break;

        case TYPE_INT16:
            BAIL_ON_NULL(int_obj = _decode_int16_32(buffer, 2));
            break;

        case TYPE_UINT32:
            BAIL_ON_NULL(int_obj = _decode_uint16_32(buffer, 4));
            break;

        case TYPE_INT32:
            BAIL_ON_NULL(int_obj = _decode_int16_32(buffer, 4));
            break;

        case TYPE_UINT64:
            BAIL_ON_NULL(int_obj = _decode_uint64(buffer));
            break;

        case TYPE_INT64:
            BAIL_ON_NULL(int_obj = _decode_int64(buffer));
            break;

        default:
            RAISE_DECODER_EXCEPTION("Integer marker expected");
    }

#if PY_MAJOR_VERSION < 3

    if (PyInt_Check(int_obj)) {
        value = PyInt_AsLong(int_obj);
    } else
#endif
    {
        value = PyLong_AsLongLong(int_obj);
    }

    if (PyErr_Occurred()) {
        goto bail;
    }

    if (value < 0) {
        RAISE_DECODER_EXCEPTION("Negative count/length unexpected");
    }

    Py_XDECREF(int_obj);
    return value;

bail:
    Py_XDECREF(int_obj);
    return -1;
}

static PyObject* _decode_float32(_bjdata_decoder_buffer_t* buffer) {
    const char* raw;
    double value;

    READ_OR_BAIL(4, raw, "float32");
    value = _pyfuncs_ubj_PyFloat_Unpack4((const unsigned char*)raw, buffer->prefs.islittle);

    if ((-1.0 == value) && PyErr_Occurred()) {
        goto bail;
    }

    return PyFloat_FromDouble(value);

bail:
    return NULL;
}

static PyObject* _decode_float64(_bjdata_decoder_buffer_t* buffer) {
    const char* raw;
    double value;

    READ_OR_BAIL(8, raw, "float64");
    value = _pyfuncs_ubj_PyFloat_Unpack8((const unsigned char*)raw, buffer->prefs.islittle);

    if ((-1.0 == value) && PyErr_Occurred()) {
        goto bail;
    }

    return PyFloat_FromDouble(value);

bail:
    return NULL;
}

static PyObject* _decode_high_prec(_bjdata_decoder_buffer_t* buffer) {
    const char* raw;
    PyObject* num_str = NULL;
    PyObject* decimal;
    long long length;

    DECODE_LENGTH_OR_BAIL(length);
    READ_OR_BAIL((Py_ssize_t)length, raw, "highprec");

    DECODE_UNICODE_OR_BAIL(num_str, raw, (Py_ssize_t)length, "highprec");

    BAIL_ON_NULL(decimal = PyObject_CallFunctionObjArgs((PyObject*)PyDec_Type, num_str, NULL));
    Py_XDECREF(num_str);
    return decimal;

bail:
    Py_XDECREF(num_str);
    return NULL;
}

static PyObject* _decode_char(_bjdata_decoder_buffer_t* buffer) {
    char value;
    PyObject* obj = NULL;

    READ_CHAR_OR_BAIL(value, "char");
    DECODE_UNICODE_OR_BAIL(obj, &value, 1, "char");
    return obj;

bail:
    Py_XDECREF(obj);
    return NULL;
}

static PyObject* _decode_string(_bjdata_decoder_buffer_t* buffer) {
    long long length;
    const char* raw;
    PyObject* obj = NULL;

    DECODE_LENGTH_OR_BAIL(length);

    if (length > 0) {
        READ_OR_BAIL((Py_ssize_t)length, raw, "string");
        DECODE_UNICODE_OR_BAIL(obj, raw, (Py_ssize_t)length, "string");
    } else {
        BAIL_ON_NULL(obj = PyUnicode_FromStringAndSize(NULL, 0));
    }

    return obj;

bail:
    Py_XDECREF(obj);
    return NULL;
}

/******************************************************************************/
/* SOA Support Functions */
/******************************************************************************/

/* Check if marker is a valid integer type marker */
static inline int _is_int_marker(char marker) {
    return (marker == TYPE_INT8 || marker == TYPE_UINT8 ||
            marker == TYPE_INT16 || marker == TYPE_UINT16 ||
            marker == TYPE_INT32 || marker == TYPE_UINT32 ||
            marker == TYPE_INT64 || marker == TYPE_UINT64 ||
            marker == TYPE_BYTE);
}

/* Unified type info lookup */
static int _get_type_info(char type, int* bytelen) {
    switch (type) {
        case TYPE_FLOAT16:
            *bytelen = 2;
            return PyArray_HALF;

        case TYPE_FLOAT32:
            *bytelen = 4;
            return PyArray_FLOAT;

        case TYPE_FLOAT64:
            *bytelen = 8;
            return PyArray_DOUBLE;

        case TYPE_INT8:
            *bytelen = 1;
            return PyArray_BYTE;

        case TYPE_UINT8:
        case TYPE_BYTE:
            *bytelen = 1;
            return PyArray_UBYTE;

        case TYPE_INT16:
            *bytelen = 2;
            return PyArray_SHORT;

        case TYPE_UINT16:
            *bytelen = 2;
            return PyArray_USHORT;

        case TYPE_INT32:
            *bytelen = 4;
            return PyArray_INT;

        case TYPE_UINT32:
            *bytelen = 4;
            return PyArray_UINT;

        case TYPE_INT64:
            *bytelen = 8;
            return PyArray_LONGLONG;

        case TYPE_UINT64:
            *bytelen = 8;
            return PyArray_ULONGLONG;

        case TYPE_CHAR:
            *bytelen = 1;
            return PyArray_STRING;

        case TYPE_BOOL_TRUE:
        case TYPE_BOOL_FALSE:
            *bytelen = 1;
            return PyArray_BOOL;

        default:
            *bytelen = 0;
            return -1;
    }
}

/* Free SOA field */
static void _free_soa_field(_soa_field_t* f) {
    if (f->name) {
        free(f->name);
    }

    Py_XDECREF(f->str_dict);

    if (f->nested_fields) {
        for (Py_ssize_t i = 0; i < f->nested_count; i++) {
            _free_soa_field(&f->nested_fields[i]);
        }

        free(f->nested_fields);
    }
}

/* Free SOA schema structure */
static void _free_soa_schema(_soa_schema_t* schema) {
    if (schema) {
        if (schema->fields) {
            for (Py_ssize_t i = 0; i < schema->num_fields; i++) {
                _free_soa_field(&schema->fields[i]);
            }

            free(schema->fields);
        }

        free(schema);
    }
}

/* Read an index value (1/2/4 bytes) */
static Py_ssize_t _soa_read_index(_bjdata_decoder_buffer_t* buffer, int size) {
    const unsigned char* raw;
    Py_ssize_t read = size, val = 0;

    raw = (const unsigned char*)buffer->read_func(buffer, &read, NULL);

    if (!raw || read < size) {
        return -1;
    }

    if (buffer->prefs.islittle) {
        for (int i = size - 1; i >= 0; i--) {
            val = (val << 8) | raw[i];
        }
    } else {
        for (int i = 0; i < size; i++) {
            val = (val << 8) | raw[i];
        }
    }

    return val;
}

/* Initialize an SOA field struct */
static void _init_soa_field(_soa_field_t* f) {
    memset(f, 0, sizeof(_soa_field_t));
    f->str_encoding = -1;
    f->num_elem = 1;
}

/* Parse a nested struct schema */
static int _decode_nested_struct_schema(_bjdata_decoder_buffer_t* buffer, _soa_field_t* f) {
    Py_ssize_t capacity = 8, count = 0;
    _soa_field_t* nested = calloc(capacity, sizeof(_soa_field_t));

    if (!nested) {
        PyErr_NoMemory();
        return -1;
    }

    char marker;
    READ_CHAR_OR_BAIL(marker, "nested struct field marker");

    while (marker != OBJECT_END) {
        if (marker == TYPE_NOOP) {
            READ_CHAR_OR_BAIL(marker, "nested struct marker after noop");
            continue;
        }

        if (count >= capacity) {
            capacity *= 2;
            _soa_field_t* new_nested = realloc(nested, capacity * sizeof(_soa_field_t));

            if (!new_nested) {
                PyErr_NoMemory();
                goto bail;
            }

            nested = new_nested;
        }

        _soa_field_t* nf = &nested[count];
        _init_soa_field(nf);

        /* Read field name */
        long long key_len;
        const char* raw;
        DECODE_LENGTH_OR_BAIL_MARKER(key_len, marker);
        READ_OR_BAIL((Py_ssize_t)key_len, raw, "nested field name");

        if (!(nf->name = malloc(key_len + 1))) {
            PyErr_NoMemory();
            goto bail;
        }

        memcpy(nf->name, raw, key_len);
        nf->name[key_len] = '\0';
        nf->name_len = key_len;

        /* Read field type */
        READ_CHAR_OR_BAIL(nf->type_marker, "nested field type");

        /* Special handling for strings in nested structs */
        if (nf->type_marker == TYPE_STRING) {
            char len_marker;
            READ_CHAR_OR_BAIL(len_marker, "nested string length marker");

            if (!_is_int_marker(len_marker)) {
                RAISE_DECODER_EXCEPTION("SOA schema only supports fixed-length types");
            }

            long long byte_size;
            DECODE_LENGTH_OR_BAIL_MARKER(byte_size, len_marker);

            nf->str_encoding = -1;
            nf->dtype_num = NPY_UNICODE;
            nf->itemsize = byte_size;
            nf->str_fixed_len = byte_size / 4;  /* UCS-4: 4 bytes per char */
        } else {
            if (_decode_soa_field_type(buffer, nf) < 0) {
                goto bail;
            }
        }

        count++;
        READ_CHAR_OR_BAIL(marker, "nested struct next marker");
    }

    f->nested_fields = nested;
    f->nested_count = count;
    f->dtype_num = -2;  /* Special marker for nested struct */

    /* Calculate total itemsize */
    f->itemsize = 0;

    for (Py_ssize_t i = 0; i < count; i++) {
        f->itemsize += nested[i].itemsize;
    }

    return 0;

bail:

    if (nested) {
        for (Py_ssize_t i = 0; i < count; i++) {
            _free_soa_field(&nested[i]);
        }

        free(nested);
    }

    return -1;
}

/* Parse field type (handles scalar, sub-array, nested struct) */
static int _decode_soa_field_type(_bjdata_decoder_buffer_t* buffer, _soa_field_t* f) {
    char marker = f->type_marker;
    int bytelen;

    if (marker == TYPE_STRING) {
        /* Fixed-length string: S<int_marker><length> */
        char len_marker;
        READ_CHAR_OR_BAIL(len_marker, "string length marker");

        if (!_is_int_marker(len_marker)) {
            RAISE_DECODER_EXCEPTION("SOA schema only supports fixed-length types");
        }

        long long byte_len;
        DECODE_LENGTH_OR_BAIL_MARKER(byte_len, len_marker);

        f->str_encoding = SOA_STRING_FIXED;
        f->str_fixed_len = byte_len;
        f->itemsize = byte_len;
        f->dtype_num = NPY_OBJECT;

    } else if (marker == OBJECT_START) {
        /* Nested struct */
        if (_decode_nested_struct_schema(buffer, f) < 0) {
            return -1;
        }

    } else if (marker == ARRAY_START) {
        /* Could be: string dict [$S#...], offset [$type], or sub-array [TTT...] */
        char next;
        READ_CHAR_OR_BAIL(next, "array first element");

        if (next == CONTAINER_TYPE) {
            READ_CHAR_OR_BAIL(next, "optimized array type");

            if (next == TYPE_STRING) {
                /* Dict string: [$S#count ... */
                f->str_encoding = SOA_STRING_DICT;
                READ_CHAR_OR_BAIL(next, "dict count marker");

                if (next != CONTAINER_COUNT) {
                    RAISE_DECODER_EXCEPTION("Expected # in dict schema");
                }

                DECODE_LENGTH_OR_BAIL(f->str_dict_count);
                BAIL_ON_NULL(f->str_dict = PyList_New(f->str_dict_count));

                for (Py_ssize_t i = 0; i < f->str_dict_count; i++) {
                    long long slen;
                    const char* raw;
                    DECODE_LENGTH_OR_BAIL(slen);
                    READ_OR_BAIL((Py_ssize_t)slen, raw, "dict string");
                    PyObject* s = PyUnicode_FromStringAndSize(raw, slen);

                    if (!s) {
                        return -1;
                    }

                    PyList_SET_ITEM(f->str_dict, i, s);
                }

                f->str_index_size = (f->str_dict_count <= 255) ? 1 :
                                    (f->str_dict_count <= 65535) ? 2 : 4;
                f->itemsize = f->str_index_size;
                f->dtype_num = NPY_OBJECT;

            } else if (_get_type_info(next, &bytelen) >= 0) {
                /* Offset string: [$type] */
                f->str_encoding = SOA_STRING_OFFSET;
                f->str_index_size = bytelen;
                f->itemsize = bytelen;
                f->dtype_num = NPY_OBJECT;
                READ_CHAR_OR_BAIL(next, "offset array end");

                if (next != ARRAY_END) {
                    RAISE_DECODER_EXCEPTION("Expected ] in offset schema");
                }
            } else {
                RAISE_DECODER_EXCEPTION("Unsupported optimized array type");
            }

        } else {
            /* Unoptimized sub-array: [TTT...] */
            char elem_type = next;
            Py_ssize_t elem_count = 1;

            if (_get_type_info(elem_type, &bytelen) < 0) {
                RAISE_DECODER_EXCEPTION("Unsupported sub-array element type");
            }

            f->dtype_num = _get_type_info(elem_type, &bytelen);
            f->itemsize = bytelen;

            READ_CHAR_OR_BAIL(next, "sub-array next");

            while (next != ARRAY_END) {
                int same_type = (next == elem_type) ||
                                ((elem_type == TYPE_BOOL_TRUE || elem_type == TYPE_BOOL_FALSE) &&
                                 (next == TYPE_BOOL_TRUE || next == TYPE_BOOL_FALSE));

                if (!same_type) {
                    RAISE_DECODER_EXCEPTION("Mixed types in sub-array");
                }

                elem_count++;
                READ_CHAR_OR_BAIL(next, "sub-array next");
            }

            f->type_marker = elem_type;
            f->num_elem = elem_count;
            f->itemsize = bytelen * elem_count;
        }

    } else if (_get_type_info(marker, &bytelen) >= 0) {
        /* Simple scalar type */
        f->dtype_num = _get_type_info(marker, &bytelen);
        f->itemsize = bytelen;
    } else {
        RAISE_DECODER_EXCEPTION("Unsupported field type in schema");
    }

    return 0;

bail:
    return -1;
}

/* Build numpy dtype for a field */
static PyObject* _build_field_dtype(_soa_field_t* f) {
    /* String with special encoding -> object dtype */
    if (f->str_encoding >= 0) {
        return (PyObject*)PyArray_DescrFromType(NPY_OBJECT);
    }

    /* Unicode string (nested struct, raw bytes) -> U<n> dtype */
    if (f->dtype_num == NPY_UNICODE && f->str_fixed_len > 0) {
        char dtype_str[32];
        snprintf(dtype_str, sizeof(dtype_str), "U%zd", f->str_fixed_len);
        PyObject* dtype_string = PyUnicode_FromString(dtype_str);

        if (!dtype_string) {
            return NULL;
        }

        PyArray_Descr* descr = NULL;

        if (PyArray_DescrConverter(dtype_string, &descr) != NPY_SUCCEED) {
            Py_DECREF(dtype_string);
            return NULL;
        }

        Py_DECREF(dtype_string);
        return (PyObject*)descr;
    }

    /* Nested struct */
    if (f->dtype_num == -2 && f->nested_fields) {
        PyObject* nested_list = PyList_New(f->nested_count);

        if (!nested_list) {
            return NULL;
        }

        for (Py_ssize_t i = 0; i < f->nested_count; i++) {
            _soa_field_t* nf = &f->nested_fields[i];
            PyObject* nfn = PyUnicode_FromStringAndSize(nf->name, nf->name_len);

            if (!nfn) {
                Py_DECREF(nested_list);
                return NULL;
            }

            PyObject* nfd = _build_field_dtype(nf);

            if (!nfd) {
                Py_DECREF(nfn);
                Py_DECREF(nested_list);
                return NULL;
            }

            PyObject* t = PyTuple_Pack(2, nfn, nfd);
            Py_DECREF(nfn);
            Py_DECREF(nfd);

            if (!t) {
                Py_DECREF(nested_list);
                return NULL;
            }

            PyList_SET_ITEM(nested_list, i, t);
        }

        return nested_list;
    }

    /* Sub-array */
    if (f->num_elem > 1) {
        PyArray_Descr* base_descr = PyArray_DescrFromType(f->dtype_num);

        if (!base_descr) {
            return NULL;
        }

        PyObject* shape = PyTuple_Pack(1, PyLong_FromSsize_t(f->num_elem));

        if (!shape) {
            Py_DECREF(base_descr);
            return NULL;
        }

        PyObject* result = PyTuple_Pack(2, (PyObject*)base_descr, shape);
        Py_DECREF(base_descr);
        Py_DECREF(shape);
        return result;
    }

    /* Simple scalar type */
    return (PyObject*)PyArray_DescrFromType(f->dtype_num);
}

/* Decode SOA schema */
static _soa_schema_t* _decode_soa_schema(_bjdata_decoder_buffer_t* buffer) {
    _soa_schema_t* schema = NULL;
    _soa_field_t* fields = NULL;
    Py_ssize_t num_fields = 0, capacity = 16;
    char marker;
    long long key_len;
    const char* raw;

    BAIL_ON_NULL(schema = (_soa_schema_t*)calloc(1, sizeof(_soa_schema_t)));
    BAIL_ON_NULL(fields = (_soa_field_t*)calloc(capacity, sizeof(_soa_field_t)));

    READ_CHAR_OR_BAIL(marker, "SOA schema marker");

    while (marker != OBJECT_END) {
        if (marker == TYPE_NOOP) {
            READ_CHAR_OR_BAIL(marker, "SOA schema marker after no-op");
            continue;
        }

        if (num_fields >= capacity) {
            capacity *= 2;
            _soa_field_t* new_fields = realloc(fields, capacity * sizeof(_soa_field_t));
            BAIL_ON_NULL(new_fields);
            fields = new_fields;
        }

        _soa_field_t* f = &fields[num_fields];
        _init_soa_field(f);

        /* Decode field name */
        DECODE_LENGTH_OR_BAIL_MARKER(key_len, marker);
        READ_OR_BAIL((Py_ssize_t)key_len, raw, "SOA field name");
        BAIL_ON_NULL(f->name = malloc(key_len + 1));
        memcpy(f->name, raw, key_len);
        f->name[key_len] = '\0';
        f->name_len = key_len;

        /* Decode field type */
        READ_CHAR_OR_BAIL(f->type_marker, "SOA field type");

        if (_decode_soa_field_type(buffer, f) < 0) {
            goto bail;
        }

        num_fields++;
        READ_CHAR_OR_BAIL(marker, "SOA schema next marker");
    }

    schema->fields = fields;
    schema->num_fields = num_fields;
    return schema;

bail:

    if (fields) {
        for (Py_ssize_t i = 0; i < num_fields; i++) {
            _free_soa_field(&fields[i]);
        }

        free(fields);
    }

    free(schema);
    return NULL;
}

/* Decode SOA payload into numpy structured array */
static PyObject* _decode_soa_payload(_bjdata_decoder_buffer_t* buffer, _soa_schema_t* schema, int is_row_major) {
    PyObject* result = NULL;
    PyObject* dtype_list = NULL;
    PyArray_Descr* dtype = NULL;
    PyArrayObject* array = NULL;
    char marker;
    long long count = 0;
    long long* dims = NULL;
    unsigned int ndims = 0;
    npy_intp* npy_dims = NULL;
    Py_ssize_t i, j;
    Py_ssize_t* field_offsets = NULL;
    Py_ssize_t itemsize;
    int has_strings = 0;
    Py_ssize_t** offset_tables = NULL;
    char** string_buffers = NULL;

    /* Check for string fields */
    for (i = 0; i < schema->num_fields; i++) {
        if (schema->fields[i].str_encoding >= 0) {
            has_strings = 1;
        }
    }

    /* Read count marker */
    READ_CHAR_OR_BAIL(marker, "SOA count marker");

    if (marker != CONTAINER_COUNT) {
        RAISE_DECODER_EXCEPTION("Expected # after SOA schema");
    }

    /* Read count value or ND dimensions */
    READ_CHAR_OR_BAIL(marker, "SOA count type");

    if (marker == ARRAY_START) {
        long long dim_val;
        BAIL_ON_NULL(dims = malloc(32 * sizeof(long long)));
        READ_CHAR_OR_BAIL(marker, "SOA dims first marker");
        count = 1;

        while (marker != ARRAY_END) {
            DECODE_LENGTH_OR_BAIL_MARKER(dim_val, marker);
            dims[ndims++] = dim_val;
            count *= dim_val;

            if (ndims >= 32) {
                BAIL_ON_NULL(dims = realloc(dims, (ndims + 32) * sizeof(long long)));
            }

            READ_CHAR_OR_BAIL(marker, "SOA dims next marker");
        }
    } else {
        DECODE_LENGTH_OR_BAIL_MARKER(count, marker);
        ndims = 1;
        BAIL_ON_NULL(dims = malloc(sizeof(long long)));
        dims[0] = count;
    }

    /* Build dtype */
    BAIL_ON_NULL(dtype_list = PyList_New(schema->num_fields));
    BAIL_ON_NULL(field_offsets = calloc(schema->num_fields, sizeof(Py_ssize_t)));

    for (i = 0; i < schema->num_fields; i++) {
        _soa_field_t* f = &schema->fields[i];
        PyObject* fn = PyUnicode_FromStringAndSize(f->name, f->name_len);
        BAIL_ON_NULL(fn);
        PyObject* fd = _build_field_dtype(f);
        BAIL_ON_NULL(fd);
        PyObject* t = PyTuple_Pack(2, fn, fd);
        Py_DECREF(fn);
        Py_DECREF(fd);
        BAIL_ON_NULL(t);
        PyList_SET_ITEM(dtype_list, i, t);
    }

    if (PyArray_DescrConverter(dtype_list, &dtype) != NPY_SUCCEED) {
        goto bail;
    }

    BAIL_ON_NULL(npy_dims = malloc(ndims * sizeof(npy_intp)));

    for (i = 0; i < (Py_ssize_t)ndims; i++) {
        npy_dims[i] = dims[i];
    }

    BAIL_ON_NULL(array = (PyArrayObject*)PyArray_SimpleNewFromDescr((int)ndims, npy_dims, dtype));
    dtype = NULL;
    itemsize = PyArray_ITEMSIZE(array);

    /* Get field offsets from actual array dtype */
    {
        int offsets_ok = 0;
        PyArray_Descr* d = PyArray_DESCR(array);
        PyObject* fdict = PyObject_GetAttrString((PyObject*)d, "fields");

        if (fdict && PyMapping_Check(fdict)) {
            for (i = 0; i < schema->num_fields; i++) {
                PyObject* key = PyUnicode_FromString(schema->fields[i].name);

                if (key) {
                    PyObject* info = PyObject_GetItem(fdict, key);
                    Py_DECREF(key);

                    if (info && PyTuple_Check(info) && PyTuple_GET_SIZE(info) >= 2) {
                        field_offsets[i] = PyLong_AsSsize_t(PyTuple_GET_ITEM(info, 1));

                        if (!PyErr_Occurred()) {
                            offsets_ok++;
                        }
                    }

                    Py_XDECREF(info);
                }
            }

            Py_DECREF(fdict);
        } else {
            Py_XDECREF(fdict);
        }

        PyErr_Clear();

        /* Fallback: calculate manually */
        if (offsets_ok != schema->num_fields) {
            Py_ssize_t offset = 0;

            for (i = 0; i < schema->num_fields; i++) {
                field_offsets[i] = offset;
                offset += schema->fields[i].itemsize;
            }
        }
    }

    /* Read payload */
    if (!has_strings) {
        /* Numeric-only path */
#define READ_NUMERIC_FIELD(fi, ri) do { \
        _soa_field_t* f = &schema->fields[fi]; \
        char* ptr = (char*)PyArray_DATA(array) + (ri)*itemsize + field_offsets[fi]; \
        if (f->type_marker == TYPE_BOOL_TRUE || f->type_marker == TYPE_BOOL_FALSE) { \
            char b; READ_CHAR_OR_BAIL(b, "SOA bool"); \
            *((npy_bool*)ptr) = (b == TYPE_BOOL_TRUE); \
        } else { \
            READ_INTO_OR_BAIL(f->itemsize, ptr, "SOA field"); \
        } \
    } while(0)

        if (is_row_major) {
            for (j = 0; j < count; j++) {
                for (i = 0; i < schema->num_fields; i++) {
                    READ_NUMERIC_FIELD(i, j);
                }
            }
        } else {
            for (i = 0; i < schema->num_fields; i++) {
                for (j = 0; j < count; j++) {
                    READ_NUMERIC_FIELD(i, j);
                }
            }
        }

#undef READ_NUMERIC_FIELD
    } else {
        /* Path with string fields */
        BAIL_ON_NULL(offset_tables = calloc(schema->num_fields, sizeof(Py_ssize_t*)));
        BAIL_ON_NULL(string_buffers = calloc(schema->num_fields, sizeof(char*)));

#define READ_SOA_FIELD(fi, ri) do { \
        _soa_field_t* f = &schema->fields[fi]; \
        char* ptr = (char*)PyArray_DATA(array) + (ri)*itemsize + field_offsets[fi]; \
        if (f->str_encoding == SOA_STRING_FIXED) { \
            const char* raw; \
            READ_OR_BAIL(f->itemsize, raw, "SOA fixed str"); \
            Py_ssize_t actual_len = f->itemsize; \
            while (actual_len > 0 && raw[actual_len - 1] == '\0') actual_len--; \
            PyObject* s = PyUnicode_FromStringAndSize(raw, actual_len); \
            if (!s) goto bail; \
            *(PyObject**)ptr = s; \
        } else if (f->str_encoding == SOA_STRING_DICT) { \
            Py_ssize_t idx = _soa_read_index(buffer, f->str_index_size); \
            if (idx < 0 || idx >= f->str_dict_count) RAISE_DECODER_EXCEPTION("SOA dict index out of range"); \
            PyObject* s = PyList_GET_ITEM(f->str_dict, idx); \
            Py_INCREF(s); \
            *(PyObject**)ptr = s; \
        } else if (f->str_encoding == SOA_STRING_OFFSET) { \
            *(Py_ssize_t*)ptr = _soa_read_index(buffer, f->str_index_size); \
        } else if (f->type_marker == TYPE_BOOL_TRUE || f->type_marker == TYPE_BOOL_FALSE) { \
            for (Py_ssize_t e = 0; e < f->num_elem; e++) { \
                char b; READ_CHAR_OR_BAIL(b, "SOA bool"); \
                ((npy_bool*)ptr)[e] = (b == TYPE_BOOL_TRUE); \
            } \
        } else if (f->dtype_num == -2 && f->nested_fields) { \
            READ_INTO_OR_BAIL(f->itemsize, ptr, "SOA nested struct"); \
        } else { \
            READ_INTO_OR_BAIL(f->itemsize, ptr, "SOA field"); \
        } \
    } while(0)

        if (is_row_major) {
            for (j = 0; j < count; j++) {
                for (i = 0; i < schema->num_fields; i++) {
                    READ_SOA_FIELD(i, j);
                }
            }
        } else {
            for (i = 0; i < schema->num_fields; i++) {
                for (j = 0; j < count; j++) {
                    READ_SOA_FIELD(i, j);
                }
            }
        }

#undef READ_SOA_FIELD

        /* Read offset tables and resolve strings */
        for (i = 0; i < schema->num_fields; i++) {
            _soa_field_t* f = &schema->fields[i];

            if (f->str_encoding != SOA_STRING_OFFSET) {
                continue;
            }

            BAIL_ON_NULL(offset_tables[i] = malloc((count + 1) * sizeof(Py_ssize_t)));

            for (j = 0; j <= count; j++) {
                offset_tables[i][j] = _soa_read_index(buffer, f->str_index_size);

                if (offset_tables[i][j] < 0) {
                    RAISE_DECODER_EXCEPTION("Failed to read SOA offset");
                }
            }

            Py_ssize_t buf_len = offset_tables[i][count];

            if (buf_len > 0) {
                const char* raw;
                READ_OR_BAIL(buf_len, raw, "SOA string buffer");
                BAIL_ON_NULL(string_buffers[i] = malloc(buf_len));
                memcpy(string_buffers[i], raw, buf_len);
            }

            for (j = 0; j < count; j++) {
                char* ptr = (char*)PyArray_DATA(array) + j * itemsize + field_offsets[i];
                Py_ssize_t start = offset_tables[i][j], end = offset_tables[i][j + 1];
                PyObject* s = PyUnicode_FromStringAndSize(
                                  string_buffers[i] ? string_buffers[i] + start : "", end - start);
                BAIL_ON_NULL(s);
                *(PyObject**)ptr = s;
            }
        }
    }

    result = (PyObject*)array;
    array = NULL;

bail:
    free(dims);
    free(npy_dims);
    free(field_offsets);

    if (offset_tables) {
        for (i = 0; i < schema->num_fields; i++) {
            free(offset_tables[i]);
        }

        free(offset_tables);
    }

    if (string_buffers) {
        for (i = 0; i < schema->num_fields; i++) {
            free(string_buffers[i]);
        }

        free(string_buffers);
    }

    Py_XDECREF(dtype_list);
    Py_XDECREF((PyObject*)dtype);
    Py_XDECREF((PyObject*)array);
    return result;
}

/******************************************************************************/

static _container_params_t _get_container_params(_bjdata_decoder_buffer_t* buffer, int in_mapping,
        unsigned int* nd_ndim, long long** nd_dims) {
    _container_params_t params = {0};
    char marker;

    READ_CHAR_OR_BAIL(marker, "container type, count or 1st key/value type");

    if (CONTAINER_TYPE == marker) {
        READ_CHAR_OR_BAIL(marker, "container type");

        /* Check for SOA: ${ indicates schema object */
        if (marker == OBJECT_START) {
            params.soa_schema = _decode_soa_schema(buffer);

            if (!params.soa_schema) {
                params.invalid = 1;
                return params;
            }

            params.is_soa = 1;
            params.type = marker;
            params.counting = 1;
            params.count = -1;
            params.marker = marker;
            params.invalid = 0;
            return params;
        }

        switch (marker) {
            case TYPE_NULL:
            case TYPE_BOOL_TRUE:
            case TYPE_BOOL_FALSE:
            case TYPE_CHAR:
            case TYPE_STRING:
            case TYPE_INT8:
            case TYPE_UINT8:
            case TYPE_INT16:
            case TYPE_INT32:
            case TYPE_INT64:
            case TYPE_FLOAT32:
            case TYPE_FLOAT64:
#ifdef USE__BJDATA
            case TYPE_UINT16:
            case TYPE_UINT32:
            case TYPE_UINT64:
            case TYPE_FLOAT16:
            case TYPE_BYTE:
#endif
            case TYPE_HIGH_PREC:
            case ARRAY_START:
            case OBJECT_START:
                params.type = marker;
                break;

            default:
                RAISE_DECODER_EXCEPTION("Invalid container type");
        }

        READ_CHAR_OR_BAIL(marker, "container count or 1st key/value type");
    } else {
        params.type = TYPE_NONE;
    }

    if (CONTAINER_COUNT == marker) {
        params.counting = 1;
#ifdef USE__BJDATA
        READ_CHAR_OR_BAIL(marker, "container count marker or optimized ND-array dimension array marker");

        if (ARRAY_START == marker && nd_ndim != NULL) {
            long long length = 0, i;
            _container_params_t dims = _get_container_params(buffer, 0, NULL, NULL);
            params.count = 1;

            if (dims.counting) {
                *nd_ndim = dims.count;

                if (dims.count && *nd_dims == NULL) {
                    *nd_dims = (long long*)malloc(sizeof(long long) * (*nd_ndim));
                }

                for (i = 0; i < dims.count; i++) {
                    DECODE_LENGTH_OR_BAIL_MARKER(length, dims.type);
                    params.count *= length;
                    (*nd_dims)[i] = length;
                }
            } else {
                unsigned int i = 0;
                *nd_ndim = 32;
                *nd_dims = (long long*)malloc(sizeof(long long) * (*nd_ndim));
                marker = dims.marker;

                while (ARRAY_END != marker) {
                    DECODE_LENGTH_OR_BAIL_MARKER(length, marker);
                    params.count *= length;
                    (*nd_dims)[i++] = length;

                    if (i >= *nd_ndim) {
                        *nd_ndim += 32;
                        *nd_dims = (long long*)realloc(*nd_dims, sizeof(long long) * (*nd_ndim));
                    }

                    READ_CHAR_OR_BAIL(marker, "Length marker");
                }

                *nd_ndim = i;
                *nd_dims = (long long*)realloc(*nd_dims, sizeof(long long) * i);
            }
        } else
#endif
            DECODE_LENGTH_OR_BAIL_MARKER(params.count, marker);

        if ((params.count > 0) && (in_mapping || (TYPE_NONE == params.type))) {
            READ_CHAR_OR_BAIL(marker, "1st key/value type");
        } else {
            marker = params.type;
        }
    } else if (TYPE_NONE == params.type) {
        params.count = 1;
        params.counting = 0;
    } else {
        RAISE_DECODER_EXCEPTION("Container type without count");
    }

    params.marker = marker;
    params.invalid = 0;
    params.is_soa = 0;
    params.soa_schema = NULL;
    return params;

bail:
    params.invalid = 1;
    return params;
}

/******************************************************************************/

static int _is_no_data_type(char type) {
    return ((TYPE_NULL == type) || (TYPE_BOOL_TRUE == type) || (TYPE_BOOL_FALSE == type));
}

static int _is_fixed_len_type(char type) {
    int bytelen;
    return (_get_type_info(type, &bytelen) >= 0 &&
            type != TYPE_BOOL_TRUE && type != TYPE_BOOL_FALSE);
}

/* Note: Does NOT reserve a new reference */
static PyObject* _no_data_type(char type) {
    switch (type) {
        case TYPE_NULL:
            return Py_None;

        case TYPE_BOOL_TRUE:
            return Py_True;

        case TYPE_BOOL_FALSE:
            return Py_False;

        default:
            PyErr_SetString(PyExc_RuntimeError, "Internal error - _no_data_type");
            return NULL;
    }
}

static PyObject* _decode_array(_bjdata_decoder_buffer_t* buffer) {
    unsigned int ndims = 0;
    long long* dims = NULL;
    _container_params_t params = _get_container_params(buffer, 0, &ndims, &dims);
    PyObject* list = NULL;
    PyObject* value = NULL;
    char marker;

    if (params.invalid) {
        goto bail;
    }

    /* Check if this is SOA format (row-major) */
    if (params.is_soa && params.soa_schema) {
        PyObject* result = _decode_soa_payload(buffer, params.soa_schema, 1);
        _free_soa_schema(params.soa_schema);
        return result;
    }

    marker = params.marker;

    if (params.counting) {
        /* Byte array special case */
        if (((buffer->prefs.uint8_bytes ? TYPE_UINT8 : TYPE_BYTE) == params.type) &&
                !buffer->prefs.no_bytes && ndims == 0) {
            BAIL_ON_NULL(list = PyBytes_FromStringAndSize(NULL, params.count));
            READ_INTO_OR_BAIL(params.count, PyBytes_AS_STRING(list), "bytes array");
            free(dims);
            return list;
        }

        /* ND-array special case */
        if (ndims && params.type) {
            int bytelen = 0;
            int pytype = _get_type_info(params.type, &bytelen);
            npy_intp* arraydim = calloc(sizeof(npy_intp), ndims);
            PyArrayObject* jdarray = NULL;

            for (unsigned int i = 0; i < ndims; i++) {
                arraydim[i] = dims[i];
            }

            BAIL_ON_NULL(jdarray = (PyArrayObject*)PyArray_SimpleNew(ndims, arraydim, pytype));
            READ_INTO_OR_BAIL(bytelen * params.count, (char*)PyArray_DATA(jdarray), "ND array");
            free(arraydim);
            free(dims);
            return PyArray_Return(jdarray);
        }

        /* No-data types special case */
        if (_is_no_data_type(params.type)) {
            BAIL_ON_NULL(list = PyList_New(params.count));
            BAIL_ON_NULL(value = _no_data_type(params.type));

            while (params.count > 0) {
                PyList_SET_ITEM(list, --params.count, value);
                Py_INCREF(value);
            }

            value = NULL;
        }
        /* 1D packed array */
        else if (_is_fixed_len_type(params.type) && params.count > 0) {
            int bytelen = 0;
            int pytype = _get_type_info(params.type, &bytelen);
            npy_intp arraydim = params.count;
            PyArrayObject* jdarray = NULL;

            BAIL_ON_NULL(jdarray = (PyArrayObject*)PyArray_SimpleNew(1, &arraydim, pytype));
            READ_INTO_OR_BAIL(bytelen * params.count, (char*)PyArray_DATA(jdarray), "1D packed array");
            free(dims);
            return PyArray_Return(jdarray);
        }
        /* Generic counted array */
        else {
            Py_ssize_t list_pos = 0;
            BAIL_ON_NULL(list = PyList_New(params.count));

            while (params.count > 0) {
                if (TYPE_NOOP == marker) {
                    READ_CHAR_OR_BAIL(marker, "array value type marker (sized, after no-op)");
                    continue;
                }

                BAIL_ON_NULL(value = _bjdata_decode_value(buffer, &marker));
                PyList_SET_ITEM(list, list_pos++, value);
                value = NULL;
                params.count--;

                if (params.count > 0 && TYPE_NONE == params.type) {
                    READ_CHAR_OR_BAIL(marker, "array value type marker (sized)");
                }
            }
        }
    } else {
        /* Uncounted array */
        BAIL_ON_NULL(list = PyList_New(0));

        while (ARRAY_END != marker) {
            if (TYPE_NOOP == marker) {
                READ_CHAR_OR_BAIL(marker, "array value type marker (after no-op)");
                continue;
            }

            BAIL_ON_NULL(value = _bjdata_decode_value(buffer, &marker));
            BAIL_ON_NONZERO(PyList_Append(list, value));
            Py_CLEAR(value);

            if (TYPE_NONE == params.type) {
                READ_CHAR_OR_BAIL(marker, "array value type marker");
            }
        }
    }

    free(dims);
    return list;

bail:
    Py_XDECREF(value);
    Py_XDECREF(list);
    free(dims);
    return NULL;
}

/******************************************************************************/

/* Decode object key (string without 'S' marker) */
static PyObject* _decode_object_key(_bjdata_decoder_buffer_t* buffer, char marker, int intern) {
    long long length;
    const char* raw;
    PyObject* key;

    DECODE_LENGTH_OR_BAIL_MARKER(length, marker);
    READ_OR_BAIL((Py_ssize_t)length, raw, "string");

    BAIL_ON_NULL(key = PyUnicode_FromStringAndSize(raw, (Py_ssize_t)length));
#if PY_MAJOR_VERSION >= 3

    if (intern) {
        PyUnicode_InternInPlace(&key);
    }

#else
    UNUSED(intern);
#endif
    return key;

bail:
    return NULL;
}

#define DECODE_OBJECT_KEY_OR_RAISE(context_str, intern) {\
        key = _decode_object_key(buffer, marker, intern);\
        if (NULL == key) {\
            RAISE_DECODER_EXCEPTION("Failed to decode object key (" context_str ")");\
        }\
    }

static PyObject* _decode_object_with_pairs_hook(_bjdata_decoder_buffer_t* buffer) {
    _container_params_t params = _get_container_params(buffer, 1, NULL, NULL);
    PyObject* obj = NULL;
    PyObject* list = NULL;
    PyObject* key = NULL;
    PyObject* value = NULL;
    PyObject* item = NULL;
    char* fixed_type;
    char marker;
    int intern = buffer->prefs.intern_object_keys;

    if (params.invalid) {
        goto bail;
    }

    /* Check if this is SOA format */
    if (params.is_soa && params.soa_schema) {
        PyObject* result = _decode_soa_payload(buffer, params.soa_schema, 0);
        _free_soa_schema(params.soa_schema);
        return result;
    }

    marker = params.marker;

    if (params.counting) {
        Py_ssize_t list_pos = 0;
        BAIL_ON_NULL(list = PyList_New(params.count));

        if (_is_no_data_type(params.type)) {
            value = _no_data_type(params.type);
            Py_INCREF(value);

            while (params.count > 0) {
                DECODE_OBJECT_KEY_OR_RAISE("sized, no data", intern);
                BAIL_ON_NULL(item = PyTuple_Pack(2, key, value));
                Py_CLEAR(key);
                PyList_SET_ITEM(list, list_pos++, item);
                item = NULL;
                params.count--;

                if (params.count > 0) {
                    READ_CHAR_OR_BAIL(marker, "object key length");
                }
            }
        } else {
            fixed_type = (TYPE_NONE == params.type) ? NULL : &params.type;

            while (params.count > 0) {
                if (TYPE_NOOP == marker) {
                    READ_CHAR_OR_BAIL(marker, "object key length (sized, after no-op)");
                    continue;
                }

                DECODE_OBJECT_KEY_OR_RAISE("sized", intern);
                BAIL_ON_NULL(value = _bjdata_decode_value(buffer, fixed_type));
                BAIL_ON_NULL(item = PyTuple_Pack(2, key, value));
                Py_CLEAR(key);
                Py_CLEAR(value);
                PyList_SET_ITEM(list, list_pos++, item);
                item = NULL;
                params.count--;

                if (params.count > 0) {
                    READ_CHAR_OR_BAIL(marker, "object key length (sized)");
                }
            }
        }
    } else {
        BAIL_ON_NULL(list = PyList_New(0));
        fixed_type = (TYPE_NONE == params.type) ? NULL : &params.type;

        while (OBJECT_END != marker) {
            if (TYPE_NOOP == marker) {
                READ_CHAR_OR_BAIL(marker, "object key length (after no-op)");
                continue;
            }

            DECODE_OBJECT_KEY_OR_RAISE("unsized", intern);
            BAIL_ON_NULL(value = _bjdata_decode_value(buffer, fixed_type));
            BAIL_ON_NULL(item = PyTuple_Pack(2, key, value));
            Py_CLEAR(key);
            Py_CLEAR(value);
            BAIL_ON_NONZERO(PyList_Append(list, item));
            Py_CLEAR(item);

            READ_CHAR_OR_BAIL(marker, "object key length");
        }
    }

    BAIL_ON_NULL(obj = PyObject_CallFunctionObjArgs(buffer->prefs.object_pairs_hook, list, NULL));
    Py_XDECREF(list);
    return obj;

bail:
    Py_XDECREF(obj);
    Py_XDECREF(list);
    Py_XDECREF(key);
    Py_XDECREF(value);
    Py_XDECREF(item);
    return NULL;
}

static PyObject* _decode_object(_bjdata_decoder_buffer_t* buffer) {
    _container_params_t params = _get_container_params(buffer, 1, NULL, NULL);
    PyObject* obj = NULL;
    PyObject* newobj = NULL;
    PyObject* key = NULL;
    PyObject* value = NULL;
    char* fixed_type;
    char marker;
    int intern = buffer->prefs.intern_object_keys;

    if (params.invalid) {
        goto bail;
    }

    /* Check if this is SOA format */
    if (params.is_soa && params.soa_schema) {
        PyObject* result = _decode_soa_payload(buffer, params.soa_schema, 0);
        _free_soa_schema(params.soa_schema);
        return result;
    }

    marker = params.marker;
    BAIL_ON_NULL(obj = PyDict_New());

    if (params.counting && _is_no_data_type(params.type)) {
        value = _no_data_type(params.type);

        while (params.count > 0) {
            DECODE_OBJECT_KEY_OR_RAISE("sized, no data", intern);
            BAIL_ON_NONZERO(PyDict_SetItem(obj, key, value));
            Py_CLEAR(key);
            Py_INCREF(value);
            params.count--;

            if (params.count > 0) {
                READ_CHAR_OR_BAIL(marker, "object key length");
            }
        }
    } else {
        fixed_type = (TYPE_NONE == params.type) ? NULL : &params.type;

        while (params.count > 0 && (params.counting || (OBJECT_END != marker))) {
            if (TYPE_NOOP == marker) {
                READ_CHAR_OR_BAIL(marker, "object key length");
                continue;
            }

            DECODE_OBJECT_KEY_OR_RAISE("sized/unsized", intern);
            BAIL_ON_NULL(value = _bjdata_decode_value(buffer, fixed_type));
            BAIL_ON_NONZERO(PyDict_SetItem(obj, key, value));
            Py_CLEAR(key);
            Py_CLEAR(value);

            if (params.counting) {
                params.count--;
            }

            if (params.count > 0) {
                READ_CHAR_OR_BAIL(marker, "object key length");
            }
        }
    }

    if (NULL != buffer->prefs.object_hook) {
        BAIL_ON_NULL(newobj = PyObject_CallFunctionObjArgs(buffer->prefs.object_hook, obj, NULL));
        Py_CLEAR(obj);
        return newobj;
    }

    return obj;

bail:
    Py_XDECREF(key);
    Py_XDECREF(value);
    Py_XDECREF(obj);
    Py_XDECREF(newobj);
    return NULL;
}

/******************************************************************************/

#define RETURN_OR_RAISE_DECODER_EXCEPTION(item, item_str) {\
        obj = (item);\
        if (NULL != obj) {\
            return obj;\
        } else if (PyErr_Occurred() && PyErr_ExceptionMatches((PyObject*)DecoderException)) {\
            goto bail;\
        } else {\
            RAISE_DECODER_EXCEPTION("Failed to decode " item_str);\
        }\
    }

PyObject* _bjdata_decode_value(_bjdata_decoder_buffer_t* buffer, char* given_marker) {
    char marker;
    PyObject* obj;

    if (NULL == given_marker) {
        READ_CHAR_OR_BAIL(marker, "Type marker");
    } else {
        marker = *given_marker;
    }

    switch (marker) {
        case TYPE_NULL:
            Py_RETURN_NONE;

        case TYPE_BOOL_TRUE:
            Py_RETURN_TRUE;

        case TYPE_BOOL_FALSE:
            Py_RETURN_FALSE;

        case TYPE_CHAR:
            RETURN_OR_RAISE_DECODER_EXCEPTION(_decode_char(buffer), "char");

        case TYPE_STRING:
            RETURN_OR_RAISE_DECODER_EXCEPTION(_decode_string(buffer), "string");

        case TYPE_INT8:
            RETURN_OR_RAISE_DECODER_EXCEPTION(_decode_int8(buffer), "int8");

        case TYPE_INT16:
            RETURN_OR_RAISE_DECODER_EXCEPTION(_decode_int16_32(buffer, 2), "int16");

        case TYPE_INT32:
            RETURN_OR_RAISE_DECODER_EXCEPTION(_decode_int16_32(buffer, 4), "int32");

        case TYPE_INT64:
            RETURN_OR_RAISE_DECODER_EXCEPTION(_decode_int64(buffer), "int64");
#ifdef USE__BJDATA

        case TYPE_BYTE:
            RETURN_OR_RAISE_DECODER_EXCEPTION(_decode_byte(buffer), "byte");

        case TYPE_UINT8:
            RETURN_OR_RAISE_DECODER_EXCEPTION(_decode_uint8(buffer), "uint8");

        case TYPE_FLOAT16:
        case TYPE_UINT16:
            RETURN_OR_RAISE_DECODER_EXCEPTION(_decode_uint16_32(buffer, 2), "uint16");

        case TYPE_UINT32:
            RETURN_OR_RAISE_DECODER_EXCEPTION(_decode_uint16_32(buffer, 4), "uint32");

        case TYPE_UINT64:
            RETURN_OR_RAISE_DECODER_EXCEPTION(_decode_uint64(buffer), "uint64");
#endif

        case TYPE_FLOAT32:
            RETURN_OR_RAISE_DECODER_EXCEPTION(_decode_float32(buffer), "float32");

        case TYPE_FLOAT64:
            RETURN_OR_RAISE_DECODER_EXCEPTION(_decode_float64(buffer), "float64");

        case TYPE_HIGH_PREC:
            RETURN_OR_RAISE_DECODER_EXCEPTION(_decode_high_prec(buffer), "highprec");

        case ARRAY_START:
            RECURSE_AND_RETURN_OR_BAIL(_decode_array(buffer), "whilst decoding a BJData array");

        case OBJECT_START:
            if (NULL == buffer->prefs.object_pairs_hook) {
                RECURSE_AND_RETURN_OR_BAIL(_decode_object(buffer), "whilst decoding a BJData object");
            } else {
                RECURSE_AND_RETURN_OR_BAIL(_decode_object_with_pairs_hook(buffer), "whilst decoding a BJData object");
            }

        default:
            RAISE_DECODER_EXCEPTION("Invalid marker");
    }

bail:
    return NULL;
}

/******************************************************************************/

int _bjdata_decoder_init(void) {
    PyObject* tmp_module = NULL;
    PyObject* tmp_obj = NULL;

    // try to determine floating point format / endianess
    _pyfuncs_ubj_detect_formats();

    // allow decoder to access DecoderException & Decimal class
    BAIL_ON_NULL(tmp_module = PyImport_ImportModule("bjdata.decoder"));
    BAIL_ON_NULL(DecoderException = PyObject_GetAttrString(tmp_module, "DecoderException"));
    Py_CLEAR(tmp_module);

    BAIL_ON_NULL(tmp_module = PyImport_ImportModule("decimal"));
    BAIL_ON_NULL(tmp_obj = PyObject_GetAttrString(tmp_module, "Decimal"));

    if (!PyType_Check(tmp_obj)) {
        PyErr_SetString(PyExc_ImportError, "decimal.Decimal type import failure");
        goto bail;
    }

    PyDec_Type = (PyTypeObject*)tmp_obj;
    Py_CLEAR(tmp_module);

    return 0;

bail:
    Py_CLEAR(DecoderException);
    Py_CLEAR(PyDec_Type);
    Py_XDECREF(tmp_obj);
    Py_XDECREF(tmp_module);
    return 1;
}

void _bjdata_decoder_cleanup(void) {
    Py_CLEAR(DecoderException);
    Py_CLEAR(PyDec_Type);
}