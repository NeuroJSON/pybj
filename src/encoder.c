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

const int numpytypes[][2] = {
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

static int _lookup_marker(npy_intp numpytypeid) {
    int i, len = (sizeof(numpytypes) >> 3);

    for (i = 0; i < len; i++) {
        if (numpytypeid == (npy_intp)numpytypes[i][0]) {
            return numpytypes[i][1];
        }
    }

    return -1;
}

/* Get BJData type marker for numpy dtype */
static int _get_soa_type_marker(int dtype_num) {
    if (dtype_num == NPY_BOOL) {
        return TYPE_BOOL_TRUE;
    } else if (dtype_num == NPY_INT8 || dtype_num == NPY_BYTE) {
        return TYPE_INT8;
    } else if (dtype_num == NPY_UINT8 || dtype_num == NPY_UBYTE) {
        return TYPE_UINT8;
    } else if (dtype_num == NPY_INT16 || dtype_num == NPY_SHORT) {
        return TYPE_INT16;
    } else if (dtype_num == NPY_UINT16 || dtype_num == NPY_USHORT) {
        return TYPE_UINT16;
    } else if (dtype_num == NPY_INT32 || dtype_num == NPY_INT) {
        return TYPE_INT32;
    } else if (dtype_num == NPY_UINT32 || dtype_num == NPY_UINT) {
        return TYPE_UINT32;
    } else if (dtype_num == NPY_INT64 || dtype_num == NPY_LONGLONG) {
        return TYPE_INT64;
    } else if (dtype_num == NPY_UINT64 || dtype_num == NPY_ULONGLONG) {
        return TYPE_UINT64;
    } else if (dtype_num == NPY_FLOAT16 || dtype_num == NPY_HALF) {
        return TYPE_FLOAT16;
    } else if (dtype_num == NPY_FLOAT32 || dtype_num == NPY_FLOAT) {
        return TYPE_FLOAT32;
    } else if (dtype_num == NPY_FLOAT64 || dtype_num == NPY_DOUBLE) {
        return TYPE_FLOAT64;
    } else {
        return -1;
    }
}

/* Get item size for a numpy type */
static int _get_type_itemsize(int type_num) {
    if (type_num == NPY_BOOL || type_num == NPY_INT8 || type_num == NPY_BYTE ||
            type_num == NPY_UINT8 || type_num == NPY_UBYTE) {
        return 1;
    } else if (type_num == NPY_INT16 || type_num == NPY_SHORT ||
               type_num == NPY_UINT16 || type_num == NPY_USHORT ||
               type_num == NPY_FLOAT16 || type_num == NPY_HALF) {
        return 2;
    } else if (type_num == NPY_INT32 || type_num == NPY_INT ||
               type_num == NPY_UINT32 || type_num == NPY_UINT ||
               type_num == NPY_FLOAT32 || type_num == NPY_FLOAT) {
        return 4;
    } else if (type_num == NPY_INT64 || type_num == NPY_LONGLONG ||
               type_num == NPY_UINT64 || type_num == NPY_ULONGLONG ||
               type_num == NPY_FLOAT64 || type_num == NPY_DOUBLE) {
        return 8;
    } else {
        return -1;
    }
}

/* Check if numpy array is a structured array suitable for SOA encoding */
static int _can_encode_as_soa(PyArrayObject* arr) {
    PyArray_Descr* dtype = PyArray_DESCR(arr);
    PyObject* names;
    Py_ssize_t i, num_fields;

    /* Must have named fields (structured array) */
    names = PyObject_GetAttrString((PyObject*)dtype, "names");

    if (!names || names == Py_None || !PyTuple_Check(names)) {
        PyErr_Clear();
        Py_XDECREF(names);
        return 0;
    }

    num_fields = PyTuple_GET_SIZE(names);

    if (num_fields == 0) {
        Py_DECREF(names);
        return 0;
    }

    /* Check each field has a supported scalar type */
    PyObject* fields_dict = PyObject_GetAttrString((PyObject*)dtype, "fields");

    if (!fields_dict) {
        PyErr_Clear();
        Py_DECREF(names);
        return 0;
    }

    /* Clear any errors from GetAttrString - fields_dict could be dict or mappingproxy */
    PyErr_Clear();

    for (i = 0; i < num_fields; i++) {
        PyObject* name = PyTuple_GET_ITEM(names, i);
        PyObject* field_info = PyObject_GetItem(fields_dict, name);

        if (!field_info || !PyTuple_Check(field_info) || PyTuple_GET_SIZE(field_info) < 1) {
            Py_XDECREF(field_info);
            Py_DECREF(fields_dict);
            Py_DECREF(names);
            return 0;
        }

        PyArray_Descr* field_dtype = (PyArray_Descr*)PyTuple_GET_ITEM(field_info, 0);

        if (!PyArray_DescrCheck(field_dtype)) {
            Py_DECREF(field_info);
            Py_DECREF(fields_dict);
            Py_DECREF(names);
            return 0;
        }

        /* Check if field dtype is a simple scalar */
        PyObject* field_shape = PyObject_GetAttrString((PyObject*)field_dtype, "shape");

        if (field_shape && PyTuple_Check(field_shape) && PyTuple_GET_SIZE(field_shape) > 0) {
            Py_DECREF(field_shape);
            Py_DECREF(field_info);
            Py_DECREF(fields_dict);
            Py_DECREF(names);
            return 0;
        }

        Py_XDECREF(field_shape);
        PyErr_Clear();  /* Clear any error from GetAttrString */

        /* Check if we have a marker for this type */
        int type_num = field_dtype->type_num;

        if (_get_soa_type_marker(type_num) < 0) {
            Py_DECREF(field_info);
            Py_DECREF(fields_dict);
            Py_DECREF(names);
            return 0;
        }

        Py_DECREF(field_info);
    }

    Py_DECREF(fields_dict);
    Py_DECREF(names);
    return 1;
}

/* Encode numpy structured array as SOA format */
static int _encode_soa(PyArrayObject* arr, _bjdata_encoder_buffer_t* buffer, int is_row_major) {
    PyArray_Descr* dtype = PyArray_DESCR(arr);
    PyObject* names = NULL;
    PyObject* fields_dict = NULL;
    PyArrayObject* flat_arr = NULL;
    Py_ssize_t i, j, num_fields;
    npy_intp count;
    int ndim;
    npy_intp* dims;

    names = PyObject_GetAttrString((PyObject*)dtype, "names");

    if (!names || !PyTuple_Check(names)) {
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_ValueError, "Array dtype has no named fields");
        }

        goto bail;
    }

    num_fields = PyTuple_GET_SIZE(names);

    fields_dict = PyObject_GetAttrString((PyObject*)dtype, "fields");

    if (!fields_dict || !PyMapping_Check(fields_dict)) {
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_ValueError, "Failed to get fields dictionary");
        }

        goto bail;
    }

    ndim = PyArray_NDIM(arr);
    dims = PyArray_DIMS(arr);
    count = PyArray_SIZE(arr);

    /* Flatten the array for easier iteration */
    flat_arr = (PyArrayObject*)PyArray_Flatten(arr, NPY_CORDER);

    if (!flat_arr) {
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_ValueError, "Failed to flatten array");
        }

        goto bail;
    }

    /* Write container start */
    if (is_row_major) {
        WRITE_CHAR_OR_BAIL(ARRAY_START);
    } else {
        WRITE_CHAR_OR_BAIL(OBJECT_START);
    }

    WRITE_CHAR_OR_BAIL(CONTAINER_TYPE);

    /* Write schema object */
    WRITE_CHAR_OR_BAIL(OBJECT_START);

    for (i = 0; i < num_fields; i++) {
        PyObject* name = PyTuple_GET_ITEM(names, i);
        PyObject* field_info = PyObject_GetItem(fields_dict, name);

        if (!field_info) {
            goto bail;
        }

        PyArray_Descr* field_dtype = (PyArray_Descr*)PyTuple_GET_ITEM(field_info, 0);

        /* Write field name */
        PyObject* name_bytes = PyUnicode_AsEncodedString(name, "utf-8", NULL);

        if (!name_bytes) {
            Py_DECREF(field_info);
            goto bail;
        }

        Py_ssize_t name_len = PyBytes_GET_SIZE(name_bytes);
        BAIL_ON_NONZERO(_encode_longlong(name_len, buffer));
        WRITE_OR_BAIL(PyBytes_AS_STRING(name_bytes), name_len);
        Py_DECREF(name_bytes);

        /* Write type marker */
        int marker = _get_soa_type_marker(field_dtype->type_num);
        WRITE_CHAR_OR_BAIL((char)marker);

        Py_DECREF(field_info);
    }

    WRITE_CHAR_OR_BAIL(OBJECT_END);

    /* Write count */
    WRITE_CHAR_OR_BAIL(CONTAINER_COUNT);

    if (ndim > 1) {
        /* ND dimensions */
        WRITE_CHAR_OR_BAIL(ARRAY_START);

        for (i = 0; i < ndim; i++) {
            BAIL_ON_NONZERO(_encode_longlong(dims[i], buffer));
        }

        WRITE_CHAR_OR_BAIL(ARRAY_END);
    } else {
        BAIL_ON_NONZERO(_encode_longlong(count, buffer));
    }

    /* Write payload */
    if (is_row_major) {
        /* Row-major (interleaved): for each record, write all fields */
        for (j = 0; j < count; j++) {
            void* record_ptr = PyArray_GETPTR1(flat_arr, j);

            for (i = 0; i < num_fields; i++) {
                PyObject* name = PyTuple_GET_ITEM(names, i);
                PyObject* field_info = PyObject_GetItem(fields_dict, name);

                if (!field_info) {
                    goto bail;
                }

                PyArray_Descr* field_dtype = (PyArray_Descr*)PyTuple_GET_ITEM(field_info, 0);
                PyObject* offset_obj = PyTuple_GET_ITEM(field_info, 1);
                Py_ssize_t offset = PyLong_AsSsize_t(offset_obj);

                if (offset == -1 && PyErr_Occurred()) {
                    Py_DECREF(field_info);
                    goto bail;
                }

                char* field_ptr = (char*)record_ptr + offset;
                int type_num = field_dtype->type_num;
                int itemsize = _get_type_itemsize(type_num);

                if (itemsize < 0) {
                    PyErr_Format(PyExc_ValueError, "Unsupported field type: %d", type_num);
                    Py_DECREF(field_info);
                    goto bail;
                }

                if (type_num == NPY_BOOL) {
                    npy_bool val = *((npy_bool*)field_ptr);
                    WRITE_CHAR_OR_BAIL(val ? TYPE_BOOL_TRUE : TYPE_BOOL_FALSE);
                } else {
                    WRITE_OR_BAIL(field_ptr, itemsize);
                }

                Py_DECREF(field_info);
            }
        }
    } else {
        /* Column-major (columnar): for each field, write all values */
        for (i = 0; i < num_fields; i++) {
            PyObject* name = PyTuple_GET_ITEM(names, i);
            PyObject* field_info = PyObject_GetItem(fields_dict, name);

            if (!field_info) {
                goto bail;
            }

            PyArray_Descr* field_dtype = (PyArray_Descr*)PyTuple_GET_ITEM(field_info, 0);
            PyObject* offset_obj = PyTuple_GET_ITEM(field_info, 1);
            Py_ssize_t offset = PyLong_AsSsize_t(offset_obj);

            if (offset == -1 && PyErr_Occurred()) {
                Py_DECREF(field_info);
                goto bail;
            }

            int type_num = field_dtype->type_num;
            int itemsize = _get_type_itemsize(type_num);

            if (itemsize < 0) {
                PyErr_Format(PyExc_ValueError, "Unsupported field type: %d", type_num);
                Py_DECREF(field_info);
                goto bail;
            }

            if (type_num == NPY_BOOL) {
                /* Boolean: write T/F for each value */
                for (j = 0; j < count; j++) {
                    void* record_ptr = PyArray_GETPTR1(flat_arr, j);
                    npy_bool val = *((npy_bool*)((char*)record_ptr + offset));
                    WRITE_CHAR_OR_BAIL(val ? TYPE_BOOL_TRUE : TYPE_BOOL_FALSE);
                }
            } else {
                /* Numeric: write raw bytes for each value */
                for (j = 0; j < count; j++) {
                    void* record_ptr = PyArray_GETPTR1(flat_arr, j);
                    char* field_ptr = (char*)record_ptr + offset;
                    WRITE_OR_BAIL(field_ptr, itemsize);
                }
            }

            Py_DECREF(field_info);
        }
    }

    Py_XDECREF(names);
    Py_XDECREF(fields_dict);
    Py_XDECREF((PyObject*)flat_arr);
    return 0;

bail:

    if (!PyErr_Occurred()) {
        PyErr_SetString(PyExc_RuntimeError, "SOA encoding failed");
    }

    Py_XDECREF(names);
    Py_XDECREF(fields_dict);
    Py_XDECREF((PyObject*)flat_arr);
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

    /* Check for SOA encoding */
    if (buffer->prefs.soa_format != SOA_FORMAT_NONE && _can_encode_as_soa(arr)) {
        int is_row_major = (buffer->prefs.soa_format == SOA_FORMAT_ROW);
        int result = _encode_soa(arr, buffer, is_row_major);
        Py_DECREF(arr);
        return result;
    }

    /* Auto-enable column-major SOA for structured arrays when soa_format is NONE */
    if (buffer->prefs.soa_format == SOA_FORMAT_NONE && _can_encode_as_soa(arr)) {
        int result = _encode_soa(arr, buffer, 0);  /* 0 = column-major */
        Py_DECREF(arr);
        return result;
    }

    int ndim = PyArray_NDIM(arr);
    int type = PyArray_TYPE(arr);
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