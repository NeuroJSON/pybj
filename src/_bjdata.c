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
#include <string.h>

#include "common.h"
#include "encoder.h"
#include "decoder.h"
#include "numpyapi.h"

/******************************************************************************/

// container_count, sort_keys, no_float32, islittle, uint8_bytes, soa_format, soa_threshold
static _bjdata_encoder_prefs_t _bjdata_encoder_prefs_defaults = {
    NULL,            /* default_func */
    0,               /* container_count */
    0,               /* sort_keys */
    1,               /* no_float32 */
    1,               /* islittle */
    0,               /* uint8_bytes */
    SOA_FORMAT_NONE, /* soa_format */
    -1.0             /* soa_threshold: -1=auto, 0=force offset, 0.0-1.0=dict ratio */
};

// no_bytes, object_pairs_hook, islittle, uint8_bytes
static _bjdata_decoder_prefs_t _bjdata_decoder_prefs_defaults = { NULL, NULL, 0, 0, 1, 0 };

/******************************************************************************/

/* Parse soa_format string parameter and convert to enum */
static int _parse_soa_format(const char* soa_str) {
    if (soa_str == NULL) {
        return SOA_FORMAT_NONE;
    }

    if (strcmp(soa_str, "col") == 0 || strcmp(soa_str, "column") == 0) {
        return SOA_FORMAT_COL;
    } else if (strcmp(soa_str, "row") == 0 || strcmp(soa_str, "r") == 0) {
        return SOA_FORMAT_ROW;
    }

    return SOA_FORMAT_NONE;
}

/******************************************************************************/

PyDoc_STRVAR(_bjdata_dump__doc__, "See pure Python version (encoder.dump) for documentation.");
#define FUNC_DEF_DUMP {"dump", (PyCFunction)_bjdata_dump, METH_VARARGS | METH_KEYWORDS, _bjdata_dump__doc__}
static PyObject*
_bjdata_dump(PyObject* self, PyObject* args, PyObject* kwargs) {
    static const char* format = "OO|iiiiiOzO:dump";
    static char* keywords[] = {"obj", "fp", "container_count", "sort_keys", "no_float32",
                               "islittle", "uint8_bytes", "default", "soa_format",
                               "soa_threshold", NULL
                              };

    _bjdata_encoder_buffer_t* buffer = NULL;
    _bjdata_encoder_prefs_t prefs = _bjdata_encoder_prefs_defaults;
    PyObject* obj;
    PyObject* fp;
    PyObject* fp_write = NULL;
    char* soa_format_str = NULL;
    PyObject* soa_threshold_obj = NULL;
    UNUSED(self);

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, format, keywords, &obj, &fp,
                                     &prefs.container_count, &prefs.sort_keys,
                                     &prefs.no_float32, &prefs.islittle,
                                     &prefs.uint8_bytes, &prefs.default_func,
                                     &soa_format_str, &soa_threshold_obj)) {
        goto bail;
    }

    prefs.soa_format = _parse_soa_format(soa_format_str);

    /* Parse soa_threshold */
    if (soa_threshold_obj && soa_threshold_obj != Py_None) {
        prefs.soa_threshold = PyFloat_Check(soa_threshold_obj)
                              ? PyFloat_AsDouble(soa_threshold_obj)
                              : (double)PyLong_AsLong(soa_threshold_obj);
    }

    BAIL_ON_NULL(fp_write = PyObject_GetAttrString(fp, "write"));
    BAIL_ON_NULL(buffer = _bjdata_encoder_buffer_create(&prefs, fp_write));
    // buffer creation has added reference
    Py_CLEAR(fp_write);

    BAIL_ON_NONZERO(_bjdata_encode_value(obj, buffer));
    BAIL_ON_NULL(obj = _bjdata_encoder_buffer_finalise(buffer));
    _bjdata_encoder_buffer_free(&buffer);
    return obj;

bail:
    Py_XDECREF(fp_write);
    _bjdata_encoder_buffer_free(&buffer);
    return NULL;
}

PyDoc_STRVAR(_bjdata_dumpb__doc__, "See pure Python version (encoder.dumpb) for documentation.");
#define FUNC_DEF_DUMPB {"dumpb", (PyCFunction)_bjdata_dumpb, METH_VARARGS | METH_KEYWORDS, _bjdata_dumpb__doc__}
static PyObject*
_bjdata_dumpb(PyObject* self, PyObject* args, PyObject* kwargs) {
    static const char* format = "O|iiiiiOzO:dumpb";
    static char* keywords[] = {"obj", "container_count", "sort_keys", "no_float32",
                               "islittle", "uint8_bytes", "default", "soa_format",
                               "soa_threshold", NULL
                              };

    _bjdata_encoder_buffer_t* buffer = NULL;
    _bjdata_encoder_prefs_t prefs = _bjdata_encoder_prefs_defaults;
    PyObject* obj;
    char* soa_format_str = NULL;
    PyObject* soa_threshold_obj = NULL;
    UNUSED(self);

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, format, keywords, &obj,
                                     &prefs.container_count, &prefs.sort_keys,
                                     &prefs.no_float32, &prefs.islittle,
                                     &prefs.uint8_bytes, &prefs.default_func,
                                     &soa_format_str, &soa_threshold_obj)) {
        goto bail;
    }

    prefs.soa_format = _parse_soa_format(soa_format_str);

    /* Parse soa_threshold */
    if (soa_threshold_obj && soa_threshold_obj != Py_None) {
        prefs.soa_threshold = PyFloat_Check(soa_threshold_obj)
                              ? PyFloat_AsDouble(soa_threshold_obj)
                              : (double)PyLong_AsLong(soa_threshold_obj);
    }

    BAIL_ON_NULL(buffer = _bjdata_encoder_buffer_create(&prefs, NULL));
    BAIL_ON_NONZERO(_bjdata_encode_value(obj, buffer));
    BAIL_ON_NULL(obj = _bjdata_encoder_buffer_finalise(buffer));
    _bjdata_encoder_buffer_free(&buffer);
    return obj;

bail:
    _bjdata_encoder_buffer_free(&buffer);
    return NULL;
}

/******************************************************************************/

PyDoc_STRVAR(_bjdata_load__doc__, "See pure Python version (encoder.load) for documentation.");
#define FUNC_DEF_LOAD {"load", (PyCFunction)_bjdata_load, METH_VARARGS | METH_KEYWORDS, _bjdata_load__doc__}
static PyObject*
_bjdata_load(PyObject* self, PyObject* args, PyObject* kwargs) {
    static const char* format = "O|iOOiii:load";
    static char* keywords[] = {"fp", "no_bytes", "object_hook", "object_pairs_hook", "intern_object_keys", "islittle", "uint8_bytes", NULL};

    _bjdata_decoder_buffer_t* buffer = NULL;
    _bjdata_decoder_prefs_t prefs = _bjdata_decoder_prefs_defaults;
    PyObject* fp;
    PyObject* fp_read = NULL;
    PyObject* fp_seek = NULL;
    PyObject* seekable = NULL;
    PyObject* obj = NULL;
    UNUSED(self);

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, format, keywords, &fp, &prefs.no_bytes,  &prefs.object_hook,
                                     &prefs.object_pairs_hook, &prefs.intern_object_keys, &prefs.islittle, &prefs.uint8_bytes)) {
        goto bail;
    }

    BAIL_ON_NULL(fp_read = PyObject_GetAttrString(fp, "read"));

    if (!PyCallable_Check(fp_read)) {
        PyErr_SetString(PyExc_TypeError, "fp.read not callable");
        goto bail;
    }

    // determine whether can seek input
    if (NULL != (seekable = PyObject_CallMethod(fp, "seekable", NULL))) {
        if (Py_True == seekable) {
            // Could also PyCallable_Check but have already checked seekable() so will just fail later
            fp_seek = PyObject_GetAttrString(fp, "seek");
        }

        Py_XDECREF(seekable);
    }

    // ignore seekable() / seek get errors
    PyErr_Clear();

    BAIL_ON_NULL(buffer = _bjdata_decoder_buffer_create(&prefs, fp_read, fp_seek));
    // buffer creation has added references
    Py_CLEAR(fp_read);
    Py_CLEAR(fp_seek);

    BAIL_ON_NULL(obj = _bjdata_decode_value(buffer, NULL));
    BAIL_ON_NONZERO(_bjdata_decoder_buffer_free(&buffer));
    return obj;

bail:
    Py_XDECREF(fp_read);
    Py_XDECREF(fp_seek);
    Py_XDECREF(obj);
    _bjdata_decoder_buffer_free(&buffer);
    return NULL;
}

PyDoc_STRVAR(_bjdata_loadb__doc__, "See pure Python version (encoder.loadb) for documentation.");
#define FUNC_DEF_LOADB {"loadb", (PyCFunction)_bjdata_loadb, METH_VARARGS | METH_KEYWORDS, _bjdata_loadb__doc__}
static PyObject*
_bjdata_loadb(PyObject* self, PyObject* args, PyObject* kwargs) {
    static const char* format = "O|iOOiii:loadb";
    static char* keywords[] = {"chars", "no_bytes", "object_hook", "object_pairs_hook", "intern_object_keys", "islittle", "uint8_bytes", NULL};

    _bjdata_decoder_buffer_t* buffer = NULL;
    _bjdata_decoder_prefs_t prefs = _bjdata_decoder_prefs_defaults;
    PyObject* chars;
    PyObject* obj = NULL;
    UNUSED(self);

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, format, keywords, &chars, &prefs.no_bytes, &prefs.object_hook,
                                     &prefs.object_pairs_hook, &prefs.intern_object_keys, &prefs.islittle, &prefs.uint8_bytes)) {
        goto bail;
    }

    if (PyUnicode_Check(chars)) {
        PyErr_SetString(PyExc_TypeError, "chars must be a bytes-like object, not str");
        goto bail;
    }

    if (!PyObject_CheckBuffer(chars)) {
        PyErr_SetString(PyExc_TypeError, "chars does not support buffer interface");
        goto bail;
    }

    BAIL_ON_NULL(buffer = _bjdata_decoder_buffer_create(&prefs, chars, NULL));

    BAIL_ON_NULL(obj = _bjdata_decode_value(buffer, NULL));
    BAIL_ON_NONZERO(_bjdata_decoder_buffer_free(&buffer));
    return obj;

bail:
    Py_XDECREF(obj);
    _bjdata_decoder_buffer_free(&buffer);
    return NULL;
}

/******************************************************************************/

static PyMethodDef UbjsonMethods[] = {
    FUNC_DEF_DUMP, FUNC_DEF_DUMPB,
    FUNC_DEF_LOAD, FUNC_DEF_LOADB,
    {NULL, NULL, 0, NULL}
};

#if PY_MAJOR_VERSION >= 3
static void module_free(PyObject* m) {
    UNUSED(m);
    _bjdata_encoder_cleanup();
    _bjdata_decoder_cleanup();
}

static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,  // m_base
    "_bjdata",              // m_name
    NULL,                   // m_doc
    -1,                     // m_size
    UbjsonMethods,          // m_methods
    NULL,                   // m_slots
    NULL,                   // m_traverse
    NULL,                   // m_clear
    (freefunc)module_free   // m_free
};

#define INITERROR return NULL
PyObject*
PyInit__bjdata(void)

#else
#define INITERROR return

PyMODINIT_FUNC
init_bjdata(void)
#endif
{
#if PY_MAJOR_VERSION >= 3
    PyObject* module = PyModule_Create(&moduledef);
#else
    PyObject* module = Py_InitModule("_bjdata", UbjsonMethods);
#endif

    import_array();
    BAIL_ON_NONZERO(_bjdata_encoder_init());
    BAIL_ON_NONZERO(_bjdata_decoder_init());

#if PY_MAJOR_VERSION >= 3
    return module;
#else
    return;
#endif

bail:
    _bjdata_encoder_cleanup();
    _bjdata_decoder_cleanup();
    Py_XDECREF(module);
    INITERROR;
}