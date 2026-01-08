/*
 * NumPy 1.x/2.x compatibility macros for PyArray_Descr access
 *
 * This version avoids using NumPy 2.x API functions by using
 * PyObject attribute access, which works in both versions.
 *
 * Include this AFTER Python.h and numpy headers.
 */

#ifndef NPY_COMPAT_H
#define NPY_COMPAT_H

#include <numpy/ndarraytypes.h>

#if NPY_ABI_VERSION < 0x02000000
/*
 * NumPy 1.x: Direct struct member access
 */
#define DESCR_ELSIZE(d)    (((PyArray_Descr*)(d))->elsize)
#define DESCR_TYPE_NUM(d)  (((PyArray_Descr*)(d))->type_num)
#else
/*
 * NumPy 2.x: Use Python attribute access to avoid API linkage issues
 * This is slightly slower but guaranteed to work.
 */
static inline npy_intp _descr_elsize_compat(PyArray_Descr* d) {
    npy_intp result = 0;
    PyObject* val = PyObject_GetAttrString((PyObject*)d, "itemsize");

    if (val) {
        result = PyLong_AsSsize_t(val);
        Py_DECREF(val);
    }

    PyErr_Clear();
    return result;
}

static inline int _descr_type_num_compat(PyArray_Descr* d) {
    int result = 0;
    PyObject* val = PyObject_GetAttrString((PyObject*)d, "num");

    if (val) {
        result = (int)PyLong_AsLong(val);
        Py_DECREF(val);
    }

    PyErr_Clear();
    return result;
}

#define DESCR_ELSIZE(d)    _descr_elsize_compat((PyArray_Descr*)(d))
#define DESCR_TYPE_NUM(d)  _descr_type_num_compat((PyArray_Descr*)(d))
#endif

#endif /* NPY_COMPAT_H */
