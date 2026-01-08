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

#ifndef COMMON_H
#define COMMON_H

#include <Python.h>

/******************************************************************************/

#define UNUSED(x) (void)(x)

#define MIN(a, b) (((a) < (b)) ? (a) : (b))
#define MAX(a, b) (((a) > (b)) ? (a) : (b))

#define BAIL_ON_NULL(action) if (NULL == (action)) goto bail
#define BAIL_ON_NONZERO(action) if (action) goto bail
#define BAIL_ON_NEGATIVE(action) if ((action) < 0) goto bail

/******************************************************************************/

/* SOA string encoding types */
#define SOA_STRING_FIXED   0   /* Fixed-length strings with null padding */
#define SOA_STRING_DICT    1   /* Dictionary encoding with indices */
#define SOA_STRING_OFFSET  2   /* Offset table encoding */

/* SOA format options */
#define SOA_FORMAT_NONE    0   /* Auto-detect / no explicit SOA */
#define SOA_FORMAT_COL     1   /* Column-major (struct of arrays) */
#define SOA_FORMAT_ROW     2   /* Row-major (array of structs) */

/******************************************************************************/

#endif /* COMMON_H */