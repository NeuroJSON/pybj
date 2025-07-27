![](https://neurojson.org/wiki/upload/neurojson_banner_long.png)

# Binary JData for Python - a lightweight binary JSON format

- Copyright: (C) Qianqian Fang (2020-2023) <q.fang at neu.edu>
- Copyright: (C) Iotic Labs Ltd. (2016-2019) <vilnis.termanis at iotic-labs.com>
- License: Apache License, Version 2.0
- Version: 0.4.1
- URL: https://pypi.org/project/bjdata/
- Github: https://github.com/NeuroJSON/pybj
- BJData Spec Version: [V1 Draft 2](https://neurojson.org/bjdata/draft2)
- Acknowledgement: This project is supported by US National Institute of Health (NIH) grant U24-NS124027

[![Build Status](https://travis-ci.com/NeuroJSON/pybj.svg?branch=master)](https://travis-ci.com/NeuroJSON/pybj)

This is a Python v3.2+ (and 2.7+) [Binary JData](https://neurojson.org) encoder
and decoder based on the [Draft-2](Binary_JData_Specification.md) specification.

## Installing / packaging
```shell
## To get from PyPI
pip3 install bjdata
```

Other instalation options
```shell
## Installing / packaging

## To get from PyPI without root/administrator privilege
pip3 install bjdata --user

## To get from PyPI via python
python3 -mpip install bjdata

## To only build extension modules in-place (e.g. in repository)
python3 setup.py build_ext -i

## To build & install globally
python3 setup.py install

## To skip building of extensions when installing (or building)
PYBJDATA_NO_EXTENSION=1 python3 setup.py install
```

This package can be directly installed on Debian Bullseye/Ubuntu 21.04 or newer via
```
sudo apt-get install python3-bjdata
```

Both `python-bjdata` (for Python 2.7+) and `python3-bjdata` (for Python 3.x) can 
also be installed on Ubuntu via
```
sudo add-apt-repository ppa:fangq/ppa
sudo apt-get update
sudo apt-get install python-bjdata python3-bjdata
```

**Notes**

- The extension module is not required but provide a significant speed boost.
- The above can also be run with v2.7+ (replacing `pip3` and `python3` above by `pip` and `python`, respectively)
- At run time, one can check whether compiled version is in use via the 
`bjdata.EXTENSION_ENABLED` boolean


## Usage
It's meant to behave very much like Python's built-in 
[JSON module](https://docs.python.org/3/library/json.html), e.g.:
```python
import bjdata as bj

obj={'a':123,'b':12.3,'c':[1,2,3,[4,5],'test']}
encoded = bj.dumpb(obj)
decoded = bj.loadb(encoded)
```
**Note**: Only unicode strings in Python 2 will be encoded as strings, plain *str* 
will be encoded as a byte array.


## Documentation
```python
import bjdata as bj
help(bj.dump)
help(bj.load)
help(bj.encoder.dump)
help(bj.decoder.load)
```

## Command-line utility
This converts between JSON and BJData formats:
```shell
python3 -mbjdata
USAGE: bjdata (fromjson|tojson) (INFILE|-) [OUTFILE]

EXAMPLES:

python3 -mbjdata fromjson input.json output.bjd
python3 -mbjdata tojson   input.bjd  output.json
```


## Tests

### Static
This library has been checked using [flake8](https://pypi.python.org/pypi/flake8) 
and [pylint](http://www.pylint.org), using a modified configuration - 
see _pylint.rc_ and _flake8.cfg_.

### Unit
```shell
python3 -mvenv py
. py/bin/activate
pip install -U pip setuptools
pip install -e .[dev]

./coverage_test.sh
```
**Note**: See `coverage_test.sh` for additional requirements.


## Limitations
- The **No-Op** type is only supported by the decoder. (This should arguably be 
  a protocol-level rather than serialisation-level option.) Specifically, it is 
  **only** allowed to occur at the start or between elements of a container and 
  **only** inside un-typed containers. (In a typed container it is impossible to 
  tell the difference between an encoded element and a No-Op.)
- Strongly-typed containers are only supported by the decoder (apart from for 
  **bytes**/**bytearray**) and not for No-Op.
- Encoder/decoder extensions are not supported at this time.
- Encoder/decoder only support numpy NDarrays and scalars, other numpy types are not supported.


## Acknowledgement

This package was modified based on the py-ubjson package developed by
[Iotic Labs Ltd.](https://www.iotics.com/) 
Project URL: https://github.com/Iotic-Labs/py-ubjson

The major changes were focused on supporting the Binary JData Specification 
[Draft 2](https://neurojson.org/bjdata/draft2) -
an extended Universal Binary JSON (UBJSON) Specification Draft-12 by adding
the below new features:

* BJData adds 4 new numeric data types: `uint16 [u]`, `uint32 [m]`, `uint64 [M]` and `float16 [h]`
* BJData adds a dedicated byte data type used in optimized array containers for binary data
* BJData supports an optimized ND array container
* BJData does not convert NaN/Inf/-Inf to `null`
* BJData uses little-Endian as the default integer/floating-point numbers while UBJSON uses big-Endian
* BJData only permits non-zero-fixed-length data types (`UiuImlMLhdDCB`) in strongly-typed array/object containers
