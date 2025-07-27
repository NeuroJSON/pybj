
PY=python3

all: pretty test build
pretty:
	$(PY) -m black test/*.py bjdata/*.py setup.py
	astyle \
	    --style=attach \
	    --indent=spaces=4 \
	    --indent-modifiers \
	    --indent-switches \
	    --indent-preproc-block \
	    --indent-preproc-define \
	    --indent-col1-comments \
	    --pad-oper \
	    --pad-header \
	    --align-pointer=type \
	    --align-reference=type \
	    --add-brackets \
	    --convert-tabs \
	    --close-templates \
	    --lineend=linux \
	    --preserve-date \
	    --suffix=none \
	    --formatted \
	    --break-blocks \
	   "src/*.c" "src/*.h"

test:
	./coverage_test.sh

build:
	$(PY) -m build


.DEFAULT_GOAL=all
.PHONY: all pretty test
