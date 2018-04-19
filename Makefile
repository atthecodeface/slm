all: help

ROOT :=	${CURDIR}
OCAML_ROOT := ${ROOT}/ocaml
SLMDATA := $(ROOT)/../slmdata

.PHONY: help
help:
	@echo "Streamlines help"
	@echo "This directory should be a sibling of the 'slmdata' repository,"
	@echo "both having the same parent directory"
	@echo "There are Python and OCaml implementations of the streamlines"
	@echo "analysis"
	@echo ""
	@echo "For help on the OCaml implementation use 'make help_ocaml'"

.PHONY:clean
clean:

include ${OCAML_ROOT}/Makefile


