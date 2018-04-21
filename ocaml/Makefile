ocaml_all: run_ocaml

BREW_PREFIX := $(shell brew config | awk -F': ' '$$1=="HOMEBREW_PREFIX" {print $$2}')

OCAML_BUILD_LOCAL := _build/default/ocaml
OCAML_BUILD       := ${ROOT}/${OCAML_BUILD_LOCAL}
OCAML_PATH        := LD_LIBRARY_PATH=${BREW_PREFIX}/lib 

help_ocaml: ocaml_help
ocaml_help:
	@echo "OCaml Help"
	@echo "=========="
	@echo ""
	@echo "Also try make ocaml_help_install, ocaml_help_utop, or ocaml_help_jupyter"
	@echo ""

ocaml_help_install:
	@echo "OCaml Help"
	@echo "=========="
	@echo ""
	@echo "The OCaml implementation should be used with OCaml and its"
	@echo "package management system 'opam'"
	@echo ""
	@echo "The best way to install OCaml is through opam, so a search"
	@echo "for installing opam is the first port of call"
	@echo ""
	@echo "For OSX, use 'brew install opam'"
	@echo "For Debian Linux, use 'apt-get install opam'"
	@echo "For Ubuntu Linux, use :"
	@echo "   add-apt-repository ppa:avsm/ppa"
	@echo "   apt-get update"
	@echo "   apt-get install ocaml ocaml-native-compilers camlp4-extra opam"
	@echo ""
	@echo "The system relies on a few Opam packages, particularly:"
	@echo "       ocaml"
	@echo "       owl"
	@echo "       owl_opencl"
	@echo "       gdal"
	@echo "       yojson"
	@echo "       batteries"
	@echo "       alcotest"
	@echo "       odoc"
	@echo ""
	@echo "Owl depends on base, stdio, configurator - this is useful info if building owl from its git repo"
	@echo ""
	@echo "Currently Owl's opam release does not enable owl_opencl, and"
	@echo "the Owl git repo (which is what I use) requires a couple of"
	@echo "patches"
	@echo ""
	@echo "Gdal is also tricky. The OCaml side is easy as it uses ctypes"
	@echo "The libraries behind it can be installed with brew on OSX."
	@echo ""
	@echo "Note that opam will install the dependencies of libraries"
	@echo "automatically"
	@echo ""
	@echo "Other useful Opam packages are merlin, tuareg, and jupyter"
	@echo ""
	@echo "Note further that camlp4 seems to be a bit of a nightmare"
	@echo "at the moment on OSX. The brew install does not work, and"
	@echo "the opam install fails but claims success. It can be installed"
	@echo "from its git repo with 'make all' and 'make install', and it"
	@echo "installs to the brew installation point (somehow)"
	@echo ""

ocaml_help_utop:
	@echo "utop is a REPL for OCaml (i.e. an interactive front-end)."
	@echo "It is installed with 'opam install utop'"
	@echo "It requires some variables to be set up to be clean"
	@echo ""
	@echo "In my .bash_profile I have:"
	@echo "   export OPAMKEEPBUILDDIR=true"
	@echo "   export OCAMLPARAM=\"_,bin-annot=1\""
	@echo "   export OCAML_TOPLEVEL_PATH=\"/Users/gavinprivate/.opam/system/lib/toplevel\""
	@echo "   And in my .ocamlinit I have:"
	@echo "   let () ="
	@echo "           try Topdirs.dir_directory (Sys.getenv \"OCAML_TOPLEVEL_PATH\")"
	@echo "           with Not_found -> ()"
	@echo "         ;;"
	@echo "   #use \"topfind\""
	@echo ""
	@echo "Then one can run 'utop' and stuff will happen."
	@echo "In utop one can, for example, do :"
	@echo "   Array.fold_left (fun acc x -> x + acc) 0 [|1;2;3;4|];;"
	@echo ""
	@echo "Note that the ';;' indicates that OCaml should treat the input so far as complete"
	@echo ""

ocaml_help_jupyter:
	@echo "To use jupyter, once when has utop running"
	@echo "   opam install jupyter"
	@echo "   jupyter kernelspec install --name ocaml-jupyter /Users/gavinprivate/.opam/system/share/jupyter"
	@echo "   jupyter notebook"

.PHONY: ocaml_run
ocaml_run:
	jbuilder build ${OCAML_BUILD_LOCAL}/run_me/run_me.exe
	${OCAML_PATH} SLMDATA=${SLMDATA} ${OCAML_BUILD}/run_me/run_me.exe -f GuadalupeDemo1.json -a 1 -m 1 -v 5

.PHONY:ocaml_doc
ocaml_doc:
	jbuilder build @doc

.PHONY:ocaml_clean
clean: ocaml_clean
ocaml_clean:
	jbuilder clean

