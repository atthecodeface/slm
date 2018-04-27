ocaml_all: run_ocaml

BREW_PREFIX := $(shell brew config | awk -F': ' '$$1=="HOMEBREW_PREFIX" {print $$2}')
HTML_TIDY   := ${BREW_PREFIX}/bin/tidy -q

OCAML_BUILD_LOCAL      := _build/default/ocaml
OCAML_BUILT_HTML_LOCAL := _build/default/_doc/_html
OCAML_BUILD       := ${ROOT}/${OCAML_BUILD_LOCAL}
OCAML_BUILT_HTML  := ${ROOT}/${OCAML_BUILT_HTML_LOCAL}
OCAML_PATH        := LD_LIBRARY_PATH=${BREW_PREFIX}/lib 
OCAML_DOC         := ${OCAML_ROOT}/doc
DOC_OCAML         := ${DOC_ROOT}/ocaml

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
	${OCAML_PATH} SLMDATA=${SLMDATA} ${OCAML_BUILD}/run_me/run_me.exe --json '{"state":{"n_work_items":"16"}}' -f GuadalupeDemo1.json -a 1 -m 1 -v 5

.PHONY:ocaml_doc
ocaml_doc:
	jbuilder build @doc

.PHONY:ocaml_doc_export
ocaml_doc_export:
	jbuilder build @doc
	cp -r ${OCAML_BUILT_HTML}/* ${DOC_OCAML}
	cp ${OCAML_DOC}/odoc.css ${DOC_OCAML}
	((cd ${DOC_OCAML}; find . -name '*.html' -exec ${HTML_TIDY} -m {} ';') 2>&1 | grep -v 'trimming empty') || true
	#${HTML_TIDY} -o ${DOC_OCAML}/index.html ${OCAML_BUILT_HTML}/index.html
	#(cd _build/default/_doc/_html; find . -name '*.html')

.PHONY:ocaml_clean
clean: ocaml_clean
ocaml_clean:
	jbuilder clean


colin_installation_notes:
	@echo "brew install opam ( installs dependencies: ocaml, ocamlbuild, camlp4, clingo, aspcud)"
	@echo "opam init"
	@echo "eval `opam config env`"
	@echo "echo 'OCAML_TOPLEVEL_PATH=/Users/colinstark/.opam/system/lib/toplevel' >> ~/.bashrc"
	@echo "echo 'export PATH=\"/usr/local/opt/icu4c/bin:$PATH\"' >> ~/.bashrc"
	@echo "echo 'export PATH=\"/usr/local/opt/icu4c/sbin:$PATH\"' >> ~/.bashrc"
	@echo "opam install jbuilder conf-m4 conf-pkg-config ocamlbuild result ocamlfind topkg ocamlmod ocamlify base-bytes integers oasis ctypes eigen"
	@echo "opam install gdal"
	@echo "opam install yojson"
	@echo "opam install batteries"
	@echo "opam install alcotest"
	@echo "opam install odoc"
	@echo "echo 'export PATH=\"/usr/local/opt/libpq/bin:$PATH\"' >> ~/.bashrc"
	@echo "PATH=/usr/local/bin:$PATH opam depext conf-openblas.0.2.0"
	@echo "opam install conf-openblas"
	@echo "opam install utop"
	@echo "PATH=/usr/local/bin:$PATH opam depext conf-zmq.0.1"
	@echo "opam install jupyter"
	@echo "PATH=/usr/local/bin:$PATH opam depext plplot.5.11.0"
	@echo "PATH=/usr/local/bin:$PATH opam install plplot"
	@echo "git clone Owl (from somewhere)"
	@echo "cd Owl"
	@echo "PATH=/usr/local/bin:$PATH make clean"
	@echo "PATH=/usr/local/bin:$PATH make"
	@echo "PATH=/usr/local/bin:$PATH make install"
	@echo "jupyter kernelspec install --name ocaml-jupyter /Users/colinstark/.opam/system/share/jupyter"
