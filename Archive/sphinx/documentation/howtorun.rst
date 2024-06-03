How to run
##########################

`Streamlines`_ workflow can be invoked in several ways.


Interactive IPython/Jupyter notebook
------------------------------------------------------------------------

The recommended approach is to deploy `Streamlines`_ in a Jupyter (browser) 
session and using a Jupyter/IPython notebook. 
`Several example notebooks`_ are provided in the `Streamlines repo`_.
The current development test notebook is: 
:doc:`IndianCreek_Test2.ipynb <../Tests/IndianCreek_Test2_nb>`


Non-interactive Python (external viewer) in a UNIX shell
-----------------------------------------------------------

The most direct approach is to invoke `Streamlines`_ as a Python shell script.
Execution is invoked from the command line with ``slm`` arguments such as these::

   streamlines/run.py -f ./IndianCreek_Test2  -a 1 -p 1
 
or more flexibly::

   streamlines/run.py -f ./IndianCreek_Test2  --analysis yes --plot all 
 
The main workflow steps can all be turned on or off using such flags: 
The ``--help`` option explains in full:

::

	streamlines/run.py  --help
			
	usage: slm.py [-h] [-a analysis_flag] [-c condition_flag] [-d debug_flag]
	              [-e preprocess_flag] [-f parameters_file] [-g geodata_flag]
	              [-i git_info_flag] [-j override_parameters] [-m mapping_flag]
	              [-p maps/pdfs/all] [-q display_flag] [-s save_flag]
	              [-t trace_flag] [-v verbose_flag] [-w n_work_items]
	
	Execute DTM streamline computation
	
	optional arguments:
	  -h, --help            show this help message and exit
	  -a analysis_flag, --analysis analysis_flag
	                        analyze streamline patterns, distributions (default:
	                        None)
	  -c condition_flag, --condition condition_flag
	                        condition DTM for best tracing (fix loops & blockages)
	                        (default: None)
	  -d debug_flag, --debug debug_flag
	                        turn on OpenCL compiler -D DEBUG flag (default: None)
	  -e preprocess_flag, --preprocess preprocess_flag
	                        peform preprocessing (optionally do conditioning;
	                        compute gradients) (default: None)
	  -f parameters_file, --file parameters_file
	                        import JSON parameters file (default: None)
	  -g geodata_flag, --geodata geodata_flag
	                        read geodata files (DTM, basins) (default: None)
	  -i git_info_flag, --info git_info_flag
	                        report info for slm git repos (default: None)
	  -j override_parameters, --json override_parameters
	                        JSON dict of override parameters & values (default:
	                        None)
	  -m mapping_flag, --mapping mapping_flag
	                        map channels, midlines (default: None)
	  -p maps/pdfs/all, --plot maps/pdfs/all
	                        carry out all plotting set in parameters files
	                        (default: None)
	  -q display_flag, --display display_flag
	                        display plots (default: None)
	  -s save_flag, --save save_flag
	                        save state, figs, map grids (default: None)
	  -t trace_flag, --trace trace_flag
	                        perform streamline tracing (default: None)
	  -v verbose_flag, --verbose verbose_flag
	                        verbose mode (default: None)
	  -w n_work_items, --workitems n_work_items
	                        number of OpenCL work items per workgroup (default:
	                        None)



Non-interactive IPython/Jupyter QtConsole (inline graphics) 
-----------------------------------------------------------
If the necessary shell paths are set appropriately, computation can be invoked 
non-interactivey in a Jupyter `QtConsole`_ running an `IPython`_ kernel. 


  
In a IPython/Jupyter QtConsole (inline graphics): interactive or non-interactive
--------------------------------------------------------------------------------

If the necessary shell paths are set appropriately, computation can be invoked 
in a Jupyter `QtConsole`_ running an `IPython`_ kernel. 

::

	Jupyter QtConsole 4.3.1
	Python 3.6.4 (default, Dec 21 2017, 20:33:17) 
	Type 'copyright', 'credits' or 'license' for more information
	IPython 6.2.1 -- An enhanced Interactive Python. Type '?' for help.
	
	run IndianCreek_Test2.ipynb
	
	**Initialization begin**
	etc...

Graphical output will (depending on :mod:`initialize <streamlines.initialize>` 
settings) be displayed inline.

Alternatively, in the  `QtConsole`_:

::

	run ../python/streamlines/slm.py -f GuadalupeMtns1.json -q 1
	
	
TBD: need to tidy up, deal with path issues, possibly with Wurlitzer & caching 
(pycache an nb cache)


In a IPython/Jupyter console (external viewer)  
----------------------------------------------------------------------

Similarly, computation can be invoked from a Jupyter console running IPython. 

::

	% jupyter-console-3.6 IndianCreek_Test2.ipynb 
	Jupyter console 5.2.0
	
	Python 3.6.4 (default, Dec 21 2017, 20:33:17) 
	Type 'copyright', 'credits' or 'license' for more information
	IPython 6.2.1 -- An enhanced Interactive Python. Type '?' for help.
	
	In [1]: run IndianCreek_Test2.ipynb
	
	
	**Initialization begin**
	etc...
	

Graphical output will be pushed to a viewer external to the shell.





.. _Several example notebooks: https://github.com/cstarkjp/Streamlines/blob/master/Tests
.. _Streamlines repo: https://github.com/cstarkjp/Streamlines
.. _Streamlines: https://github.com/cstarkjp/Streamlines
.. _QtConsole: https://ipython.org/ipython-doc/3/interactive/qtconsole.html
.. _IPython: http://ipython.org/ipython-doc/3/interactive/

