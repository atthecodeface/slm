How to run
##########################

`Streamlines`_ workflow can be invoked in several ways.


Interactive IPython/Jupyter notebook
------------------------------------------------------------------------

The recommended approach is to deploy `Streamlines`_ in a Jupyter (browser) session and using
an IPython notebook. 
`Several example notebooks`_ are provided in the `Streamlines repo`_.
The current development test notebook is: :doc:`IndianCreekDemo1.ipynb <../Tests/IndianCreekDemo1_nb>`


Non-interactive Python (external viewer) in a UNIX shell
-----------------------------------------------------------

The most direct approach is to invoke `Streamlines`_ as a Python shell script.
Instead of using an IPython notebook, execution is determined using command line  
arguments such as these::

   streamlines/run.py -f ./IndianCreekDemo1  -a 1 -p 1
 
or more flexibly::

   streamlines/run.py -f ./IndianCreekDemo1  --analysis yes --plot all 
 
The main workflow steps can all be turned on or off using such flags: 
The ``--help`` option explains in full:

::

	streamlines/run.py -f ./IndianCreekDemo1  --help
			
	usage: run.py [-h] [-v verbose_flag] [-f parameters_file]
	              [-r reload_state_flag] [-g geodata_flag] [-e preprocess_flag]
	              [-c condition_flag] [-t trace_flag] [-a analysis_flag]
	              [-p maps/pdfs/all] [-s save_state_flag] [-x export_flag]
	
	Execute DTM streamline computation
	
	optional arguments:
	  -h, --help            show this help message and exit
	  -v verbose_flag, --verbose verbose_flag
	                        verbose mode (default: None)
	  -f parameters_file, --file parameters_file
	                        import parameters file (default: None)
	  -r reload_state_flag, --reload reload_state_flag
	                        reload previous runtime state from files (default:
	                        None)
	  -g geodata_flag, --geodata geodata_flag
	                        read geodata files (DTM, basins) (default: None)
	  -e preprocess_flag, --preprocess preprocess_flag
	                        peform preprocessing (optionally do conditioning;
	                        compute gradients) (default: None)
	  -c condition_flag, --condition condition_flag
	                        condition DTM for best tracing (fix loops & blockages)
	                        (default: None)
	  -t trace_flag, --trace trace_flag
	                        perform streamline tracing (default: None)
	  -a analysis_flag, --analysis analysis_flag
	                        analyze streamline patterns, distributions (default:
	                        None)
	  -p maps/pdfs/all, --plot maps/pdfs/all
	                        carry out all plotting set in parameters files
	                        (default: None)
	  -s save_state_flag, --save save_state_flag
	                        save runtime state to files at completion (default:
	                        None)
	  -x export_flag, --export export_flag
	                        export figures to files (default: None)


  
Interactive IPython/Jupyter QtConsole (inline graphics) 
----------------------------------------------------------------------

If the necessary shell paths are set appropriately, computation can be invoked 
from a Jupyter `QtConsole`_
running an `IPython`_ kernel. 

::

	Jupyter QtConsole 4.3.1
	Python 3.6.4 (default, Dec 21 2017, 20:33:17) 
	Type 'copyright', 'credits' or 'license' for more information
	IPython 6.2.1 -- An enhanced Interactive Python. Type '?' for help.
	
	run IndianCreekDemo1.ipynb
	
	**Initialization begin**
	etc...

Graphical output will (depending on :mod:`initialize <streamlines.initialize>` 
settings) be displayed inline.


Interactive IPython/Jupyter console (external viewer)  
----------------------------------------------------------------------

Similarly, computation can be invoked from a Jupyter running IPython. 

::

	% jupyter-console-3.6 IndianCreekDemo1.ipynb 
	Jupyter console 5.2.0
	
	Python 3.6.4 (default, Dec 21 2017, 20:33:17) 
	Type 'copyright', 'credits' or 'license' for more information
	IPython 6.2.1 -- An enhanced Interactive Python. Type '?' for help.
	
	In [1]: run IndianCreekDemo1.ipynb
	
	
	**Initialization begin**
	etc...
	
Graphical output will be pushed to a viewer external to the shell.





.. _Several example notebooks: https://github.com/cstarknyc/Streamlines/blob/master/Tests
.. _Streamlines repo: https://github.com/cstarknyc/Streamlines
.. _Streamlines: https://github.com/cstarknyc/Streamlines
.. _QtConsole: https://ipython.org/ipython-doc/3/interactive/qtconsole.html
.. _IPython: http://ipython.org/ipython-doc/3/interactive/

