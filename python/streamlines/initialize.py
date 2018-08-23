"""
---------------------------------------------------------------------

Config to run :py:mod:`slm` in `IPython`_.

Sets up `IPython`_ environment if we're running :py:mod:`slm` in a `Jupyter notebook`_ or 
`Jupyter QtConsole`_. 

 - prepares Matplotlib to display inline and at a 'retina' resolution -- if this
   is not available, a benign error report is made and progress continues
 - enables automatic reloading of :py:mod:`slm` (in case the code has been modded) when 
   a notebook is re-run in-situ
 - enables piping of print/error output from the GPU OpenCL device -- useful for 
   monitoring progress of slow jobs, but prone to problems



---------------------------------------------------------------------

Requires `matplotlib`_ and `IPython`_.

Uses IPython extensions `autoreload`_ and `Wurlitzer`_.

The  `autoreload`_ extension forces the :py:mod:`slm` package to be reloaded on 
restart. This makes code modding and subsequent rerunning of a notebook
smooth and seamless. It is not needed for normal operation, and if unavailable processing 
continues regardless.

The `Wurlitzer`_  extension provides C-level stdout/stderr pipes in Python, which allows
:py:mod:`slm` to connect to pipes of stdout/stderr from OpenCL kernels, 
i.e., to get to read printf output and error reports from kernel instances, subject
to the vagaries of GPU-CPU piping. It is useful but not required for normal operation
of :py:mod:`slm`.

---------------------------------------------------------------------

.. _matplotlib: https://matplotlib.org/
.. _autoreload: https://ipython.org/ipython-doc/3/config/extensions/autoreload.html
.. _Wurlitzer: https://github.com/minrk/wurlitzer
.. _IPython: https://ipython.readthedocs.io/en/stable/
.. _Jupyter notebook: https://jupyter-notebook.readthedocs.io/en/stable/
.. _Jupyter QtConsole: https://qtconsole.readthedocs.io/en/stable/



"""

# Jupyter `%magic` commands `%load_ext`, `%aimport`, and `%autoreload` 
#  are needed here to force the notebook to reload the `streamline` module, 
#  and its constituent modules, as changes are made to it.
# Force module to reload

import matplotlib as mpl

try:
    get_ipython().magic("config InlineBackend.figure_format = 'retina'")
except NameError as error:
    print('Error trying to invoke get_ipython(), probably because not running IPython:', 
          error)
    pass
except:
    print('Possibly benign error trying to config Matplotlib backend')
    import traceback
    print(traceback.format_exc())
    pass
 
try:
    get_ipython().magic('matplotlib inline')
except NameError as error:
    print('Error trying to invoke get_ipython(), probably because not running IPython:', 
          error)
    pass
except:
    print('Possibly benign error trying to config Matplotlib backend')
    import traceback
    print(traceback.format_exc())
    pass
 
try:
    get_ipython().magic('load_ext autoreload')
    get_ipython().magic('autoreload 2')
    get_ipython().magic('aimport streamlines')
except NameError as error:
    print('Error trying to invoke get_ipython(), probably because not running IPython:', 
          error)
    pass
except:
    print('Possibly benign error trying to config autoreload')
    import traceback
    print(traceback.format_exc())
    pass
 
try:
    get_ipython().magic('load_ext wurlitzer')
except NameError as error:
    print('Error trying to invoke get_ipython(), probably because not running IPython:', 
          error)
    pass
except:
    print('Possibly benign error loading Python "wurlitzer" package,'
          +' used to pipe stdout from GPU back to Jupyter notebook')
#     import traceback
#     print(traceback.format_exc())
    pass
 
