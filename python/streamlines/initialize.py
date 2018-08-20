"""
---------------------------------------------------------------------

IPython config.

Sets up IPython environment if we're running in a Jupyter notebook (or an IPython console).

---------------------------------------------------------------------

Requires `matplotlib`_, `IPython`_, and IPython extensions 
`autoreload`_ and `Wurlitzer`_.

`Wurlitzer`_ provides C-level stdout/stderr pipes in Python, which allows
**slm** to connect to pipes of stdout/stderr from OpenCL kernels, 
i.e., to get printf output and error reports from kernel instances, subject
to the vagaries of GPU-CPU piping.

---------------------------------------------------------------------

.. _matplotlib: https://matplotlib.org/
.. _IPython: https://ipython.org/
.. _autoreload: https://ipython.org/ipython-doc/3/config/extensions/autoreload.html
.. _Wurlitzer: https://github.com/minrk/wurlitzer


"""

# Jupyter `%magic` commands `%load_ext`, `%aimport`, and `%autoreload` 
#  are needed here to force the notebook to reload the `streamline` module, 
#  and its constituent modules, as changes are made to it.
# Force module to reload

import matplotlib as mpl

try:
    get_ipython().magic('load_ext autoreload')
    get_ipython().magic('autoreload 2')
    get_ipython().magic('aimport streamlines')
except:
    pass
 
try:
    get_ipython().magic('load_ext wurlitzer')
except:
#     print('Possibly benign error loading Python "wurlitzer" package,'
#           +' used to pipe stdout from GPU back to Jupyter notebook')
#     import traceback
#     print(traceback.format_exc())
    pass
 
try:
    get_ipython().magic("config InlineBackend.figure_format = 'retina'")
    get_ipython().magic('matplotlib inline')
except:
#     print('Possibly benign error trying to config Matplotlib backend')
#     import traceback
#     print(traceback.format_exc())
    pass
