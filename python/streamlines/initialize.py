"""
Jupyter notebook config
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
