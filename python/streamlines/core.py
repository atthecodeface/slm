"""
This :mod:`core <streamlines.core>` module provides a base initialization class 
that is inherited by each workflow class.
On instantiation of each such class, the base initialization parses the parameter 
dictionary loaded from the 'job' :py:mod:`JSON <json>` file specified at the command line,
extracts the set of parameters pertaining to the workflow class,
and assigns the values to the workflow class instance accordingly. 

The following workflow classes inherit
the :mod:`core <streamlines.core>` class:

- :class:`streamlines.geodata.Geodata`
- :class:`streamlines.preprocess.Preprocess`
- :class:`streamlines.trace.Trace`
- :class:`streamlines.analysis.Analysis`
- :class:`streamlines.analysis.Mapping`
- :class:`streamlines.plot.Plot`
- :class:`streamlines.plot.Export`
"""

from json import dumps
import numpy as np
import os
import sys
import pprint
import streamlines
pdebug = print

__all__ = ['Core']

class Core():
    """
    Class inherited by all workflow classes to provide a one-stop-shop 
    initialization that loads up their parameter sets as class instance attributes.
    """
    def __init__(self,state,imported_parameters):
        """
        Initialize the class instance by loading up parameters as attributes.

        Args:
            state (object): The :class:`State <.state.State>` class instance.
            imported_parameters (dict): The parameters dictionary loaded from 
                                         a :py:mod:`JSON <json>` file.

        Attributes:
            Class instance variables (various types): 
                Set according to the parameters sub-dict corresponding to the class name. 
        
        Example:
            When instantiating the :class:`Geodata <.geodata.Geodata>` class, 
            the ``geodata`` sub-dictionary is extracted from the imported parameters 
            dictionary, and its parameters are parsed out and set as attributes of the  
            :class:`Geodata <.geodata.Geodata>`: 
            e.g., ``[title: "Indian Creek"]`` becomes the variable ``Geodata.title`` 
            with the value ``"Indian Creek"``.
        """
        # Fetch path to the ``streamlines`` module so we can later find the CL code
        self.path = streamlines.__path__[0]
        self.cl_src_path = os.path.join(self.path,'..','..','opencl')
        if state is not None:
            self.state = state
            try:
                self.state.obj_list += [self]
            except:
                pass
        else:
            self.active_masks_dict = {}
            pdebug(self.active_masks_dict)
            
        workflow_class_name = self.__module__.split('.')[1]
        for item in imported_parameters[workflow_class_name].items():
            if '_path' in item[0]:
                # Replace env var in DTM data path (list) if need be
                try:
                    moditem =  [os.environ[seg.replace('$','')] if '$' in seg else seg
                                for seg in item[1] ]
                except:
                    error = '  Error trying to parse environment variable ' \
                             + [seg for seg in item[1] if '$' in seg][0] \
                             + ' - please set this variable in your shell rc'
                    raise ValueError(error)
            else:
                moditem = item[1]
            setattr(self, item[0], moditem)

    def inventorize(self,state):
        """
        Args:
             state (object): The :class:`State <.state.State>` class instance.

        Attributes:
            Workflow state inventory (dict): 
                The state inventory is updated with 
                a sub-dict added (replacing if necessary) corresponding to the class 
                instance calling this method. The sub-dict items are lists entitled
                *jsonable*, *nparray*, *list_nparrays*, *object* and *other*: each list
                is populated with class attributes whose type matches these items.
            
        Example: 
            If the :class:`Trace <.trace.Trace>` instance calls
            this method, the :class:`Trace <.state.State>` inventory dict
            will be assigned a sub-dict entitled *trace* with list items as above
            to record the state of the :class:`Trace <streamlines.trace.Trace>` object.
            
        """
        class_dict =  self.__dict__
        data_dict = {
            'jsonable' : [],
            'nparray' : [],
            'list_nparrays' : [],
            'object' : [],
            'other' : []
            } 
        for item in class_dict.items():
            if hasattr(item[1],'__dict__'):
                data_dict['object'] += [item[0]]
            elif isinstance(item[1],np.ndarray):
                data_dict['nparray'] += [item[0]]
            else:
                if type(item[1])==np.float32 or type(item[1])==np.float64:
                    conv_item = float(item[1])
                elif type(item[1])==np.int8 or type(item[1])==np.int16 \
                        or type(item[1])==np.int32 or type(item[1])==np.int64 \
                        or type(item[1])==np.uint8 or type(item[1])==np.uint16 \
                        or type(item[1])==np.uint32 or type(item[1])==np.uint64:
                        conv_item = int(item[1])
                elif type(item[1])==np.bool8:
                    conv_item = bool(item[1])
                else:
                    conv_item = item[1]
#                 pdebug('\nConverting:', item,type(item[1]), '->',conv_item,type(conv_item))
                try:
                    test_if_jsonable = dumps(conv_item)
                    data_dict['jsonable'] += [item[0]]
                except:
                    if item[0]=='figs' or isinstance(conv_item[0],np.ndarray):
                        data_dict['list_nparrays'] += [item[0]]
                    else:
                        data_dict['other'] += [item[0]]

        workflow_class_name = (self.__module__.split('.')[1])
        state.inventory.update({workflow_class_name : data_dict})

    def print(self, *args, **kwargs):
        if self.state.verbose:
            print(*args, **kwargs)

    def pprint(self, *args, **kwargs):
        if self.state.verbose:
            pprint.pprint(*args, **kwargs)

    def vprint(self, *args, **kwargs):
        if self.state.verbose:
            print(*args, **kwargs, flush=True)
            # Try to really force this line to print before the GPU prints anything
            sys.stdout.flush()
