"""
Compute streamlines and their density grids.

Todo:
    Fix likely bug in parameters file path wrangling
"""

import os
from mixin import mixin
from json import dumps
from os import environ
environ['PYTHONUNBUFFERED']='True'

from streamlines.core import Core
from streamlines.parameters import read_json_file, import_parameters
from streamlines.state import State
from streamlines.geodata import Geodata
from streamlines.preprocess import Preprocess
from streamlines.trace import Trace
from streamlines.analysis import Analysis
from streamlines.mapping import Mapping
from streamlines.plot import Plot
from streamlines.export import Export

__all__ = ['Streamlining']

pdebug = print

class Streamlining(Core):
    """
    Class providing set of methods to compute streamline trajectories and 
    densities from raw DTM data.
    
    Provides top-level methods to: (1) prepare DTM grid for streamline computation
    by fixing blockages (single-diagonal-outflow pixels) and loops (divergence, curl
    and net vector magnitude exceeding trio of thresholds); (2) set 'seed' points aka
    start locations (sub-pixel positions) of all streamlines; (3) generate streamlines
    from all seed points either upstream or downstream, returning seed point locations
    if generated in-situ, and returning arrays of streamline points and their mean
    spacing; (4) generate all streamlines (up and downstream) and compute the overall
    mean streamline point spacing.
    
    Args:
        parameters_file (str): Name of JSON parameters file prefixed by full path.
        do_reload_state (bool): Flag whether to reload computation state from file(s).

    Attributes:
        parameters_file (str): Name of JSON parameters file 
                               (parsed from kwargs 'parameters_file').
        parameters_dir (str): Path to folder containing JSON parameters file 
                              (parsed from kwargs 'parameters_file').
    
    
        """      
    def __init__(self, do_reload_state=False, **kwargs):
        """
        Initialize the principal 'streamlines' class instance, whose object
        will contain references to the each of the key class instances of 
        the streamlines workflow, e.g., geodata(), trace(), analysis()
        Each such subobject will contain: 
        
        (1) attributes corresponding to all the parameters pertinent 
            to that stage of the workflow, e.g. state.do_plot, trace.do_trace_upstream,
            parsed the parameters 'dictionary of dictionaries' file;
        (2) a reference to the inventorize() method, inherited from the Core() class,
            used to do the parameters parsing;
        (3) back-references to all the key class instances needed for its work,
            e.g., trace.geodata(), plot.state();
        (4) references to methods needed for its work,
            e.g., preprocess.find_blockages(), trace.do();
        (5) attributes and references to objects generated during its work, 
            notably data arrays containing results, 
            e.g., geodata.roi_array, trace.streamline_arrays_list.
        """
        if 'parameters_file' not in kwargs.keys():
            raise ValueError('Must specify a parameters JSON file')
        # Remove trailing .json for now if there is one
        parameters_path, parameters_file  = os.path.split(kwargs['parameters_file'])
        if parameters_path=='':
            parameters_path='.'
        parameters_file = ''.join(parameters_file.split('.json',-1))
        
        # Read in parameters and assign to the Trajectories class instance
        imported_parameters = import_parameters(parameters_path, parameters_file)
            
        if (('verbose' not in kwargs.keys() and 
             'verbose' in imported_parameters['state'].keys() 
             and imported_parameters['state']['verbose']) 
             or ('verbose' in kwargs.keys()  and kwargs['verbose'])):
            print('\n**Initialization begin**') 
            
        self.state = State(None,imported_parameters)

        for item in kwargs.items():
            if item[0]=='do_plot':
                if item[1]=='maps':
                    self.state.do_plot=True
                    imported_parameters['plot']['do_plot_maps']=True
                    imported_parameters['plot']['do_plot_distributions']=False
                elif item[1]=='pdfs' or item[1]=='distributions':
                    self.state.do_plot=True
                    imported_parameters['plot']['do_plot_maps']=False
                    imported_parameters['plot']['do_plot_distributions']=True
                elif item[1]=='all' or item[1]=='1' \
                        or item[1]=='True' or item[1]=='true':
                    self.state.do_plot=True
                    imported_parameters['plot']['do_plot_maps']=True
                    imported_parameters['plot']['do_plot_distributions']=True
            elif item[1] is not None:
                setattr(self.state, item[0],item[1])
        self.state.obj_list=[self.state]


        self.state.parameters_path = parameters_path
        self.state.parameters_file = parameters_file
        self.geodata = Geodata(self.state,imported_parameters)
        self.preprocess = Preprocess(self.state,imported_parameters,self.geodata)
        self.trace = Trace(self.state,imported_parameters,self.geodata,self.preprocess)
        self.state.trace = self.trace
        self.analysis = Analysis(self.state,imported_parameters,self.geodata,self.trace)
        self.mapping = Mapping(self.state,imported_parameters,
                         self.geodata,self.preprocess,self.trace,self.analysis)
        self.plot = Plot(self.state,imported_parameters, self.geodata,self.preprocess,
                         self.trace,self.analysis,self.mapping)
        self.export = Export(self.state,imported_parameters,self.plot)
                             
        self.print('**Initialization end**\n') 
