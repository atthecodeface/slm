#!/usr/bin/env python3

"""
Carry out the streamline computation workflow.

This workflow proceeds as follows:

1) Parse command line -- in particular to specify the parameters file.

2) Create an instance of the :class:`.Streamlining` class, which acts   
   as a global container for all the data and methods used in the computation. 

3) TBD...

"""

#
# Run from command line using e.g., 
#   python ${PATH_TO_STREAMLINE_MODULE}/streamlines/postprocess.py -f IndianCreekTest0
# Basic package requirements:
#  itertools
#  os, sys, getopt
#  json
#
# Pip install package requirements (showing version used during development):
#  matplotlib (2.0.2)
#  numpy (1.13.0)
#  scipy (0.19.0)
#  scikit-image (0.13.0)
#  scikit-learn (0.18.1) inc. sklearn.externals.joblib
#  GDAL (2.1.3; version 2.2.0 fails on darwin)
#
# Optional package:
#   numba (0.36.2) - for acceleration
#
from argparse import ArgumentParser,ArgumentTypeError,ArgumentDefaultsHelpFormatter
from pprint import pprint
from streamlines.streamlining import Streamlining

__all__ = ['run','_parse_cmd_line_args','_str2bool']

pdebug = print

def run(**kwargs):
    """
    Main function to drive streamline computation workflow when 
    called either from command line or Jupyter notebook.
    
    Return:
        object:  instance of :class:`.Streamlining` class
    """

    # Create the streamline workflow object
    #
    # The first initialization step is to read the parameters file specified from the 
    # command line, placing its result in a dict. 
    # A set of workflow classes are then instantiated.
    # Each instantiation involves a parsing of the parameters dict to extract 
    # those parameters relevant to the corresponding workflow step.
    sl = Streamlining(**kwargs)
    
    if sl.state.do_reload_state:
        return sl
    if sl.state.do_geodata:
        sl.geodata.do()
    if sl.state.do_preprocess:
        sl.preprocess.do()
    if sl.state.do_trace:
        sl.trace.do()
    if sl.state.do_analysis:
        sl.analysis.do()
    if sl.state.do_mapping:
        sl.mapping.do()
    if sl.state.do_plot:
        sl.plot.do()
        try:
            sl.plot.show()
        except:
            raise ValueError('Cannot display graphs')
    if sl.state.do_save_state:
        sl.state.inventorize_run_state()
        sl.state.save_state()
    if sl.state.do_export:
        sl.export.do()
        
    return sl
             
def _str2bool(arg):
    """
    Convert string to boolean during command line argument parsing.
    
    Args:
        arg (str): command line boolean parameter as string
        
    Return:
        bool:  command line parameter

    """
    if arg.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif arg.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise ArgumentTypeError('Boolean cmd line argument expected')
    
def _parse_cmd_line_args():
    """
    Parse the command line arguments using :mod:`argparse`.
    The arguments are assumed to be passed via `_sys.argv[1:]`.

    Return:
        :obj:`argparse.Namespace`:  parsed command line arguments
    """
    usage = '''Execute DTM streamline computation'''
    parser = ArgumentParser(description=usage, 
                            formatter_class=ArgumentDefaultsHelpFormatter)
    
    
    parser.add_argument('-v','--verbose', dest='verbose', 
                        default=None, type=_str2bool, action="store",  
                        metavar='verbose_flag',
                        help='verbose mode')
    
    parser.add_argument('-f', '--file', dest='parameters_file',
                        default=None, type=str,  action="store",  
                        metavar='parameters_file',
                        help='import parameters file')
    
    parser.add_argument('-r','--reload', dest='do_reload_state', 
                        default=None, type=_str2bool, action="store",  
                        metavar='reload_state_flag',
                        help='reload previous runtime state from files')

    parser.add_argument('-g', '--geodata', dest='do_geodata',
                        default=None, type=_str2bool,  action="store", 
                        metavar='geodata_flag',
                        help='read geodata files (DTM, basins)')

    parser.add_argument('-e', '--preprocess', dest='do_preprocess',
                        default=None, type=_str2bool,  action="store", 
                        metavar='preprocess_flag',
                        help='peform preprocessing ' \
                            +'(optionally do conditioning; compute gradients)')
                        
    parser.add_argument('-c', '--condition', dest='do_condition',
                        default=None, type=_str2bool,  action="store",  
                        metavar='condition_flag',
                        help='condition DTM for best tracing (fix loops & blockages)')
    
    parser.add_argument('-t', '--trace', dest='do_trace',
                        default=None, type=_str2bool,  action="store",  
                        metavar='trace_flag',
                        help='perform streamline tracing')
    
    parser.add_argument('-a', '--analysis', dest='do_analysis',
                        default=None, type=_str2bool,  action="store", 
                        metavar='analysis_flag',
                        help='analyze streamline patterns, distributions')
    
    parser.add_argument('-m', '--mapping', dest='do_mapping',
                        default=None, type=_str2bool,  action="store", 
                        metavar='mapping flag',
                        help='map channels, midlines')
    
    parser.add_argument('-p', '--plot', dest='do_plot',
                        default=None, type=str,  action="store", 
                        metavar='maps/pdfs/all',
                        help='carry out all plotting set in parameters files')
    
    parser.add_argument('-s','--save', dest='do_save_state', 
                        default=None, type=_str2bool, action="store",  
                        metavar='save_state_flag',
                        help='save runtime state to files at completion')
    
    parser.add_argument('-x','--export', dest='do_export', 
                        default=None, type=_str2bool, action="store",  
                        metavar='export_flag',
                        help='export figures to files')
    
    args = parser.parse_args()
    return args
             
if __name__ == '__main__':
    kwargs = vars(_parse_cmd_line_args())
    run(**kwargs)
