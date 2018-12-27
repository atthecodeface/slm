#!/usr/bin/env python3

"""
---------------------------------------------------------------------

Carry out ``Streamlines`` mapping workflow.

This workflow proceeds as follows:
  - Parse command line -- in particular to specify the parameters file.
  - Create an instance ``sl`` of the :class:`.Streamlining` class, which acts   
    as a global container for all the data and methods used in the computation.
     
    This class and all other principal classes inherit the :class:`Core <.Core>` class.

---------------------------------------------------------------------

Requires Python packages/modules:
  -  :mod:`argparse`
  -  :mod:`pprint`

Imports  :class:`Streamlining <streamlines.streamlining.Streamlining>` class
from the :mod:`streamlining <streamlines.streamlining>`  module.

---------------------------------------------------------------------

.. _argparse: https://docs.python.org/3/library/argparse.html
.. _pprint: https://docs.python.org/3/library/pprint.html


"""

from argparse import ArgumentParser,ArgumentTypeError,ArgumentDefaultsHelpFormatter
from pprint   import pprint
from streamlines.streamlining import Streamlining

__all__ = ['run','_str2bool','_parse_cmd_line_args']

pdebug = print

def run(**kwargs):
    """
    Main function to drive slm analysis workflow when 
    called either from command line or Jupyter notebook.
    
    Return:
        obj:  instance of :class:`.Streamlining` class
    """

    # Create the slm workflow object
    #
    # The first initialization step is to read the parameters file specified from the 
    # command line, placing its result in a dict. 
    # A set of workflow classes are then instantiated.
    # Each instantiation involves a parsing of the parameters dict to extract 
    # those parameters relevant to the corresponding workflow step.
    sl = Streamlining(**kwargs)
    
    # Execute the slm workflow
    #   => geodata  - read DTM
    #    => preprocess - compute uv vector field
    #     => trace - integrate streamlines
    #       => mapping - map geomorphic structures & fields
    #        => plot - graphs & maps
    #         => save state (currently defunct)
    #          => export - write plots to files
    if sl.state.do_geodata:
        sl.geodata.do()
    if sl.state.do_preprocess:
        sl.preprocess.do()
    if sl.state.do_trace:
        sl.trace.do()
#     if sl.state.do_analysis:
#         sl.analysis.do()
    if sl.state.do_analysis:
        sl.analysis.do()
    if sl.state.do_mapping:
        sl.mapping.do()
    if sl.state.do_extra:
        sl.analysis.extra()
    if sl.state.do_plot:
        sl.plot.do()
    if sl.state.do_save:
        sl.save.do(sl)
    if sl.state.do_plot:
        if sl.state.do_display:
            try:
                sl.plot.show()
            except:
                raise ValueError('Cannot display graphs')
        
    return sl
             
def _str2bool(arg):
    """
    Convert string to boolean during command line argument parsing.
    
    Args:
        arg (str): command line boolean parameter as string
        
    Return:
        bool:  command line parameter

    """
    if arg.lower() in ('yes', 'true', 't', 'y', '1', 'on'):
        return True
    elif arg.lower() in ('no', 'false', 'f', 'n', '0', 'off'):
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
    
    parser.add_argument('-a', '--analysis', dest='do_analysis',
                        default=None, type=_str2bool,  action="store", 
                        metavar='analysis_flag',
                        help='analyze streamline patterns, distributions')
                        
    parser.add_argument('-c', '--condition', dest='do_condition',
                        default=None, type=_str2bool,  action="store",  
                        metavar='condition_flag',
                        help='condition DTM for best tracing (fix loops & blockages)')
                        
    parser.add_argument('-d', '--debug', dest='debug',
                        default=None, type=_str2bool,  action="store",  
                        metavar='debug_flag',
                        help='turn on OpenCL compiler -D DEBUG flag')

    parser.add_argument('-e', '--preprocess', dest='do_preprocess',
                        default=None, type=_str2bool,  action="store", 
                        metavar='preprocess_flag',
                        help='peform preprocessing ' \
                            +'(optionally do conditioning; compute gradients)')
    
    parser.add_argument('-f', '--file', dest='parameters_file',
                        default=None, type=str,  action="store",  
                        metavar='parameters_file',
                        help='import JSON parameters file')

    parser.add_argument('-g', '--geodata', dest='do_geodata',
                        default=None, type=_str2bool,  action="store", 
                        metavar='geodata_flag',
                        help='read geodata files (DTM, basins)')

    parser.add_argument('-i','--info', dest='do_git_info', 
                        default=None, type=_str2bool, action="store",  
                        metavar='git_info_flag',
                        help='report info for slm git repos')
    
    parser.add_argument('-j', '--json', dest='override_parameters',
                        default=None, type=str,  action="store", 
                        metavar='override_parameters',
                        help='JSON dict of override parameters & values')
    
    parser.add_argument('-m', '--mapping', dest='do_mapping',
                        default=None, type=_str2bool,  action="store", 
                        metavar='mapping_flag',
                        help='map channels, midlines')
    
    parser.add_argument('-p', '--plot', dest='do_plot',
                        default=None, type=str,  action="store", 
                        metavar='maps/pdfs/all',
                        help='carry out all plotting set in parameters files')
    
    parser.add_argument('-q', '--display', dest='do_display',
                        default=None, type=_str2bool,  action="store", 
                        metavar='display_flag',
                        help='display plots')
    
    parser.add_argument('-s','--save', dest='do_save', 
                        default=None, type=_str2bool, action="store",  
                        metavar='save_flag',
                        help='save state, figs, map grids')
    
    parser.add_argument('-t', '--trace', dest='do_trace',
                        default=None, type=_str2bool,  action="store",  
                        metavar='trace_flag',
                        help='perform streamline tracing')
    
    parser.add_argument('-v','--verbose', dest='verbose', 
                        default=None, type=_str2bool, action="store",  
                        metavar='verbose_flag',
                        help='verbose mode')
    
    parser.add_argument('-w','--workitems', dest='n_work_items', 
                        default=None, type=int, action="store",  
                        metavar='n_work_items',
                        help='number of OpenCL work items per workgroup')
    
    parser.add_argument('-x','--extra', dest='do_extra', 
                        default=None, type=_str2bool, action="store",  
                        metavar='extra_flag',
                        help='more analysis')
    
    args = parser.parse_args()
    return args
             
if __name__ == '__main__':
    kwargs = vars(_parse_cmd_line_args())
    run(**kwargs)
