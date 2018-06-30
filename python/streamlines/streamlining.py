"""
Perform slm analysis of topographic structure
"""

from json     import loads
from datetime import datetime
from dateutil import tz
import os
os.environ['PYTHONUNBUFFERED']='True'

from streamlines.core       import Core
from streamlines.parameters import read_json_file, import_parameters
from streamlines.state      import State
from streamlines.geodata    import Geodata
from streamlines.preprocess import Preprocess
from streamlines.trace      import Trace
from streamlines.analysis   import Analysis
from streamlines.mapping    import Mapping
from streamlines.plot       import Plot
from streamlines.export     import Export

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

    Attributes:
        parameters_file (str): Name of JSON parameters file 
                               (parsed from kwargs 'parameters_file').
        parameters_dir (str): Path to folder containing JSON parameters file 
                              (parsed from kwargs 'parameters_file').
    
    
        """  
    def __init__(self, **kwargs):
        """
        Initialize the principal 'streamlines' class instance, whose object
        will contain references to the each of the key class instances of 
        the streamlines workflow, e.g., geodata(), trace(), analysis()
        Each such subobject will contain: 
        
        (1) attributes corresponding to all the parameters pertinent 
            to that stage of the workflow, e.g. state.do_plot, trace.do_trace_upstream,
            parsed the parameters 'dictionary of dictionaries' file;
        (2) a reference to the inventorize() method, inherited from the Core() class,
            used to do the parameter parsing;
        (3) back-references to all the key class instances needed for its work,
            e.g., trace.geodata(), plot.state();
        (4) references to methods needed for its work,
            e.g., preprocess.find_blockages(), trace.do();
        (5) attributes and references to objects generated during its work, 
            notably data arrays containing results
        """
        
        #
        # Parse workflow parameters in JSON files and command line
        #
        if 'parameters_file' not in kwargs.keys():
            raise ValueError('Must specify a parameters JSON file')
        parameters_path, parameters_file  = os.path.split(kwargs['parameters_file'])
        # Remove trailing .json for now if there is one
        parameters_file = ''.join(parameters_file.split('.json',-1))
        # Look for the JSON file in several likely places
        if parameters_path=='':   
            # Try the current directory - in UNIX, the dir from which slm was invoked         
            possible_paths = ['.']
            # Try the slm "home" dir given in the SLM environment variable
            try:
                possible_paths += [os.path.join(os.environ['SLM'],'json')]
            except:
                pass
            # If we're running a Jupyter notebook in slmnb/, try the likely
            #   relative path to slm/json/
            guess = os.path.join('..','..','slm','json')
            if os.path.isdir(guess):
                possible_paths += [guess]
            for path in possible_paths:
                if os.path.isfile(os.path.realpath(
                            os.path.join(path, parameters_file+'.json'))):
                    parameters_path = path
                    break
            # If we still can't find the specified JSON file, bail
            if parameters_path=='':
                raise ValueError('Cannot find JSON parameters file in {}'
                                 .format(possible_paths))
            
        # Read in parameters and assign to the Trajectories class instance
        imported_parameters, slm_path, slmdata_path, slmnb_path \
            = import_parameters(parameters_path, parameters_file)
        if ( ('verbose' not in kwargs.keys() or kwargs['verbose'] is None and 
                  'verbose' in imported_parameters['state'].keys() 
                   and imported_parameters['state']['verbose']) 
             or ('verbose' in kwargs.keys() and kwargs['verbose'] is not None 
                   and kwargs['verbose'])):
            # dateutil seems like best way to insert local TZ info into naive datetime
            now = datetime.now().replace(tzinfo=tz.tzlocal())            
            print(now.strftime('\n%a %Y-%m-%d %H:%M:%S %Z%z'))
            print('\n**Initialization begin**') 
            print('Loaded JSON parameters file "{}"'
                  .format(os.path.realpath(os.path.join(parameters_path, 
                                                        parameters_file+'.json'))))
        # If the command line requires override of JSON-file parameters, make it happen
        try:
            override_parameters = kwargs['override_parameters']
        except:
            override_parameters = None
        if override_parameters is not None and override_parameters!='':
            # The override parameter string is itself JSON data 
            override_dict = loads(override_parameters)
            for item in override_dict.items():
                imported_parameters[item[0]].update(item[1])

        # Instantiate the workflow "state"
        self.state = State(None,imported_parameters)
        # Record the JSON parameters file path & name 
        self.state.parameters_path = parameters_path
        self.state.parameters_file = parameters_file

        # Parse command line args and assign to State attributes in most cases
        for item in kwargs.items():
            if item[0]=='do_plot':
                if item[1]=='0' or item[1]=='off' or item[1]=='false':
                    self.state.do_plot=False
                elif item[1]=='maps':
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
                    self.state.do_plot = item[1]
            elif item[1] is not None:
                setattr(self.state, item[0],item[1])
        # Used by State.inventorize_run_state() and other State methods
        self.state.obj_list=[self.state]
        
        # Try to fetch latest slm-related git repo information
        #   - notably the commit hash, author, date & time
        if self.state.do_git_info:
            # Avoid the need to have the Python git module installed
            #   by only importing if do_git_info is true
            import git
            for repo_name, repo_path in (('slm',slm_path),
                                         ('slmnb',slmnb_path),
                                         ('slmdata',slmdata_path)):
                try:
                    # Create a short-lived git repo class instance
                    repo = git.Repo(repo_path)
                    # Grab its summary - seems to be the fastest way to get git info
                    summary = repo.git.show('--summary').split('\n')
                    git_info = [  [summary[0]]+[summary[1].split(' <')[0]]+[summary[2]] 
                                + ([summary[4]] if summary[4]!='' else []) 
                                + ([summary[5]] if summary[5]!='' else []) ]
                    setattr(self.state,repo_name+'_gitinfo',git_info)
                    # Print git info if verbose mode is on
                    self.print('{} git:'.format(repo_name))
                    self.pprint(git_info)
                except:
                    pass
            
        # Instantiate slm workflow classes
        self.geodata    = Geodata(self.state,imported_parameters)
        self.preprocess = Preprocess(self.state,imported_parameters,self.geodata)
        self.trace      = Trace(self.state,imported_parameters,self.geodata,
                                self.preprocess)
        self.analysis   = Analysis(self.state,imported_parameters,self.geodata,
                                    self.trace)
        self.mapping    = Mapping(self.state,imported_parameters,
                                  self.geodata,self.preprocess,self.trace,self.analysis)
        self.plot       = Plot(self.state,imported_parameters, self.geodata,
                               self.preprocess, self.trace, self.analysis, self.mapping)
        # Hackish way to allow plotting from mapping
        self.mapping._augment(self.plot)
        self.export     = Export(self.state,imported_parameters,
                                 self.geodata, self.analysis, self.trace,
                                 self.mapping, self.plot)
        # Used by State.save_state()
        self.state.trace = self.trace
                             
        self.print('**Initialization end**\n') 
    
