"""
---------------------------------------------------------------------

Module providing tools to write results, grids, data to output files.

Requires Python packages/modules:
  -  :mod:`matplotlib.pyplot`
  -  :mod:`json`


Imports :class:`.Core` class and functions from the :mod:`.useful` module.

---------------------------------------------------------------------

.. _matplotlib: https://matplotlib.org/
.. _pyplot: https://matplotlib.org/api/pyplot_api.html

"""

import numpy  as np
from matplotlib.pyplot import savefig, figure
import os
import json

from streamlines.core   import Core
from streamlines.useful import write_geotiff

__all__ = ['Save']

pdebug = print

class Save(Core):       
    """
    Save plots to files
    
    Args:
        TBD (TBD): 
    
    TBD

    Returns:
        TBD: 
        TBD
    """
    def __init__(self, state, imported_parameters, 
                 geodata, preprocess, trace, analysis, mapping, plot):
        """
        Args:
            TBD (TBD): 
        
        TBD
    
        Returns:
            TBD: 
            TBD
        """  
        super().__init__(state,imported_parameters)
        self.state      = state
        self.geodata    = geodata
        self.preprocess = preprocess
        self.trace      = trace
        self.analysis   = analysis
        self.mapping    = mapping
        self.plot       = plot
        
    def do(self,sl):
        """
        Save all Matplotlib plots, mapping grids
        
        Args:
            TBD (TBD): 
        
        TBD
    
        Returns:
            TBD: 
            TBD
        """
        self.print('\n**Write results to files begin**') 
        
        file_stem_list =[None]*3
        for idx, export_path in enumerate([self.geodata.export_analyses_path, 
                                           self.geodata.export_maps_path, 
                                           self.geodata.export_figs_path]):
            
            if not os.path.exists(os.path.join(*export_path)):
                self.print('Creating export directory "{}"'
                           .format(os.path.join(*export_path)))  
                try:
                    os.mkdir(os.path.join(*export_path))
                except:
                    raise OSError('Cannot create export directory "'
                                  + str(os.path.join(*export_path)) + '"')
            else:
                if not os.path.isdir(os.path.join(*export_path)):
                    err = '"'+os.path.join(*export_path) +'" is not a directory'
                    print("OS error: {0}".format(err))
                    raise OSError
                
            file_stem_list[idx] = os.path.realpath(os.path.join(*export_path,
                                                   self.state.parameters_file))

        # Save myriad analyses
        if self.do_save_analyses:
            self.save_analyses(sl, file_stem=file_stem_list[0])
        # Save mapping grids
        if self.do_save_maps:
            self.save_maps(file_stem=file_stem_list[1])
        # Save graphics
        if self.do_save_figs:
            self.save_figs(file_stem=file_stem_list[2])
        
        self.print('**Write results to files end**\n')  
        
    @staticmethod
    def is_jsonable(item):
        try:
            json.dumps(item)
            return True
        except:
            return False


    def save_analyses(self, sl, analysis_name=None, file_stem=None):  
        """
        Args:
            TBD (TBD): 
        
        TBD
    
        Returns:
            TBD: 
            TBD
        """  
#         from pprint   import pprint
        self.print('Saving analyses...') 
        if file_stem is None:
            file_stem = os.path.realpath(os.path.join(*self.geodata.export_analyses_path,
                                    self.state.parameters_file))
        file_name = file_stem.replace('.json','')+self.analyses_suffix+'.json'
            
        jsonable_list = []
        pdict = {}
        pdict.update({'save': {}})
        for top_item in sl.__dict__.items():
            top_obj = top_item[0]
            top_val = top_item[1]
            if hasattr(top_val, '__dict__'):
#                 if not hasattr(pdict,top_obj):
                pdict.update({top_obj : {}})
                for sub_item in top_val.__dict__.items():                        
                    sub_obj = sub_item[0]
                    sub_val = sub_item[1]
                    if self.is_jsonable(sub_val):
#                         pdebug('jsonable:',top_obj,sub_obj)
                        pdict[top_obj].update({sub_obj : sub_val})
                    elif isinstance(sub_val, (list,tuple) ):
#                         pdebug('list/tuple:',top_obj,sub_obj)
                        if isinstance(sub_val[0], np.ndarray ) \
                                and sub_val[0].size<=self.max_nparray_size:
                            pass
#                             pdict[top_obj].update({sub_obj : 
# #                              ['{}'.format(sub_val_arrray).replace('\n','')
#                              [list(sub_val_arrray)
#                               for sub_val_arrray in sub_val]})
                    elif isinstance(sub_val, np.ndarray ):
#                         pdebug('ndarray:',top_obj,sub_obj)
                        if sub_val.size<=self.max_nparray_size:
#                             print(list(sub_val))
                            if self.is_jsonable(list(sub_val)):
                                pdict[top_obj].update({sub_obj : 
                                                       list(sub_val)})
#                                                    '{}'.format(sub_val).replace('\n','')})
                    else:
#                         pdebug('other:',top_obj,sub_obj)
                        pass
#                         pdict[top_obj].update({sub_obj : '{}'.format(sub_val)})
            else:
                pdict['save'].update({top_obj: top_val})
#         pprint(pdict)
    
        with open(file_name,'w') as json_file:
            self.print('Writing to "{}"'.format(file_name))
            try:
                json.dump(pdict, json_file, sort_keys=True, indent=4)
            except:
                print('Failed to write analysis results JSON file')
                            
        self.print('...done') 
        
    def save_maps(self, fig_name=None, file_stem=None):  
        """
        Args:
            TBD (TBD): 
        
        TBD
    
        Returns:
            TBD: 
            TBD
        """  
        self.print('Saving maps...') 
        if file_stem is None:
            file_stem = os.path.realpath(os.path.join(*self.geodata.export_maps_path,
                                                       self.state.parameters_file))
        obj_list = [self.geodata,self.preprocess,self.trace,self.mapping]
        pad = self.geodata.pad_width
        nx  = self.geodata.roi_nx
        ny  = self.geodata.roi_ny
        nxp = nx+pad*2
        nyp = ny+pad*2
        pslice = np.index_exp[pad:-pad,pad:-pad]
        pdebug(type(pslice))
        format = 'tif'
        for obj in obj_list:
            for item in obj.__dict__:
                ref = getattr(obj,item)
                if type(ref) is np.ndarray: # or type(ref) is np.ma.core.MaskedArray:
                    array_shape = ref.shape
                    if array_shape[0]==nxp and array_shape[1]==nyp:
                        file_name = file_stem+'_'+item.replace('_array','')+'.'+format
                        print(file_name, array_shape)
                        if len(array_shape)==3:
                            npd = array_shape[2]
                        else:
                            npd = 1
                        if npd==1:
                            write_geotiff(file_stem, file_name, ref, nx,ny,npd, pslice,
                                          self.geodata)
        self.print('...done') 
        
    def save_figs(self, fig_name=None, file_stem=None, file_format_list=None):
        """
        Args:
            TBD (TBD): 
        
        TBD
    
        Returns:
            TBD: 
            TBD
        """  
        self.print('Saving figs...') 
        fig_items = self.plot.figs.items()
        if file_stem is None:
            file_stem = os.path.realpath(os.path.join(*self.geodata.export_figs_path,
                                                       self.state.parameters_file))
        if file_format_list is None:
            file_format_list = self.figs_format
        for fig_item in fig_items:
            if fig_name is not None and fig_item[0]!=fig_name:   
                continue
            pdebug(fig_item)
            fig_obj = fig_item[1]
            for format in file_format_list:
                file_name = file_stem+'_'+fig_item[0]+'.'+format
                self.print('Writing <{0}> to "{1}"'.format( fig_item[0],file_name ) )
                fig_obj.savefig(file_name,format=format,**self.figs_options)
        self.print('...done') 
