"""
Map and graph plotting
"""

import numpy  as np
from matplotlib.pyplot import savefig, figure
import os

from streamlines.core   import Core
from streamlines.useful import write_geotiff

__all__ = ['Export']

pdebug = print

class Export(Core):       
    """
    Export plots to files
    """
    def __init__(self, state, imported_parameters, 
                 geodata, preprocess, trace, analysis, mapping, plot):
        """
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
        
    def do(self):
        """
        Export all Matplotlib plots, mapping grids
        """
        self.print('\n**Export results to files begin**') 
            
        if not os.path.exists(os.path.join(*self.export_path)):
            self.print('Creating export directory "{}"'
                       .format(os.path.join(*self.export_path)))  
            try:
                os.mkdir(os.path.join(*self.export_path))
            except:
                raise OSError('Cannot create export directory "'
                              + str(os.path.join(*self.export_path)) + '"')
        else:
            if not os.path.isdir(os.path.join(*self.export_path)):
                err = '"'+os.path.join(*self.export_path) +'" is not a directory'
                print("OS error: {0}".format(err))
                raise OSError
            
        file_stem = os.path.realpath(os.path.join(*self.export_path,
                                                 self.state.parameters_file))
        
        # Export mapping grids
        self.save_maps(file_stem=file_stem)
        # Export graphics
        self.save_figs(file_stem=file_stem)
        
        self.print('**Export results to files end**\n')  
        
    def save_maps(self, fig_name=None, file_stem=None): 
        self.print('Writing maps...') 
        if file_stem is None:
            file_stem = os.path.realpath(os.path.join(*self.export_path,
                                                       self.state.parameters_file))
        obj_list = [self.geodata,self.preprocess,self.trace,self.mapping]
        pad = self.geodata.pad_width
        nx  = self.geodata.roi_nx
        ny  = self.geodata.roi_ny
        nxp = nx+pad*2
        nyp = ny+pad*2
        pslice = np.index_exp[pad:-pad,pad:-pad]
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
        
    def save_figs(self, fig_name=None, file_stem=None):
        self.print('Writing figs...') 
        fig_items = self.plot.figs.items()
        if file_stem is None:
            file_stem = os.path.realpath(os.path.join(*self.export_path,
                                                       self.state.parameters_file))
        for fig_item in fig_items:
            if fig_name is not None and fig_item[0]!=fig_name:   
                continue
            pdebug(fig_item)
            fig_obj = fig_item[1]
            for format in self.figs_format:
                file_name = file_stem+'_'+fig_item[0]+'.'+format
                self.print('Writing <{0}> to "{1}"'.format( fig_item[0],file_name ) )
                fig_obj.savefig(file_name,format=format,**self.figs_options)
        self.print('...done') 
