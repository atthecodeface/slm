"""
Map and graph plotting
"""

from matplotlib.pyplot import savefig, figure
import os

from streamlines.core import Core

__all__ = ['Export']

pdebug = print

class Export(Core):       
    """
    Export plots to files
    """
    def __init__(self,state,imported_parameters,geodata,trace,analysis,mapping,plot):
        """
        TBD
        """
        super().__init__(state,imported_parameters)  
        self.state    = state
        self.geodata  = geodata
        self.trace    = trace
        self.analysis = analysis
        self.mapping  = mapping
        self.plot     = plot
        
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
        
        # Export graphics
        self.save_figs(file_stem=file_stem)
        # Export mapping grids
        self.save_maps(file_stem=file_stem)
        
        self.print('**Export results to files end**\n')  
        
    def save_maps(self, fig_name=None, file_stem=None): 
        pass
        
    def save_figs(self, fig_name=None, file_stem=None):
        fig_items = self.plot.figs.items()
        if file_stem is None:
            file_stem = os.path.realpath(os.path.join(*self.export_path,
                                                       self.state.parameters_file))
        for fig_item in fig_items:
            if fig_name is not None and fig_item[0]!=fig_name:   
                continue
            pdebug(fig_item)
            fig_obj = fig_item[1]
            for format in self.format:
                file_name = file_stem+'_'+fig_item[0]+'.'+format
                self.print('Exporting <{0}> to "{1}"'.format( fig_item[0],file_name ) )
                fig_obj.savefig(file_name,format=format,**self.options)
        