"""
Map and graph plotting
"""

from matplotlib.pyplot import savefig,figure
import os

from streamlines.core import Core

__all__ = ['Export']

class Export(Core):       
    """
    Export plots to files
    """
    def __init__(self,state,imported_parameters,plot):
        """
        TBD
        """
        super().__init__(state,imported_parameters)  
        self.state = state
        self.plot = plot
        
    def do(self):
        """
        Export all Matplotlib plots
        """
        self.print('\n**Export plots begin**', flush=True) 
            
        if not os.path.exists(os.path.join(*self.export_path)):
            if self.verbose:
                print('Export directory doesn\'t exist: creating "%s"'
                      % os.path.join(*self.export_path), flush=True)  
            try:
                os.mkdir(os.path.join(*self.export_path))
            except:
                raise OSError('Cannot create export directory "'
                              + str(os.path.join(*self.export_path)) + '"', 
                              flush=True)
        else:
            if not os.path.isdir(os.path.join(*self.export_path)):
                err = '"'+os.path.join(*self.export_path) +'" is not a directory'
                print("OS error: {0}".format(err))
                raise OSError
            
        filestem = os.path.realpath(os.path.join(*self.export_path,self.state.parameters_file))

        for fig_item in self.plot.figs.items():
            fig_obj = fig_item[1]
            for format in self.format:
                filename = filestem+'_'+fig_item[0]+'.'+format
                self.print('Exporting <{0}> to "{1}"'.format( fig_item[0],filename ) )
                fig_obj.savefig(filename,format=format,**self.options)
        self.print('**Export plots end**\n', flush=True)  
        