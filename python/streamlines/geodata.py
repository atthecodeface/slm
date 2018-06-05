"""
GeoTIFF data file handling to provide DTM grid and handle masking

Todo:
    For now, only GeoTIFF read and parsing operations are working. 
    Might be useful to get GeoTIFF write working.
"""

import os
import numpy as np
from scipy.ndimage.morphology import binary_dilation, generate_binary_structure
from osgeo import gdal
import json
# Needs GDAL. Had issues with 2.2.0 and had to do:
#    pip install gdal==2.1.3

pdebug=print

from streamlines.core import Core

__all__ = ['Geodata']

class Geodata(Core):
    """
    Class providing methods for reading GeoTIFF files for DTM and related 
    drainage basins layer, and a tool to parse basin indices into a 
    streamline mask layer.
        
    """
    def do(self):
        """
        Wrapper method to read DTM file and drainage basins file and 
        then to parse basin indices into a mask layer.

        Attributes:
            self.dtm_path (str): absolute path to DTM file (should really be a list)
        """
        self.print('\n**Geodata begin**', flush=True)  
        self.read_dtm_file()
        self.make_dtm_mask()
        if self.do_basin_masking:
            self.read_basins_file()
            self.make_basins_mask()
        self.print('**Geodata end**\n', flush=True)  

    def read_geotiff(self, path, filename):
        """
        Read GeoTIFF file.
        Assumes file contains a single-band 2d grid of equal x-y dimension pixels.

        Args:
            path (str): Path to folder containing GeoTIFF file to be read
            filename (str): GeoTIFF filename
        
        Raises:
            ValueError if file cannot be opened for reading
            
        Returns:
            numpy.ndarray, float: Numpy array of GeoTIFF grid, size (in meters) 
                                    of a grid pixel
        """          
        fullpath_filename = os.path.join(path,filename)
        if self.state.noisy:
            self.print(fullpath_filename)
        tiff=gdal.Open(fullpath_filename)
        if tiff is None:
            raise ValueError('Cannot open GeoTIFF file "%s" for reading' % fullpath_filename)
        geotransform = tiff.GetGeoTransform()
        x_easting_bottomleft  = geotransform[0]
        y_northing_bottomleft = geotransform[3]+geotransform[5]
        pixel_size = geotransform[1]
        self.print('DTM GeoTIFF coordinate "geotransform":',geotransform)
        if not np.isclose(pixel_size,geotransform[5]*(-1),rtol=1e-3):
            raise ValueError(
                'Pixel x=%g and y=%g dimensions not equal in "%s": cannot handle non-square pixels'
                             % (pixel_size,(-1)*geotransform[5],fullpath_filename) )
        return (tiff.GetRasterBand(1).ReadAsArray().copy().astype(np.float32), 
                pixel_size, geotransform)
        
    def read_dtm_file(self):
        """
        Read GeoTIFF-format DTM file into numpy array and parse out important metadata, 
        e.g., pixel grid dimensions and pixel size in meters [TBD!!]. 
        Extract region of interest (ROI) as specified in parameters 
        (default is to take entire grid). 
        In the process, flip array orientation (up-down).

        Attributes:
            self.roi_nx (int):  number of x pixels in ROI
            self.roi_ny (int):  number of y pixels in ROI
            self.roi_x_origin (numpy.float32): x coordinate of bottom-left pixel center
            self.roi_y_origin (numpy.float32): y coordinate of bottom-left pixel center
            self.pixel_size (float): size (in meters from GeoTIFF) of pixels 
                                     (assuming equant)
            self.roi_pixel_size (float): size (in meters from GeoTIFF) of pixels in ROI 
                                         (could be downsampled)
            self.roi_x_bounds (list): x bounds on ROI (in grid pixels, not coordinates)
            self.roi_y_bounds (list): y bounds on ROI (in grid pixels, not coordinates)
            self.dtm_array (numpy.ndarray float32): DTM topographic grid 
                                                    read from GeoTIFF file
            self.roi_array (numpy.ndarray float32): region of interest (ROI) of DTM grid
            self.x_roi_n_pixel_centers (numpy.ndarray float32): meshgrid vector of 
                                                    x coordinates of ROI pixel centers
            self.y_roi_n_pixel_centers (numpy.ndarray float32): meshgrid vector of 
                                                    y coordinates of ROI pixel centers

        """
        try:
            self.dtm_path
        except:
            self.dtm_path = os.path.dirname(os.path.realpath(
                os.path.join(os.path.realpath(self.state.parameters_path),
                             *self.data_path,self.dtm_file) 
                ))
        self.print('Reading DTM from GeoTIFF file "%s/%s"'
              % (self.dtm_path,self.dtm_file))
        self.dtm_array, self.pixel_size, self.geotransform\
             = self.read_geotiff(self.dtm_path,self.dtm_file)
        self.x_easting_bottomleft  = self.geotransform[0]
        self.y_northing_bottomleft = self.geotransform[3] \
                                      +self.geotransform[5]*self.dtm_array.shape[0]
        self.print('DTM size: {0} x {1} = {2:,} pixels'.format(
              self.dtm_array.shape[1],self.dtm_array.shape[0],
              self.dtm_array.shape[1]*self.dtm_array.shape[0]) )
        self.print('DTM pixel size: {0}m'.format(self.pixel_size))
        self.print('DTM origin:')
        self.print('  - bottom-left pixel center: [{0:0.2f}mE, {1:0.2f}mN]'
              .format(self.x_easting_bottomleft, self.y_northing_bottomleft))
        self.print('  - bottom-left pixel corner: [{0:0.2f}mE, {1:0.2f}mN]'
              .format(self.x_easting_bottomleft-self.pixel_size/2, 
                      self.y_northing_bottomleft-self.pixel_size/2))

        for self.no_data_value in self.no_data_values:
            self.dtm_array[self.dtm_array==self.no_data_value] = np.nan
            
        # Handle empty ROI bounds which imply full DTM
        try:
            if self.roi_x_bounds==[]:
                raise
        except:
            self.roi_x_bounds = [0,self.dtm_array.shape[1]]
        try:
            if self.roi_y_bounds==[]:
                raise
        except:
            self.roi_y_bounds = [0,self.dtm_array.shape[0]]
        # Trap ROI bounds error
        if np.any(np.array(self.roi_x_bounds)
                  !=np.clip(np.array(self.roi_x_bounds),0,self.dtm_array.shape[1])):
            msg = ("ROI out of bounds in x: "
                   +str(np.array(self.roi_x_bounds))+' > '+str(self.dtm_array.shape[1]))
            raise ValueError(msg)        
        if np.any(np.array(self.roi_y_bounds)
                  !=np.clip(np.array(self.roi_y_bounds),0,self.dtm_array.shape[0])):
            msg = ("ROI out of bounds in y: "
                   +str(np.array(self.roi_y_bounds))+' > '+str(self.dtm_array.shape[0]))
            raise ValueError(msg)    
        
        # Generate roi array and x,y index vectors now because we'll need them later
        self.roi_array = np.zeros((self.roi_x_bounds[1]-self.roi_x_bounds[0],
                                   self.roi_y_bounds[1]-self.roi_y_bounds[0]),
                                   dtype=np.float32)
        self.roi_array = (np.flipud(self.dtm_array)[
                                        self.roi_y_bounds[0]:self.roi_y_bounds[1],
                                        self.roi_x_bounds[0]:self.roi_x_bounds[1]]
                                        ).T.copy()
        # Remember that np.array range extractions exclude last cell 
        #   such that self.roi_x_bounds[1]-1 is last x cell index
        self.x_roi_n_pixel_centers = np.linspace(self.roi_x_bounds[0]+0.5,
                                                 self.roi_x_bounds[1]-0.5, 
                                          self.roi_array.shape[0], dtype=np.float32)
        self.y_roi_n_pixel_centers = np.linspace(self.roi_y_bounds[0]+0.5,
                                                 self.roi_y_bounds[1]-0.5, 
                                          self.roi_array.shape[1], dtype=np.float32)
        self.roi_nx = len(self.x_roi_n_pixel_centers)
        self.roi_ny = len(self.y_roi_n_pixel_centers)
        
        self.print('ROI pixel bounds: ', [[self.roi_x_bounds[0],self.roi_x_bounds[-1]-1], 
                                     [self.roi_y_bounds[0],self.roi_y_bounds[-1]-1]])
        self.print('ROI pixel grid: ',  self.roi_nx, 'x', self.roi_ny, 
              '= {:,} pixels'.format(self.roi_nx*self.roi_ny))

        ######################################################
        ## Extremely important: 
        ##     these x_origin, y_origin coordinates
        ##     will be +0.5*pixel_size offset from
        ##     the lower-left corner of the grid
        ##  i.e., this origin gives PIXEL CENTERS
        ######################################################
        self.roi_x_origin = self.x_roi_n_pixel_centers[0]
        self.roi_y_origin = self.y_roi_n_pixel_centers[0]
        # Allow for future downsampling of ROI
        self.roi_pixel_size = self.pixel_size
        
        roi_width  = self.x_roi_n_pixel_centers[-1]-self.x_roi_n_pixel_centers[0]
        roi_height = self.y_roi_n_pixel_centers[-1]-self.y_roi_n_pixel_centers[0]
        roi_dx = self.x_roi_n_pixel_centers[1]-self.x_roi_n_pixel_centers[0]
        roi_dy = self.y_roi_n_pixel_centers[1]-self.y_roi_n_pixel_centers[0]
        self.print('ROI pixel-edge boundaries (assuming pixel-as-area)')
        self.print('  - in pixel units: [x: {0}<=>{1}] , [y: {2}<=>{3}]'.format(
                    (self.roi_x_origin-roi_dx/2),
                    (self.roi_x_origin+roi_width+roi_dx/2),
                    (self.roi_y_origin-roi_dy/2),
                    (self.roi_y_origin+roi_height+roi_dy/2)))
        self.print('  - in meters:      [x: {0}<=>{1}] , [y: {2}<=>{3}]'.format(
                    (self.roi_x_origin-roi_dx/2)*self.pixel_size,
                    (self.roi_x_origin+roi_width+roi_dx/2)*self.pixel_size,
                    (self.roi_y_origin-roi_dy/2)*self.pixel_size,
                    (self.roi_y_origin+roi_height+roi_dy/2)*self.pixel_size))
        if self.h_min!='none':
            self.roi_array[np.isnan(self.roi_array)] = self.h_min
        
    def read_basins_file(self):
        """
        Read GeoTIFF file of drainage basin indexes computed on the DTM by e.g. GRASS.

        Attributes:
            self.basins_array (numpy.ndarray float32):
        """ 
        self.print('Reading basins from GeoTIFF file "%s/%s"'
              % (self.dtm_path,self.basins_file))
        basins_array, _, _ = self.read_geotiff(self.dtm_path,self.basins_file)
        # Check size
        if (self.dtm_array.shape[1]!=basins_array.shape[1] 
            or self.dtm_array.shape[0]!=basins_array.shape[0]):
            raise ValueError(
                'DTM grid and basins grid sizes do not match - DTM: %s, basins=%s' 
                             % (str(self.dtm_array.shape),str(basins_array)))
        # Match orientations                      
        self.basins_array = (np.flipud(basins_array)[
              self.roi_y_bounds[0]:self.roi_y_bounds[1],
              self.roi_x_bounds[0]:self.roi_x_bounds[1]]).T.copy()

    def make_dtm_mask(self):
        """
        TBD
        """ 
        mask_unpadded_array = np.zeros_like(self.roi_array,dtype=np.bool8)
        mask_unpadded_array[np.isnan(self.roi_array)] = True
        if self.h_min!='none':
            mask_unpadded_array[self.roi_array<=self.h_min] = True
        self.dtm_mask_array = np.pad(mask_unpadded_array,
                                     (int(self.pad_width), int(self.pad_width)), 
                                     'constant', constant_values=(True,True))
        self.add_active_mask(self.dtm_mask_array)

    def make_basins_mask(self):
        """
        Generate a boolean mask array from the drainage basin index layer 
        and a list of non-mask indexes.

        Attributes:
            self.basin_mask_array (numpy.ndarray bool):
            self.basin_fatmask_array (numpy.ndarray bool):
        """ 
        self.print('Mask out all but basin numbers {}'.format(str(self.basins)))
        # Generate boolean grid with True at unmasked basin pixels
        basin_mask_unpadded_array = np.zeros_like(self.basins_array,dtype=np.bool8)
        for basin in self.basins:
            basin_mask_unpadded_array[self.basins_array==basin] = True
        dilation_structure = generate_binary_structure(2, 2)
        basin_fatmask_unpadded_array = binary_dilation(basin_mask_unpadded_array, 
                                                   structure=dilation_structure, 
                                                   iterations=1)
        # True = masked out; False = data we want to see
        basin_mask_unpadded_array = np.invert(basin_mask_unpadded_array)    
        basin_fatmask_unpadded_array = np.invert(basin_fatmask_unpadded_array)
            
        self.basin_mask_array = np.pad(basin_mask_unpadded_array,
                                       (int(self.pad_width),int(self.pad_width)), 
                                       'constant', constant_values=(True,True))
        self.basin_fatmask_array = np.pad(basin_fatmask_unpadded_array, 
                                          (int(self.pad_width), int(self.pad_width)), 
                                          'constant', constant_values=(True,True))
        self.add_active_mask(self.basin_mask_array)

    def add_active_mask(self, mask_array):
        """
        TBD
        """ 
        try:
            self.active_masks += [mask_array]
        except:
            self.active_masks = [mask_array]
        
    def remove_active_mask(self, mask_array):
        """
        TBD
        """ 
        active_masks = [mask for mask in self.active_masks if mask is not mask_array]
        self.active_masks = active_masks
        
    def merge_active_masks(self):
        """
        TBD
        """ 
        mask_array = self.active_masks[0].copy()
        for active_mask in self.active_masks[1:]:
            mask_array |= active_mask
        return mask_array
        