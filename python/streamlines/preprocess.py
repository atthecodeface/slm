"""
DTM preprocessing steps. 

These include: computing the topographic gradient vector field, fixing loops 
and blockages, and computing a streamline 'flow speed' field.

"""

from numba.decorators import njit
import numpy as np
from numpy.linalg import eigvals
from scipy.ndimage.filters import generic_filter
from os import environ
environ['PYTHONUNBUFFERED']='True'
environ['NUMBA_WARNINGS']='1'
environ['NUMBA_DEBUG_ARRAY_OPT_STATS']='1'

from streamlines.core import Core

pdebug= print

__all__ = ['Preprocess']

# Not numbarizable
def compute_topo_gradient_field(roi_array):
    """
    Differentiate ROI topo in x and y directions using simple numpy method.
    
    Args:
        roi_array (numpy.array):
        
    Returns:
        numpy.array,numpy.array: ROI topo x gradient, y gradient
    """
    return np.gradient(roi_array, axis=0), np.gradient(roi_array, axis=1)

# Not numbarizable
def pad_array(roi_array,pad_width):
    """
    Pad ROI-size array around grid edges, repeating edge values.
    """
    return np.pad(roi_array, (int(pad_width),int(pad_width)),'edge')
    
def pad_arrays(u_array,v_array,pad_width):
    """
    Pad gradient vector arrays around grid edges, repeating edge values.
    """
    return pad_array(u_array,pad_width), pad_array(v_array,pad_width)
    
# @njit(cache=False)
def set_velocity_field(roi_gradx_array,roi_grady_array):
    """
    Convert topo gradient vector field into equivalent 'flow velocity' field.
    """
    return -roi_gradx_array.copy(),-roi_grady_array.copy()

# @njit(cache=False)
def set_speed_field(u_array,v_array):
    """
    Convert 'flow velocity' field into equivalent 'flow speed' field.
    """
    speed_array = np.hypot(u_array,v_array)
    speed_array[(~np.isfinite(speed_array)) | (speed_array==0.0)]=1.0
    return speed_array


# @njit(cache=False)
def normalize_velocity_field(u_array,v_array,speed_array):
    """
    Normalize 'flow velocity' field into unit vector field.
    """
    return u_array/speed_array, v_array/speed_array

# @njit(cache=False)
def unnormalize_velocity_field(u_array,v_array,speed_array):
    """
    Return 'flow velocity' field to true vector magnitudes.
    """
    return u_array*speed_array, v_array*speed_array

# @njit(cache=False)
def compute_gradient_velocity_field(roi_gradx_array, roi_grady_array):
    """
    Compute normalized gradient velocity vector field from ROI topo grid.
    """
    u_array, v_array = set_velocity_field(roi_gradx_array,roi_grady_array)
    speed_array = set_speed_field(u_array,v_array)
    u_array, v_array = normalize_velocity_field(u_array,v_array,speed_array)
    return u_array, v_array, speed_array

@njit(cache=False)
def get_flow_vector(nn):
    """
    TBD
    """
    dx,dy = np.int8(0),np.int8(0)
    if   nn & 16:   #SW
        dx,dy = np.int8(-1),np.int8(-1)
    elif nn & 32:   #SE
        dx,dy = np.int8(+1),np.int8(-1)
    elif nn & 64:   #NW
        dx,dy = np.int8(-1),np.int8(+1)
    elif nn & 128:  #NE
        dx,dy = np.int8(+1),np.int8(+1) 
    elif nn & 1:   #W
        dx,dy = np.int8(-1),np.int8( 0)
    elif nn & 2:   #E
        dx,dy = np.int8(+1),np.int8( 0)
    elif nn & 4:   #S
        dx,dy =  np.int8(0),np.int8(-1)
    elif nn & 8:   #N
        dx,dy =  np.int8(0),np.int8(+1)  
    return dx,dy

@njit(cache=False)
def check_has_loop(x,y,u,v):
    uv00 = np.array([u[x,y],v[x,y]],dtype=np.float32)
    uv10 = np.array([u[x+1,y],v[x+1,y]],dtype=np.float32)
    uv11 = np.array([u[x+1,y+1],v[x+1,y+1]],dtype=np.float32)
    uv01 = np.array([u[x,y+1],v[x,y+1]],dtype=np.float32)
    velocity = (uv00+uv10+uv11+uv01)/4.0
    speed = np.float32(np.sqrt(np.dot(velocity,velocity)))
    divergence = (
                np.dot(uv00,np.array([-1,-1],dtype=np.float32)) +
                np.dot(uv10,np.array([+1,-1],dtype=np.float32)) +
                np.dot(uv11,np.array([+1,+1],dtype=np.float32)) +
                np.dot(uv01,np.array([-1,+1],dtype=np.float32))
                )
    curl = (
                np.dot(uv00,np.array([+1,-1],dtype=np.float32)) +
                np.dot(uv10,np.array([+1,+1],dtype=np.float32)) +
                np.dot(uv11,np.array([-1,+1],dtype=np.float32)) +
                np.dot(uv01,np.array([-1,-1],dtype=np.float32))
            )
    return speed,divergence,curl

@njit(cache=False)
def break_out_of_loop(pt, 
                      u_array,v_array,
                      roi_array,
                      roi_nx,roi_ny):
    """
    TBD
    """
    def _within_bounds(xi,yi):
        """Return True if point is a valid index of grid"""
        return xi>=0 and xi<=roi_nx-1 and yi>=0 and yi<=roi_ny-1
    def _well_within_bounds(xi,yi):
        """Return True if point is a valid index of grid"""
        return xi>=1 and xi<=roi_nx-2 and yi>=1 and yi<=roi_ny-2

    vec_list = [get_flow_vector(nn) for nn in [1,2,4,8,16,32,64,128]]
    h_min = np.finfo(np.float32).max # set to flt_max
    lowest_h_idx = np.uint32(0)
    for idx,vec in enumerate(vec_list):
        x,y = pt[0],pt[1]
        if _within_bounds(x+vec[0],y+vec[1]) and _well_within_bounds(x,y):
            if h_min>roi_array[x+vec[0],y+vec[1]]:
                h_min = roi_array[x+vec[0],y+vec[1]]
                lowest_h_idx = idx
        else:
             return
    vec = vec_list[ lowest_h_idx ]
    vec_len = np.hypot(vec[0],vec[1])
    u_array[x,y] = vec[0]/vec_len
    v_array[x,y] = vec[1]/vec_len

@njit(cache=False)
def find_and_fix_loops(roi_array,
                       u_array, v_array,
                       x_roi_n_pixel_centers,
                       y_roi_n_pixel_centers,
                       roi_nx, roi_ny,
                       vecsum_threshold,
                       divergence_threshold,
                       curl_threshold,
                       verbose):
    """
    TBD
    """
    if verbose:
        print('Finding and fixing loops...')
        
    n_loops = np.uint32(0);
    # HACK: this malloc needs to be reined in and controlled more precisely
    where_looped_array = np.zeros((roi_nx*roi_ny,2),dtype=np.uint32)
    for xi,x in enumerate(x_roi_n_pixel_centers[:-1]):
        for yi,y in enumerate(y_roi_n_pixel_centers[:-1]):
            vecsum,divergence,curl = check_has_loop(xi,yi,u_array,v_array)
            if (vecsum<vecsum_threshold 
                    and divergence<=divergence_threshold 
                        and np.abs(curl)>=curl_threshold):
                where_looped_array[n_loops+0] = [xi,yi]
                where_looped_array[n_loops+1] = [xi+1,yi]
                where_looped_array[n_loops+2] = [xi,yi+1]
                where_looped_array[n_loops+3] = [xi+1,yi+1]
                n_loops += 4
    for idx in range(n_loops):
        break_out_of_loop(where_looped_array[idx],
                          u_array,v_array,
                          roi_array,
                          roi_nx, roi_ny
                          )
    if verbose:
        print('...done')
    return where_looped_array[:n_loops], n_loops
        
@njit(cache=False)
def fix_blockages(where_blockages_array,
                  blockages_array,
                  where_blocked_neighbors_array,
                  blocked_neighbors_array,
                  u_array, v_array,
                  verbose):
    """
    TBD
    """
    if verbose:
        print('Fixing blockages...')
    if where_blockages_array is not None:
        for idx in range(where_blockages_array.shape[0]):
            x,y = where_blockages_array[idx]
            dx,dy = get_flow_vector(blockages_array[x,y])
            u_array[x,y] = np.float32(dx)/np.sqrt(2.0)
            v_array[x,y] = np.float32(dy)/np.sqrt(2.0)
        for idx in range(where_blocked_neighbors_array.shape[0]):
            x,y = where_blocked_neighbors_array[idx]
            dx,dy = get_flow_vector(blocked_neighbors_array[x,y])
            u_array[x,y] = np.float32(dx)/np.sqrt(2.0)
            v_array[x,y] = np.float32(dy)/np.sqrt(2.0)
        if verbose:
            print('...done')
    else:
        if verbose:
            print('...none to fix')
        
@njit(cache=False)
def has_one_diagonal_outflow(pixel_neighborhood):
    """
    Should actually say "has at least one diagonal"
    """
    # Fetch the current (central pixel) elevation
    h = pixel_neighborhood[4]
    nn = np.uint8(0)
    # Scan neighboring pixels and flag if lower than central
    if pixel_neighborhood[0]<h:  # SW
        nn += 16
    if pixel_neighborhood[1]<h:  # S
        nn += 4
    if pixel_neighborhood[2]<h:  # SE
        nn += 32
    if pixel_neighborhood[3]<h:  # W
        nn += 1
    if pixel_neighborhood[5]<h:  # E
        nn += 2
    if pixel_neighborhood[6]<h:  # NW
        nn += 64
    if pixel_neighborhood[7]<h:  # N
        nn += 8
    if pixel_neighborhood[8]<h:  # NE
        nn += 128
        # Check whether any of the outflows is cardinal
    if (nn & (16+32+64+128))==nn:
        # None of the outflows is cardinal
        return nn
    else:
        return 0

@njit(cache=False)
def upstream_of_diagonal_outflow(pixel_neighborhood):
    """
    TBD
    """
    nn = np.uint8(0)
    # Scan neighboring pixels
    if pixel_neighborhood[0]==16:  # SW => SW
        nn += 16
    if pixel_neighborhood[1]==16:  # S => SW
        nn += 4
    if pixel_neighborhood[1]==32:  # S => SE
        nn += 4
    if pixel_neighborhood[2]==32:  # SE => SE
        nn += 32
    if pixel_neighborhood[3]==16:  # W => SW
        nn += 1
    if pixel_neighborhood[3]==64:  # W => NW
        nn += 1
    if pixel_neighborhood[5]==128: # E => NE
        nn += 2
    if pixel_neighborhood[5]==32:  # E => SE
        nn += 2
    if pixel_neighborhood[6]==64:  # NW => NW
        nn += 64
    if pixel_neighborhood[7]==64:  # N => NW
        nn += 8
    if pixel_neighborhood[7]==128: # N => NE
        nn += 8
    if pixel_neighborhood[8]==128: # NE => NE
        nn += 128
    return nn

class Preprocess(Core):
    """
    Class providing set of methods to prepare raw DTM data for streamline tracing.
    
    Provides top-level methods to: (1) condition DTM grid for streamline computation
    by fixing blockages (single-diagonal-outflow pixels) and loops (divergence, curl
    and net vector magnitude exceeding trio of thresholds); (2) Compute topographic
    gradient vector field using either simple bilinear interpolation or using
    bivariate spline interpolator functions at each grid cell.
    
    Args:
        state (object):  TBD
        imported_parameters (dict):  TBD
        geodata (object):  TBD

    Attributes:
        workflow parameters (various):  TBD
    """   
    def __init__(self,state,imported_parameters,geodata):
        super().__init__(state,imported_parameters)  
        self.geodata = geodata

    def do(self):
        """
        TBD
        """
        self.print('\n**Preprocess begin**')  
        if self.state.do_condition:
            self.conditioned_gradient_vector_field()
        else:
            self.raw_gradient_vector_field()
        # POSSIBLE BUG - may want to leave flat pixels (zero u & v) unmasked
        self.mask_nan_uv()
        self.print('**Preprocess end**\n')  
  
    def conditioned_gradient_vector_field(self):
        """
        Compute topographic gradient vector field on a preconditioned DTM.
        
        The preconditioning steps are:
        
        1. Find blockages in gradient vector field 
        2. Calculate surface derivatives (gradient & 'curvature')
        3. Set gradient vector field magnitudes ('speeds')
        4. Find and fix loops in gradient vector field
        5. Fix blockages in gradient vector field 
        6. Set initial streamline points ('seeds')
        """
        self.print('Precondition gradient vector field by fixing loops & blockages')  
            
        do_fixes = True
        
        if do_fixes:
            self.find_blockages()    
            
        self.roi_gradx_array, self.roi_grady_array \
            = compute_topo_gradient_field(self.geodata.roi_array.astype(np.float32))
        u_array, v_array, raw_speed_array \
            = compute_gradient_velocity_field(self.roi_gradx_array, self.roi_grady_array)
        
        if do_fixes:
            self.where_looped_array, self.n_loops = \
                find_and_fix_loops(self.geodata.roi_array,
                                   u_array, v_array,
                                   self.geodata.x_roi_n_pixel_centers,
                                   self.geodata.y_roi_n_pixel_centers,
                                   self.geodata.roi_nx, self.geodata.roi_ny,
                                   self.vecsum_threshold,
                                   self.divergence_threshold,
                                   self.curl_threshold,
                                   self.state.verbose)
        if do_fixes:
            fix_blockages(self.where_blockages_array.astype(np.uint32),
                          self.blockages_array.astype(np.uint8),
                          self.where_blocked_neighbors_array.astype(np.uint32),
                          self.blocked_neighbors_array.astype(np.uint8),
                          u_array, v_array,
                          self.state.verbose)
         
        if self.do_normalize_speed:
            speed_array = set_speed_field(u_array,v_array)
            (u_array, v_array) \
                = normalize_velocity_field(u_array,v_array,speed_array)
        else:
            (u_array, v_array) \
                = unnormalize_velocity_field(u_array,v_array,raw_speed_array)
            
        self.uv_array = np.stack( pad_arrays(u_array,v_array,self.geodata.pad_width), 
                                  axis=2 ).copy().astype(dtype=np.float32)
        self.slope_array = np.rad2deg(np.arctan(pad_array(raw_speed_array,
                                                          self.geodata.pad_width)))\
                                   .astype(dtype=np.float32)
        self.aspect_array = np.rad2deg(np.arctan2(self.uv_array[:,:,1],
                                                  self.uv_array[:,:,0]))
    
    def mask_nan_uv(self):
        self.print('Mask out bad uv pixels...', end='')
        self.uv_mask_array = np.zeros_like(self.geodata.dtm_mask_array)
        self.uv_mask_array[  np.isnan(self.uv_array[:,:,0]) 
                           | np.isnan(self.uv_array[:,:,1]) ] = True
        self.state.add_active_mask({'uv_mask_array': self.uv_mask_array})
        self.uv_array[self.uv_mask_array] = [0.0,0.0]
        self.print('done')
        
    def raw_gradient_vector_field(self):
        """
        Compute topographic gradient vector field without preconditioning the DTM.
        """     
        self.print('Compute raw gradient vector field')  
        (self.roi_gradx_array,self.roi_grady_array) = derivatives(self.geodata.roi_array)
        u_array, v_array, speed_array \
            = compute_gradient_velocity_field(self.roi_gradx_array, self.roi_grady_array)
        (u_array, v_array) \
            = pad_arrays(u_array,v_array,self.geodata.pad_width)
        self.uv_array = np.stack( pad_arrays(u_array,v_array,self.geodata.pad_width), 
                                  axis=2 ).copy().astype(dtype=np.float32)

    def find_blockages(self):
        """
        TBD
        """
        self.print('Finding blockages...', end='')
        # Create a blank array - will flag where blockages lie
        self.blockages_array = np.zeros_like(self.geodata.roi_array.T, dtype=np.uint8)
        self.blocked_neighbors_array = np.zeros_like(self.geodata.roi_array.T,
                                                     dtype=np.uint8)
        # Use a fast filter technique to process whole DTM to find blockages
        #   i.e., to find all pixels whose diagonal neighbor is the only outflow pixel
        generic_filter(self.geodata.roi_array.T, 
                        has_one_diagonal_outflow, 
                        output=self.blockages_array,
                        size=3)
        generic_filter(self.blockages_array, 
                        upstream_of_diagonal_outflow, 
                        output=self.blocked_neighbors_array,
                        size=3)
        self.blockages_array = self.blockages_array.T
        self.blocked_neighbors_array = self.blocked_neighbors_array.T
        # Having mapped the blockages as a 'directions' array,
        #   generate a simple vector of their x,y locations
        self.where_blockages_array = np.where(self.blockages_array)
        self.where_blockages_array \
            = np.stack((self.where_blockages_array[0],
                        self.where_blockages_array[1])).T
        self.where_blocked_neighbors_array = np.where(self.blocked_neighbors_array)
        self.where_blocked_neighbors_array \
            = np.stack((self.where_blocked_neighbors_array[0],
                        self.where_blocked_neighbors_array[1])).T
        self.print('found {}'.format(self.where_blockages_array.shape[0]), end='')
        if self.state.noisy:
            if self.where_blockages_array.shape[1]!=0:
                self.print('\nBlockages at:')
                for x,y in self.where_blockages_array:
                    dx,dy = get_flow_vector(self.blockages_array[x,y])
                    self.print('  {0} @ {1} => {2}'.format(str([dx,dy]),
                                                           str([x,y]),
                                                           str([x+dx,y+dy])))
            else:
                self.print('none found...', end='')
        self.print('...done')

