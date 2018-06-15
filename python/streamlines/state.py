"""
Read/write/assign the data contained in the :class:`.Trajectories` object, which includes all the 
parameters used to control streamline computation and all the data arrays used to 
hold their results.



Todo: 
    Fix issues with HDF5 r/w. There remain issues with writing lists of arrays.
"""
import os
import sys
import numpy as np
import json
from pympler.asizeof import asizeof
from collections     import ChainMap
from pprint import pprint

from streamlines.core import Core
from streamlines.useful import neatly, true_size
pdebug = print

__all__ = ['State']

class State(Core):
    """
    Class providing a variety of read/write and assignment methods to initialize, 
    record and reload run state.
    """   
    def set_savestate_filename(self):
        """
        Set workflow state JSON filename.
        
        The filename is set according to whether the input parameters file is 
        a numbered '_savestateXX.json' file.
        If no, the input JSON filename has '_savestate0' appended.
        If yes, its iteration number XX is incremented.
        
        Attributes:
            state_filename (str): Name of destination JSON file to save workflow state, 
            with increment number suffix set appropriately.
        """
        base_filename_split = self.parameters_file.split('_savestate')
#         print(base_filename_split)
        try:
            save_iteration = int(base_filename_split[1])+1
        except:
            save_iteration = 0
        state_filename = base_filename_split[0]+'_savestate'+str(save_iteration)
#         if os.path.isfile(os.path.join(*self.export_path,state_filename+'.json')):
#             print(state_filename)
        self.state_filename = state_filename
    
    def inventorize_run_state(self):
        """
        Build dictionary of lists of workflow class instance variables, grouped 
        according to how they can be exported to files.
        """
        self.print('\n**Inventorize run state begin**')  
        self.inventory = {}
        for obj in self.obj_list:
            self.print(obj)
            obj.inventorize(self)
        self.print('**Inventorize run state end**\n')
     
    def get_dict_of_jsonables(self):
        """
        Collect lists of JSONable objects in hierarchical dictionary
        """        
        full_jsonable_export_list = []
        total_usage = 0
        for obj in self.obj_list:
            obj_name = obj.__module__.split('.')[1]
            inventory_item = self.inventory[obj_name]
#             print('\n',obj_name,inventory_item,'\n')
            
            jsonable_list = inventory_item['jsonable']
            jsonable_export_list = []
            for jsonable_item in self.inventory[obj_name]['jsonable']:
                if obj_name=='state' and jsonable_item=='inventory':
                    continue
#                 elif obj_name=='plot' and jsonable_item=='figs':
#                     continue
                jsonable_obj = getattr(obj,jsonable_item)
                jsonable_export_list += [{jsonable_item : jsonable_obj} ]
                total_usage += true_size(jsonable_obj)
            d = dict(ChainMap(*jsonable_export_list))
            full_jsonable_export_list += [{obj_name : d}]
        return dict(ChainMap(*full_jsonable_export_list)), total_usage
    
    def savez_dicts_of_nparrays(self, filestem):
        """
        Collect lists of savezable numpy arrays in hierarchical dictionary
        and save to compressed file (npz)
        """        
        trimmed_obj_list = [obj for obj in self.obj_list 
                            if obj.__module__.split('.')[1]
                            in ['preprocess','trace','analysis']]
        for obj in trimmed_obj_list:

            obj_name = obj.__module__.split('.')[1]
            filename = filestem+'_'+obj_name+'.npz'
            
            self.print('Saving "'+obj_name+'" state np arrays to: "'+filename+'"...', 
                      end='')
            
            nparray_export_list = []
            for nparray in self.inventory[obj_name]['nparray']:
                nparray_export_list += [{obj_name+'.'+nparray : 
                                         getattr(obj,nparray)} ]
            np.savez_compressed(filename, **dict(ChainMap(*nparray_export_list)))
            
            self.print('...done')

    def get_sizes_of_nparrays(self):
        """
        Collect lists of savezable numpy arrays in hierarchical dictionary
        and calculate their total memory usage
        """        
        trimmed_obj_list = [obj for obj in self.obj_list 
                            if obj.__module__.split('.')[1] in 
                            ['preprocess','trace','analysis']]
        for obj in trimmed_obj_list:

            obj_name = obj.__module__.split('.')[1]            
            total_usage = 0
            for nparray in self.inventory[obj_name]['nparray']:
                total_usage += true_size(getattr(obj,nparray))

        return total_usage
    
    def get_streamlines_dict(self, array_list):
        """
        Collect list of streamline numpy arrays 
        """   
        return dict(ChainMap(*[{str(idx):arr} 
                               for idx,arr in enumerate(array_list)]))
        
    def get_streamlines_sizes(self, array_list):
        """
        Collect list of streamline numpy arrays 
        """   
        return sum([true_size(arr) for arr in array_list])
        
    def save_state(self):
        """
        Save working state to a set of JSON and other files.
        """

        #################################################################
        self.print('\n**Save state begin**')  
        #################################################################
            
        if not os.path.exists(os.path.join(*self.export_path)):
            self.print('State directory doesn\'t exist: creating "%s"'
                      % os.path.join(*self.export_path))  
            try:
                os.mkdir(os.path.join(*self.export_path))
            except:
                raise OSError('Cannot create state directory "'
                              + str(os.path.join(*self.export_path)) + '"')
        else:
            if not os.path.isdir(os.path.join(*self.export_path)):
                err = '"'+os.path.join(*self.export_path) +'" is not a directory'
                print("OS error: {0}".format(err))
                raise OSError
        
    
        self.set_savestate_filename()
        filestem = os.path.realpath(os.path.join(*self.export_path,self.state_filename))
        
        # JSONables
        filename = filestem+'.json'
        self.print('Saving runtime state JSONable parameters to: "'+filename+'"...', 
                  end='')       
        dict_of_jsonable_dicts, total_jsonable_usage = self.get_dict_of_jsonables()
#         pdebug('\n\n\nAllegedly jsonable dict:', dict_of_jsonable_dicts)
#         pdebug('\n\n\nAllegedly jsonable dict:', json.dumps(dict_of_jsonable_dicts,indent=3))
        copy_dict_of_jsonable_dicts = dict_of_jsonable_dicts.copy()
        for jsonable_dict_tuple in copy_dict_of_jsonable_dicts.items():
            copy_jsonable_dict = jsonable_dict_tuple[1].copy()
#             pdebug('\nConverting dict?:', jsonable_dict[0],jsonable_dict[1],type(jsonable_dict[1]))
            for jsonable_item in copy_jsonable_dict.items():
                if type(jsonable_item[1])==np.float32 or type(jsonable_item[1])==np.float64:
                    conv_jsonable_item = float(jsonable_item[1])
                elif type(jsonable_item[1])==np.int8 or type(jsonable_item[1])==np.int16 \
                        or type(jsonable_item[1])==np.int32 or type(jsonable_item[1])==np.int64 \
                        or type(jsonable_item[1])==np.uint8 or type(jsonable_item[1])==np.uint16 \
                        or type(jsonable_item[1])==np.uint32 or type(jsonable_item[1])==np.uint64:
                        conv_jsonable_item = int(jsonable_item[1])
                elif type(jsonable_item[1])==np.bool8:
                    conv_jsonable_item = bool(jsonable_item[1])
                else:
                    conv_jsonable_item = jsonable_item[1]
#                 pdebug('\nConverting:', jsonable_item,type(jsonable_item[1]), '->',conv_jsonable_item,type(conv_jsonable_item))
                jsonable_dict_tuple[1][jsonable_item[0]] = conv_jsonable_item
#                 pdebug(jsonable_dict_tuple)
#             dict_of_jsonable_dicts[jsonable_dict_tuple[0]] = jsonable_dict_tuple[1]
            
#         pdebug(copy_dict_of_jsonable_dicts)
        with open(filename,'w') as fp:
            json.dump(copy_dict_of_jsonable_dicts, fp, sort_keys=True, indent=4)
        self.print('...done')

        # Numpy arrays
        self.savez_dicts_of_nparrays(filestem)
        total_nparray_usage = self.get_sizes_of_nparrays()

        # Downstreamline and upstreamline lists of arrays
        total_streamlines_usage = 0
        for up_or_down_str,array_list in [['downstreamline',
                                            self.trace.streamline_arrays_list[0]],
                                          ['upstreamline',
                                            self.trace.streamline_arrays_list[1]]]:
            filename = filestem+'_'+up_or_down_str
            self.print('Saving '+up_or_down_str+'lines to: "'+filename+'.npz'+'"...', 
                      end='')
            streamlines_dict = self.get_streamlines_dict(array_list)
            total_streamlines_usage += self.get_streamlines_sizes(array_list)
            np.savez_compressed(filename, **streamlines_dict)
            del(streamlines_dict)
            self.print('...done')

        #################################################################
        self.print('Total JSONable memory usage:', 
                  neatly(total_jsonable_usage))
        self.print('Total numpy arrays memory usage (exc streamlines):', 
                  neatly(total_nparray_usage))
        self.print('Total streamline arrays memory usage:', 
                  neatly(total_streamlines_usage))
        self.print('**Save state end**\n')
        #################################################################

#     def read_state(self,filename):
#         """
#         Read archived run state from a group of archive files.
#         """
#         if self.do_rw_savez:
#             try:
#                 self.read_savez(filename)
#             except:
#                 raise ValueError('Cannot open savez file:', filename+'.npz')
#         if self.do_rw_hdf5:
#             try:
#                 self.read_hdf5(filename)
#             except:
#                 raise ValueError('Cannot open HDF5 file:', filename+'.h5')

    def write_hdf5(self, filename, nparray_list, nparraylist_list):
        """
        TBD
        """
        nparray_dict = {}
        for item in nparray_list:
            value = getattr(self, item)
            nparray_dict.update({item:value})
        with h5py.File(filename+'.h5','w') as hf:
            group = hf.create_group('nparrays')
#             print('writing to',filename)
            self.print('Writing to HDF5 group "nparrays":')
#             group.create_dataset('dtm_array', data=self.dtm_array)
#             group.create_dataset('roi_array', data=self.roi_array)
            for item in nparray_list:

                if item=='dtm_array':
                    if self.noisy:
                        print('\nSkip writing of DTM since we can fetch from original')
                    continue
                self.print(item, end=' ')
                if 'array_list' not in item:
#                     if self.noisy:
#                         print(item)
#                     if getattr(self, item).size==1 and getattr(self, item)==None: 
#                         continue
                    group.create_dataset(item, data=getattr(self, item))
#                                          ,dtype=getattr(self, item).dtype
#                                          ,compression='gzip')
                else:
                    subgroup = group.create_group(item)
                    for idx,array in enumerate(getattr(self, item)):
                        subgroup.create_dataset(str(idx), data=array)
#                                          ,dtype=getattr(self, item).dtype
#                                          ,compression='gzip')
            self.print()
                
    def read_hdf5(self, filename):
        """
        TBD
        """
        with h5py.File(filename+'.h5', 'r') as hf:
            print('Trying to read HDF5 file "%s"' % (filename+'.h5') )
            arrays = hf[filename][:]
        if self.noisy:
            print(arrays)
#         for item in arrays:
#             setattr(self, item,arrays[item])
#         if self.noisy:
#             print('Arrays read from hdf5:')
#             [print(npz_array) for npz_array in arrays.files]
#             print('')
        del arrays
        
    def add_active_mask(self, mask_item):
        """
        TBD
        """ 
        # Don't try to add if already there
        if list(mask_item.keys())[0] not in self.active_masks_dict.keys():
            self.active_masks_dict.update(mask_item)
        
    def remove_active_mask(self, mask_name):
        """
        TBD
        """ 
        # Don't try to remove if not there
        if mask_name in self.active_masks_dict.keys():
            self.active_masks_dict.pop(mask_name)
        
    def reset_active_masks(self):
        """
        TBD
        """ 
        # Clear all but the most basic masks
        masks_keep_list = ['dtm', 'basin', 'uv']
        # Rebuild dict since in-situ deletion in list comprehension doesn't work
        self.active_masks_dict \
            = {k: self.active_masks_dict[k] for k in self.active_masks_dict 
                                            if  k in masks_keep_list}
        
    def merge_active_masks(self):
        """
        TBD
        """ 
        # Create a mask from a blend of all those active
        for idx, mask_array in enumerate(self.active_masks_dict.values()):
            if idx==0:
                active_mask_array = mask_array.copy()
            else:
                active_mask_array |= mask_array
        return active_mask_array
        
                