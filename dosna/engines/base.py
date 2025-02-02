#!/usr/bin/env python
"""Base classes for every engine"""


import logging

import numpy as np

log = logging.getLogger(__name__)


class BackendWrapper(object):

    instance = None

    def __init__(self, instance):
        self.instance = instance

    def __getattr__(self, attr):
        """
        Attributes/Functions that do not exist in the extended class
        are going to be passed to the instance being wrapped
        """
        return self.instance.__getattribute__(attr)

    def __enter__(self):
        self.instance.__enter__()
        return self

    def __exit__(self, *args):
        self.instance.__exit__(*args)


class EngineConnection(BackendWrapper):
    
    def create_group(self, name, attrs={}):
        group = self.instance.create_group(name, attrs)
        return group
    
    def get_group(self, name):
        """
        group = self.instance.get_group(name) # TODO
        return group
        """
        raise NotImplemented('get_group not implemented for this engine')

    def get_object(self, name):
        object = self.instance._get_group_object(name)
        return object
    
    def del_group(self, name):
        return self.instance.del_group(name)

    def create_dataset(self, name, shape=None, dtype=np.float32, fillvalue=0,
                       data=None, chunk_size=None):
        self.instance.create_dataset(name, shape, dtype, fillvalue,
                                     data, chunk_size)
        dataset = self.get_dataset(name)
        if data is not None:
            dataset.load(data)
        return dataset

    def get_dataset(self, name):
        # this is meant to wrap the dataset with the specific engine class
        raise NotImplementedError('`get_dataset` not implemented '
                                  'for this engine')

    def del_dataset(self, name):
        dataset = self.get_dataset(name)
        dataset.clear()
        self.instance.del_dataset(name)

    def __getitem__(self, object_name):
        return self.get_object(object_name)
    
class EngineGroup(BackendWrapper):
    
    def create_group(self, name, attrs={}):
        self.instance.create_group(name, attrs={})
        group = self.get_group(name)
        return group
        
    def get_group(self, name):
        #group = self.get_group(name)
        #return group
        raise NotImplemented('get_group not implemented for this engine')


    def create_dataset(self, name, shape=None, dtype=np.float32, fillvalue=0,
                       data=None, chunk_size=None):
        self.instance.create_dataset(name, shape, dtype, fillvalue,
                                     data, chunk_size)
        dataset = self.get_dataset(name)
        if data is not None:
            dataset.load(data)
        return dataset
    
    def get_dataset(self, name):
        raise NotImplemented('get_group not implemented for this engine')

    def del_dataset(self, name):
        dataset = self.get_dataset(name)
        if self.name == self.path_split:
            if dataset.name[:len(self.name)] != self.name:
                raise ParentDatasetError(self.name, name)
        elif dataset.name[:len(self.name)+1] != self.name + self.path_split:
            raise ParentDatasetError(self.name, name)
        dataset.clear()
        self.instance.del_dataset(name)

    def get_object(self, name):
        object = self.get_object(name)
        return object

    def __getitem__(self, name):
        return self.get_object(name)
    
class EngineLink(BackendWrapper):
    
    def get_source(self):
        return self.get_source()
    
    def get_target(self):
        return self.get_source()
    
    def get_name(self):
        return self.get_name()
    


class EngineDataset(BackendWrapper):

    def get_data(self, slices=None):
        raise NotImplementedError('`get_data` not implemented for this engine')

    def set_data(self, values, slices=None):
        raise NotImplementedError('`set_data` not implemented for this engine')

    def clear(self):
        raise NotImplementedError('`clear` not implemented for this engine')

    def delete(self):
        self.clear()
        self.instance.connection.del_dataset(self.name)

    def load(self, data):
        raise NotImplementedError('`load` not implemented for this engine')

    def map(self, func, output_name):
        raise NotImplementedError('`map` not implemented for this engine')

    def apply(self, func):
        raise NotImplementedError('`apply` not implemented for this engine')

    def clone(self, output_name):
        raise NotImplementedError('`clone` not implemented for this engine')

    def create_chunk(self, idx, data=None, slices=None):
        self.instance.create_chunk(idx, data, slices)
        return self.get_chunk(idx)

    def get_chunk(self, idx):
        raise NotImplementedError('`create_chunk` not implemented '
                                  'for this backend')

    def del_chunk(self, idx):
        # just for base completeness
        self.instance.del_chunk(idx)

    def __getitem__(self, slices):
        return self.get_data(slices)

    def __setitem__(self, slices, values):
        self.set_data(values, slices=slices)


class EngineDataChunk(BackendWrapper):
    pass


class ParentDatasetError(Exception):
    def __init__(self, parent, dataset):
        self.parent = parent
        self.dataset = dataset
        self.message = "Can not delete dataset as " + self.parent + " is not a parent to " + self.dataset
        super().__init__(self.message)