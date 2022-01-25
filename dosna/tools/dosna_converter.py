#!/usr/bin/env python
""" Tool to translate HDF5 files to DosNa"""

import numpy as np

import h5py
import json
from contextlib import contextmanager

_SHAPE = 'shape'
_DTYPE = 'dtype'
_NDIM = 'ndim'
_NBYTES = 'nbytes'
_FILLVALUE = 'fillvalue'
_CHUNK_SIZE = 'chunk_size'
_CHUNK_GRID = 'chunk_grid'
_IS_DATASET = 'is_dataset'
_DATASET_NAME = 'name'
_DATASET_VALUE = 'dataset_value'
_PATH_SPLIT = '/'
_ABSOLUTE_PATH = 'absolute_path'

_METADATA = 'metadata'
_ATTRS = 'attrs'
_LINKS = 'links'
_DATASETS = 'datasets'
@contextmanager
def hdf_file(hdf, *args, **kwargs):
    if isinstance(hdf, str):
        yield h5py.File(hdf, 'r',*args, **kwargs)
    else:
        yield hdf

class Dosnatohdf5(object):

    def __init__(self, connection):
        self._connection = connection

    def dosna2dict(self):

        def _recurse(group, links, dosnadict):
            for key, value in links.items():
                subgroup = group.get_group(value.target)
                dosnadict[key] = dict()
                dosnadict[key][_ATTRS] = subgroup.get_attrs()
                dosnadict[key] = _recurse(subgroup, subgroup.get_links(), dosnadict[key])
                if subgroup.get_datasets() != {}:
                    dosnadict[key][_DATASETS] = {}
                    for dset in subgroup.get_datasets():
                        dosnadict[key][_DATASETS][dset] = subgroup.get_dataset(dset)
            return dosnadict

        dosnadict = _recurse(self._connection.get_group("/"), self._connection.get_group("/").get_links(), {})

        return dosnadict

    def dosna2hdf(self, h5file):

        def _recurse(group, links, hdfobject):
            for key, value in links.items():
                subgroup = group.get_group(value.target)
                hdfgroup = hdfobject.create_group(key)
                for k, v in subgroup.get_attrs().items():
                    hdfgroup[key].attrs[k] = v
                _recurse(subgroup, subgroup.get_links(), hdfgroup[key])
                if subgroup.get_datasets() != {}:
                    for dset in subgroup.get_datasets():
                        dosna_dataset = subgroup.get_dataset(dset)
                        hdf_dataset = hdfobject.create_dataset(
                                    dosna_dataset.name,
                                    shape=dosna_dataset.shape,
                                    chunks=dosna_dataset.chunk_size,
                                    dtype=dosna_dataset.dtype,
                                )
                        if hdf_dataset.chunks is not None:
                            for s in hdf_dataset.iter_chunks():
                                hdf_dataset[s] = dosna_dataset[s]

        with h5py.File(h5file, 'w') as hdf:
            _recurse(self._connection.get_group("/"), self._connection.get_group("/").get_links(), hdf)
            return hdf

    def dosna2json(self, jsonfile):

        def _recurse(group, links, jsondict):
            for key, value in links.items():
                subgroup = group.get_group(value.target)
                jsondict[key] = dict()
                jsondict[key][_ATTRS] = subgroup.get_attrs()
                jsondict[key] = _recurse(subgroup, subgroup.get_links(), jsondict[key])
                if subgroup.get_datasets() != {}:
                    jsondict[key][_DATASETS] = {}
                    for dset in subgroup.get_datasets():
                        dataset = subgroup.get_dataset(dset)
                        jsondict[key][_DATASETS][dset] = {}
                        jsondict[key][_DATASETS][dset][_DATASET_NAME] = dataset.name
                        jsondict[key][_DATASETS][dset][_SHAPE] = dataset.shape
                        jsondict[key][_DATASETS][dset][_NDIM] = dataset.ndim
                        jsondict[key][_DATASETS][dset][_DTYPE] = dataset.dtype
                        jsondict[key][_DATASETS][dset][_FILLVALUE] = float(dataset.fillvalue)
                        jsondict[key][_DATASETS][dset][_CHUNK_SIZE] = dataset.chunk_size
                        jsondict[key][_DATASETS][dset][_CHUNK_GRID] = dataset.chunk_grid
                        jsondict[key][_DATASETS][dset][_IS_DATASET] = True
                        data = dataset[:]
                        jsondict[key][_DATASETS][dset][_DATASET_VALUE] = data.tolist()
            return jsondict

        jsondict = _recurse(self._connection.get_group("/"), self._connection.get_group("/").get_links(), {})
        def json_encoder(obj):
            if isinstance(obj, np.ndarray):
                object_list = obj.tolist()
                return [str(x) for x in object_list]
            if isinstance(obj, bytes):
                return str(obj)
            raise TypeError('Not serializable: ', type(obj))

        with open(jsonfile, 'w') as f:
            f.write(json.dumps(jsondict, default=json_encoder))

        return jsondict
