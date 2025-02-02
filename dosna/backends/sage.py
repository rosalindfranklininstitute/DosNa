#!/usr/bin/env python

import logging
import os

import numpy as np

from dosna.backends import Backend
from dosna.backends.base import (BackendConnection, BackendDataChunk,
                                 BackendDataset, ConnectionError,
                                 DatasetNotFoundError, IndexOutOfRangeError)
from dosna.util import dtype2str, shape2str, str2shape
from dosna.util.data import slices2shape

try:
    from dosna.support.pyclovis import Clovis
except ImportError:
    raise ImportError("PyClovis module not found, "
                      "you need to compile it, please run `make` in "
                      "dosna/support/pyclovis, also note that PyClovis "
                      "rely on the mero dynamic library")

log = logging.getLogger(__name__)


class SageConnection(BackendConnection):

    def __init__(self, name, conffile='sage.conf', prefix_conffile=False,
                 *args, **kwargs):
        super(SageConnection, self).__init__(name, *args, **kwargs)
        if prefix_conffile:
            import socket
            from mpi4py import MPI
            host = socket.gethostname()
            rank = MPI.COMM_WORLD.Get_rank()
            conffile_name = os.path.basename(conffile)
            conffile_dir = os.path.dirname(conffile)
            host_rank_conffile = os.path.join(conffile_dir,
                                              "{}.{}.{}".format(host, rank,
                                                                conffile_name))
            rank_conffile = os.path.join(conffile_dir,
                                         "{}.{}".format(rank, conffile_name))
            if os.access(host_rank_conffile, os.R_OK):
                conffile = host_rank_conffile
            elif os.access(rank_conffile, os.R_OK):
                conffile = rank_conffile
        if not os.access(conffile, os.R_OK):
            raise ConnectionError("Configuration file {} not found"
                                  .format(conffile))
        log.info('Sage configuration file used: %s', conffile)
        self.conffile = conffile
        self.clovis = Clovis(conffile=self.conffile)

    def connect(self):
        if self.connected:
            raise ConnectionError(
                'Connection {} is already open'.format(self.name))
        super(SageConnection, self).connect()
        self.clovis.connect()

    def disconnect(self):
        super(SageConnection, self).disconnect()
        self.clovis.disconnect()

    def create_dataset(self, name, shape=None, dtype=np.float32, fillvalue=0,
                       data=None, chunk_size=None):
        if not ((shape is not None and dtype is not None) or data is not None):
            raise Exception('Provide `shape` and `dtype` or `data`')
        if self.has_dataset(name):
            raise Exception('Dataset `%s` already exists' % name)

        if data is not None:
            shape = data.shape
            dtype = data.dtype

        if chunk_size is None:
            chunk_size = shape

        chunk_grid = (np.ceil(np.asarray(shape, float) / chunk_size))\
            .astype(int)

        log.debug('creating dataset %s with shape:%s chunk_size:%s '
                  'chunk_grid:%s', name, shape, chunk_size, chunk_grid)
        self.clovis.create_object_metadata(name)
        dataset_metadata = {
            "shape": shape2str(shape),
            "dtype": dtype2str(dtype),
            "fillvalue": repr(fillvalue),
            "chunk_grid": shape2str(chunk_grid),
            "chunk_size": shape2str(chunk_size)
        }
        self.clovis.set_object_metadata(name, dataset_metadata)
        dataset = SageDataset(self, name, shape, dtype, fillvalue,
                              chunk_grid, chunk_size)

        return dataset

    def has_dataset(self, name):
        return self.clovis.has_object_metadata(name)

    def get_dataset(self, name):
        if not self.has_dataset(name):
            raise DatasetNotFoundError('Dataset `%s` does not exist' % name)
        metadata = self.clovis.get_object_metadata(name)
        shape = str2shape(metadata['shape'])
        dtype = metadata['dtype']
        fillvalue = int(metadata['fillvalue'])
        chunk_grid = str2shape(metadata['chunk_grid'])
        chunk_size = str2shape(metadata['chunk_size'])
        dataset = SageDataset(self, name, shape, dtype, fillvalue,
                              chunk_grid, chunk_size)
        return dataset

    def del_dataset(self, name):
        self.clovis.delete_object_metadata(name)


class SageDataset(BackendDataset):
    def _idx2name(self, idx):
        # this should be an integer for this backend
        return np.ravel_multi_index(idx, self.chunk_grid)

    def create_chunk(self, idx, data=None, slices=None):
        if idx > self._idx_from_flat(self.total_chunks - 1):
            raise IndexOutOfRangeError(idx, self._idx_from_flat(self.total_chunks - 1))
        if self.has_chunk(idx):
            raise Exception('DataChunk `{}{}` already exists'.format(self.name,
                                                                    idx))
        name = self._idx2name(idx)
        dtype = self.dtype
        shape = self.chunk_size
        fillvalue = self.fillvalue
        log.debug('creating datachunk %s with idx:%s shape:%s', name, idx,
                  shape)
        self.connection.clovis.create_object_chunk(self.name, name)
        datachunk = SageDataChunk(self, idx, name, shape, dtype, fillvalue)
        if data is None:
            data = np.full(shape, fillvalue, dtype)
        datachunk.set_data(data, slices, fill_others=True)
        return datachunk

    def get_chunk(self, idx):
        if idx > self._idx_from_flat(self.total_chunks - 1):
            raise IndexOutOfRangeError(idx, self._idx_from_flat(self.total_chunks - 1))
        if self.has_chunk(idx):
            name = self._idx2name(idx)
            dtype = self.dtype
            shape = self.chunk_size
            fillvalue = self.fillvalue
            return SageDataChunk(self, idx, name, shape, dtype, fillvalue)
        return self.create_chunk(idx)

    def has_chunk(self, idx):
        return self.connection.clovis.has_object_chunk(self.name,
                                                       self._idx2name(idx))

    def del_chunk(self, idx):
        log.debug("deleting datachunk from %s with idx:%s", self.name, idx)
        try:
            self.connection.clovis.delete_object_chunk(self.name,
                                                       self._idx2name(idx))
        except KeyError:
            logging.debug('chunk %s with idx:%s was not found while deleting',
                          self.name, idx)
            return False

        return True


class SageDataChunk(BackendDataChunk):

    def set_data(self, values, slices=None, fill_others=False):
        if slices is None or slices2shape(slices) == self.shape:
            self.write(values.tobytes())
        else:
            if fill_others:
                cdata = np.full(self.shape, self.fillvalue, self.dtype)
            else:
                cdata = self.get_data()
            cdata[slices] = values
            self.write(cdata.tobytes())

    def get_data(self, slices=None):
        if slices is None:
            slices = slice(None)
        data = np.fromstring(self.read(), dtype=self.dtype, count=self.size)
        data.shape = self.shape
        return data[slices]

    def write(self, data):
        self.dataset.connection.clovis.write_object_chunk(self.dataset.name,
                                                          int(self.name),
                                                          data,
                                                          len(data))

    def read(self):
        return self.dataset.connection.clovis.read_object_chunk(
            self.dataset.name, self.name, self.byte_count)


_backend = Backend('sage', SageConnection, SageDataset, SageDataChunk)
