#!/usr/bin/env python
"""Backend ceph uses a ceph cluster to store the dataset and chunks data"""

import logging

import numpy as np

import rados
from dosna.backends import Backend
from dosna.backends.base import (BackendConnection, BackendDataChunk,
                                 BackendDataset, BackendGroup, BackendLink, ConnectionError,
                                 DatasetNotFoundError)
from dosna.util import dtype2str, shape2str, str2shape, str2dict, dict2str
from dosna.util.data import slices2shape
_PATH_SPLIT = '/'
_SIGNATURE = "DosNa Dataset"
_SIGNATURE_GROUP = "DosNa Group"
_SIGNATURE_LINK =  "Dosna Link"
_ENCODING = "utf-8"
log = logging.getLogger(__name__)


class CephConnection(BackendConnection):
    """
    A Ceph Cluster that wraps LibRados.Cluster
    """

    def __init__(self, name, conffile='ceph.conf', timeout=5,
                 client_id=None, *args, **kwargs):
        super(CephConnection, self).__init__(name, *args, **kwargs)

        rados_options = {
            "conffile": conffile
        }
        if client_id is not None:
            client_name = "client.{}".format(client_id)
            rados_options["name"] = client_name

        self._cluster = rados.Rados(**rados_options)
        self._timeout = timeout
        self._ioctx = None
        self.root_group = None

    def connect(self):
        if self.connected:
            raise ConnectionError(
                'Connection {} is already open'.format(self.name))
        self._cluster.connect(timeout=self._timeout)
        self._ioctx = self._cluster.open_ioctx(self.name)
        super(CephConnection, self).connect()
        self.create_root_group()

    def create_root_group(self):
        self.ioctx.write(_PATH_SPLIT, _SIGNATURE_GROUP.encode(_ENCODING))
        self.ioctx.set_xattr(_PATH_SPLIT, "name", str(_PATH_SPLIT).encode(_ENCODING))
        self.ioctx.set_xattr(_PATH_SPLIT, "attrs", str({}).encode(_ENCODING))
        self.ioctx.set_xattr(_PATH_SPLIT, "absolute_path", str(_PATH_SPLIT).encode(_ENCODING))
        self.ioctx.set_xattr(_PATH_SPLIT, "datasets", str({}).encode(_ENCODING))
        self.ioctx.set_xattr(_PATH_SPLIT, "links", str({}).encode(_ENCODING))
        self.root_group = CephGroup(self, "/")

    def disconnect(self):
        if self.connected:
            self.ioctx.close()
            self._cluster.shutdown()
            super(CephConnection, self).disconnect()

    @property
    def ioctx(self):
        return self._ioctx

    def create_group(self, path, attrs={}):
        if path == _PATH_SPLIT:
            raise Exception('Group: ', path, 'already exists')
        else:
            return self.root_group.create_group(path, attrs)

    def get_group(self):
        raise NotImplementedError('`get_group` not implemented '                           'for this backend')

    def has_group(self, name):
        try:
            valid = self.ioctx.stat(name)[0] == len(_SIGNATURE_GROUP.encode(_ENCODING)) and \
                    self.ioctx.read(name) == _SIGNATURE_GROUP.encode(_ENCODING)
        except rados.ObjectNotFound:
            return False
        return valid

    def del_group(self):
        raise NotImplementedError('`del_group` not implemented '
                                  'for this backend')

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
        self.ioctx.write(name, _SIGNATURE.encode(_ENCODING))
        self.ioctx.set_xattr(name, 'shape', shape2str(shape).encode(_ENCODING))
        self.ioctx.set_xattr(name, 'dtype', dtype2str(dtype).encode(_ENCODING))
        self.ioctx.set_xattr(name, 'fillvalue', repr(fillvalue).encode(_ENCODING))
        self.ioctx.set_xattr(name, 'chunk_grid', shape2str(chunk_grid).encode(_ENCODING))
        self.ioctx.set_xattr(name, 'chunk_size', shape2str(chunk_size).encode(_ENCODING))
        dataset = CephDataset(self, name, shape, dtype, fillvalue,
                              chunk_grid, chunk_size)

        return dataset

    def get_dataset(self, name):
        if not self.has_dataset(name):
            raise DatasetNotFoundError('Dataset `%s` does not exist' % name)
        shape = str2shape(self.ioctx.get_xattr(name, 'shape').decode())
        dtype = self.ioctx.get_xattr(name, 'dtype').decode()
        fillvalue = int(self.ioctx.get_xattr(name, 'fillvalue').decode())
        chunk_grid = str2shape(self.ioctx.get_xattr(name, 'chunk_grid').decode())
        chunk_size = str2shape(self.ioctx.get_xattr(name, 'chunk_size').decode())
        dataset = CephDataset(self, name, shape, dtype, fillvalue,
                              chunk_grid, chunk_size)
        return dataset

    def has_dataset(self, name):
        try:
            valid = self.ioctx.stat(name)[0] == len(_SIGNATURE.encode(_ENCODING)) and \
                self.ioctx.read(name) == _SIGNATURE.encode(_ENCODING)
        except rados.ObjectNotFound:
            return False
        return valid

    def del_dataset(self, name):
        log.debug("Removing dataset %s", name)
        if self.has_dataset(name):
            self.ioctx.remove_object(name)
        else:
            raise DatasetNotFoundError(
                'Dataset `{}` does not exist'.format(name))

class CephLink(BackendLink):
    def __init__(self, source, target, name):
        super(CephLink, self).__init__(source, target, name)

    def get_source(self):
        return self.source

    def get_target(self):
        return self.target

    def get_name(self):
        return self.name


class CephGroup(BackendGroup):
    def __init__(self, parent, name, attrs, path_split="/", *args, **kwargs):
        super(CephGroup, self).__init__(parent, name, attrs)

    def create_group(self, parent, name, attrs={}):
        raise NotImplementedError('`create_group` not implemented '
                                  'for this backend')

    def get_group(self):
        raise NotImplementedError('`get_group` not implemented '
                                  'for this backend')


    def has_group(self, name):
        try:
            valid = self.ioctx.stat(name)[0] == len(_SIGNATURE_GROUP.encode(_ENCODING)) and \
                    self.ioctx.read(name) == _SIGNATURE_GROUP.encode(_ENCODING)
        except rados.ObjectNotFound:
            return False
        return valid


    def del_group(self):
        raise NotImplementedError('`del_group` not implemented '
                                  'for this backend')

    def create_dataset(self, name, shape=None, dtype=np.float32, fillvalue=0,
                       data=None, chunk_size=None):
        raise NotImplementedError('`create_dataset` not implemented '
                                  'for this backend')

    def get_dataset(self, name):
        raise NotImplementedError('`get_dataset` not implemented '
                                  'for this backend')

    def has_dataset(self, name):
        raise NotImplementedError('`has_dataset` not implemented '
                                  'for this backend')

    def del_dataset(self, name):
        """Remove dataset metadata only"""
        raise NotImplementedError('`del_dataset` not implemented '
                                  'for this backend')


class CephDataset(BackendDataset):
    """
    CephDataset wraps an instance of Rados.Object
    """

    @property
    def ioctx(self):
        return self.connection.ioctx

    def _idx2name(self, idx):
        return '{}/{}'.format(self.name, '.'.join(map(str, idx)))

    def create_chunk(self, idx, data=None, slices=None):
        if self.has_chunk(idx):
            raise Exception('DataChunk `{}{}` already exists'.format(self.name,
                                                                     idx))
        name = self._idx2name(idx)
        dtype = self.dtype
        shape = self.chunk_size
        fillvalue = self.fillvalue
        datachunk = CephDataChunk(self, idx, name, shape, dtype, fillvalue)
        if data is None:
            data = np.full(shape, fillvalue, dtype)
        datachunk.set_data(data, slices, fill_others=True)
        return datachunk

    def get_chunk(self, idx):
        if self.has_chunk(idx):
            name = self._idx2name(idx)
            dtype = self.dtype
            shape = self.chunk_size
            fillvalue = self.fillvalue
            return CephDataChunk(self, idx, name, shape, dtype, fillvalue)
        return self.create_chunk(idx)

    def has_chunk(self, idx):
        name = self._idx2name(idx)
        try:
            self.ioctx.stat(name)
        except rados.ObjectNotFound:
            return False
        return True

    def del_chunk(self, idx):
        if self.has_chunk(idx):
            self.ioctx.remove_object(self._idx2name(idx))
            return True
        return False


class CephDataChunk(BackendDataChunk):

    @property
    def ioctx(self):
        return self.dataset.ioctx

    def get_data(self, slices=None):
        if slices is None:
            slices = slice(None)
        data = np.frombuffer(self.read(), dtype=self.dtype, count=self.size).copy()
        data.shape = self.shape
        return data[slices]

    def set_data(self, values, slices=None, fill_others=False):
        if slices is None or slices2shape(slices) == self.shape:
            self.write_full(values.tobytes())
        else:
            if fill_others:
                cdata = np.full(self.shape, self.fillvalue, self.dtype)
            else:
                cdata = self.get_data()
            cdata[slices] = values
            self.write_full(cdata.tobytes())

    def write_full(self, data):
        self.ioctx.write_full(self.name, data)

    def read(self, length=None, offset=0):
        if length is None:
            length = self.byte_count
        return self.ioctx.read(self.name, length=length, offset=offset)


_backend = Backend('ceph', CephConnection, CephDataset, CephDataChunk)
