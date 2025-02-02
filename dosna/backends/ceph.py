#!/usr/bin/env python
"""Backend ceph uses a ceph cluster to store the dataset and chunks data"""
import logging
import rados

import numpy as np

from dosna.backends import Backend
from dosna.backends.base import (BackendConnection, BackendDataChunk,
                                 BackendDataset, BackendGroup, BackendLink, ConnectionError,
                                 DatasetNotFoundError, GroupNotFoundError, GroupExistsError, DatasetExistsError,                                 ParentLinkError, IndexOutOfRangeError)
from dosna.engines.base import ParentDatasetError
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
        self._root_group = None

    def connect(self):
        if self.connected:
            raise ConnectionError(
                'Connection {} is already open'.format(self.name))
        self._cluster.connect(timeout=self._timeout)
        self._ioctx = self._cluster.open_ioctx(self.name)
        super(CephConnection, self).connect()
        if self.has_group_object(_PATH_SPLIT) == False:
            self.create_root_group()
        self._root_group = self._get_root_group()

    def create_root_group(self):
        self.ioctx.write(_PATH_SPLIT, _SIGNATURE_GROUP.encode(_ENCODING))
        self.ioctx.set_xattr(_PATH_SPLIT, "name", str(_PATH_SPLIT).encode(_ENCODING))
        self.ioctx.set_xattr(_PATH_SPLIT, "attrs", str({}).encode(_ENCODING))
        self.ioctx.set_xattr(_PATH_SPLIT, "absolute_path", str(_PATH_SPLIT).encode(_ENCODING))
        self.ioctx.set_xattr(_PATH_SPLIT, "datasets", str({}).encode(_ENCODING))
        self.ioctx.set_xattr(_PATH_SPLIT, "links", str({}).encode(_ENCODING))
        self.ioctx.set_xattr(_PATH_SPLIT, "parent", str(_PATH_SPLIT).encode(_ENCODING))

    def _get_root_group(self,):
        name = self.ioctx.get_xattr(_PATH_SPLIT, "name").decode()
        absolute_path = self.ioctx.get_xattr(name, "absolute_path").decode()
        group = CephGroup(self, name, absolute_path=absolute_path)
        return group

    def disconnect(self):
        if self.connected:
            self.ioctx.close()
            self._cluster.shutdown()
            super(CephConnection, self).disconnect()

    @property
    def ioctx(self):
        return self._ioctx

    def create_group(self, path, attrs={}):
        return self._root_group.create_group(path, attrs)

    def get_group(self, name):
        if name == _PATH_SPLIT:
            return self._get_root_group()
        return self._root_group.get_group(name)

    def has_group_object(self, name):
        try:
            valid = self.ioctx.stat(name)[0] == len(_SIGNATURE_GROUP.encode(_ENCODING)) and \
                    self.ioctx.read(name) == _SIGNATURE_GROUP.encode(_ENCODING)
        except rados.ObjectNotFound:
            return False
        return valid

    def del_group(self, path):
        return self._root_group.del_group(path)

    def create_dataset(self, name, shape=None, dtype=np.float32, fillvalue=0,
                       data=None, chunk_size=None):
        if not ((shape is not None and dtype is not None) or data is not None):
            raise Exception('Provide `shape` and `dtype` or `data`')
        if self.has_dataset(name):
            raise DatasetExistsError(name)

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


class CephGroup(BackendGroup):
    def __init__(self, parent, name, absolute_path, path_split="/"):
        super(CephGroup, self).__init__(parent, name)
        self.path_split = path_split
        self.absolute_path = absolute_path


    @property
    def ioctx(self):
        return self.parent.ioctx

    def create_absolute_path(self, path):
        current_path = self.absolute_path
        if current_path == self.path_split:
            current_path = path
        else:
            current_path += path
        return current_path

    def create_group(self, path, attrs={}):

        def _create_subgroup(path, group, attrs={}):
            path_elements = path.split(self.path_split)
            for i in range(len(path_elements)-2, 0, -1):
                parent = "/".join(path_elements[:-i])
                if group.name == parent:
                    group = group
                elif group.has_group(parent):
                    group = group._get_group_object(parent)
                else:
                    group = group._create_group_object(parent)
            group = group._create_group_object(path, attrs)
            return group

        if path[0] != "/":
            path = "/" + path
        if self.name != self.path_split:
            path = self.name + path
        if self.has_group(path):
            raise GroupExistsError(path)
        group = _create_subgroup(path, self, attrs)
        return group

    def _create_group_link(self, path):
        links = str2dict(self.ioctx.get_xattr(self.name,"links").decode())
        link_name = self.name + "->" + path
        links[path] = {"name": link_name,
                       "source": self.name,
                       "target": path}
        self.ioctx.set_xattr(self.name, "links", dict2str(links).encode(_ENCODING))

    def _create_dataset_link(self, path):
        datasets = str2dict(self.ioctx.get_xattr(self.name, "datasets").decode())
        datasets[path] = path
        self.ioctx.set_xattr(self.name, "datasets", dict2str(datasets).encode(_ENCODING))

    def create_link(self, path):
        if self._has_group_object(path):
            self._create_group_link(path)
            return True
        elif self._has_dataset_object(path):
            self._create_dataset_link(path)
            return True
        return False

    def _del_group_link(self, name):
        links = str2dict(self.ioctx.get_xattr(self.name, "links").decode())
        if name in links:
            link_parent = self.ioctx.get_xattr(name, "parent").decode()
            if link_parent != self.name:
                del links[name]
                self.ioctx.set_xattr(self.name, "links", dict2str(links).encode(_ENCODING))
                return links
            raise ParentLinkError(self.name, name)
        raise GroupNotFoundError(self.name)

    def _del_dataset_link(self, name):
        datasets = str2dict(self.ioctx.get_xattr(self.name, "datasets").decode())
        if name in datasets:
            if self.name == self.path_split:
                if name[:len(self.name)] == self.name:
                    raise ParentLinkError(self.name, name)
            elif name[:len(self.name) + 1] == self.name + self.path_split:
                raise ParentLinkError(self.name, name)
            del datasets[name]
            self.ioctx.set_xattr(self.name, "datasets", str(datasets).encode(_ENCODING))
            return datasets
        raise DatasetNotFoundError(name)

    def del_link(self, name):
        if self._has_group_object(name):
            self._del_group_link(name)
            return True
        elif self._has_dataset_object(name):
            self._del_dataset_link(name)
            return True
        return False

    def _create_group_object(self, path, attrs={}):
        attrs = {str(key): str(value) for key, value in attrs.items()}
        absolute_path = self.create_absolute_path(path)
        self.ioctx.write(path, _SIGNATURE_GROUP.encode(_ENCODING))
        self.ioctx.set_xattr(path, "name", str(path).encode(_ENCODING))
        self.ioctx.set_xattr(path, "attrs", str(attrs).encode(_ENCODING))
        self.ioctx.set_xattr(path, "absolute_path", str(absolute_path).encode(_ENCODING))
        self.ioctx.set_xattr(path, "datasets", str({}).encode(_ENCODING))
        self.ioctx.set_xattr(path, "links", str({}).encode(_ENCODING))
        self.ioctx.set_xattr(path, "parent", str(self.name).encode(_ENCODING))
        group = CephGroup(self, path, absolute_path)
        self.create_link(path)
        return group

    def get_group(self, path):
        def _find_group(path):
            group = self
            for i in range(1, len(path)+1):
                links = group.get_links()
                link_path = self.path_split.join(path[:i])
                if link_path in links:
                    group = group._get_group_object(link_path)
            return group

        path_elements = path.split(self.path_split)
        group = _find_group(path_elements)
        if group == self or group.name != path:
            raise GroupNotFoundError(path)
        return group

    def _get_group_object(self, name):
        if self._has_group_object(name):
            name = self.ioctx.get_xattr(name, "name").decode()
            absolute_path = self.ioctx.get_xattr(name, "absolute_path").decode()
            group = CephGroup(self, name, absolute_path, path_split="/")
            return group
        raise GroupNotFoundError(name)

    def has_group(self, name):
        try:
            valid = self.ioctx.stat(name)[0] == len(_SIGNATURE_GROUP.encode(_ENCODING)) and \
                    self.ioctx.read(name) == _SIGNATURE_GROUP.encode(_ENCODING)
        except rados.ObjectNotFound:
            return False
        return valid

    def _has_group_object(self, name):
        try:
            valid = self.ioctx.stat(name)[0] == len(_SIGNATURE_GROUP.encode(_ENCODING)) and \
                    self.ioctx.read(name) == _SIGNATURE_GROUP.encode(_ENCODING)
        except rados.ObjectNotFound:
            return False
        return valid

    def _del_group_object(self, path):
        if self._has_group_object(path):
            parent = self.ioctx.get_xattr(path, "parent").decode()
            if self._has_group_object(parent):
                links = str2dict(self.ioctx.get_xattr(parent, "links").decode())
                del links[path]
                self.ioctx.set_xattr(parent, "links", dict2str(links).encode(_ENCODING))
            self.ioctx.remove_object(path)
            return True
        return False

    def del_group(self, path, root=None):
        def del_sub_group(node, root, datasets):
            links = node.get_links()
            for link in links:
                node = self._get_group_object(link)
                del_sub_group(node, root, datasets)
                if node.absolute_path[:len(root.absolute_path)+1] == root.name + root.path_split:
                    if node.get_datasets() is not {}:
                        for data in node.get_datasets():
                            if data[:len(node.name)+1] == node.name+self.path_split:
                                datasets.append(data)
                    root._del_group_object(node.name)

        if self.has_group(path):
            datasets = []
            root = self._get_group_object(path)
            del_sub_group(root, root, datasets)
            if root.get_datasets() is not {}:
                for key in root.get_datasets():
                    if key[:len(root.name) + 1] == root.name + self.path_split:
                        datasets.append(key)
            self._del_group_object(path)
            return datasets
        raise GroupNotFoundError(path)

    def get_attrs(self):
        return str2dict(self.ioctx.get_xattr(self.name, "attrs").decode())

    def set_attrs(self, attrs):
        self.ioctx.set_xattr(self.name, "attrs", str(attrs).encode(_ENCODING))
        return str2dict(self.ioctx.get_xattr(self.name, "attrs").decode())

    def get_links(self):
        links = str2dict(self.ioctx.get_xattr(self.name, "links").decode())
        for key, value in links.items():
            path = value["name"]
            source = value["source"]
            target = value["target"]
            if self.has_group(value["target"]):
                links[key] = CephLink(source, target, path)
            else:
                target = None
                links[key] = CephLink(source, target, path)
        return links


    def create_dataset(self, name, shape=None, dtype=np.float32, fillvalue=0,
                       data=None, chunk_size=None):
        if not ((shape is not None and dtype is not None) or data is not None):
            raise Exception('Provide `shape` and `dtype` or `data`')
        if self.name == self.path_split:
            if name[0] != self.path_split:
                name = self.name + name
        elif name[:len(self.name) + 1] != self.name + self.path_split:
            name = self.name + self.path_split + name
        if self._has_dataset_object(name):
            raise DatasetExistsError(name)

        if data is not None:
            shape = data.shape
            dtype = data.dtype

        if chunk_size is None:
            chunk_size = shape

        chunk_grid = (np.ceil(np.asarray(shape, float) / chunk_size)) \
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
        datasets = str2dict(self.ioctx.get_xattr(self.name, "datasets").decode())
        dataset_name = name
        datasets[dataset_name] = dataset_name
        self.ioctx.set_xattr(self.name, "datasets", dict2str(datasets).encode(_ENCODING))
        return dataset

    def _get_dataset(self, name):
        shape = str2shape(self.ioctx.get_xattr(name, 'shape').decode())
        dtype = self.ioctx.get_xattr(name, 'dtype').decode()
        fillvalue = int(self.ioctx.get_xattr(name, 'fillvalue').decode())
        chunk_grid = str2shape(self.ioctx.get_xattr(name, 'chunk_grid').decode())
        chunk_size = str2shape(self.ioctx.get_xattr(name, 'chunk_size').decode())
        dataset = CephDataset(self, name, shape, dtype, fillvalue,
                              chunk_grid, chunk_size)
        return dataset

    def has_dataset(self, name):
        if name in str2dict(self.ioctx.get_xattr(self.name, "datasets").decode()):
            return True
        return False

    def get_datasets(self):
        datasets = str2dict(self.ioctx.get_xattr(self.name, "datasets").decode())
        return datasets

    def get_dataset(self, name):
        if not self.has_dataset(name):
            raise DatasetNotFoundError('Dataset `%s` does not exist' % name)
        if not self._has_dataset_object(name):
            return None
        return self._get_dataset(name)

    def del_dataset(self, name):
        log.debug("Removing dataset %s", name)
        if self.has_dataset(name):
            self.ioctx.remove_object(name)
            datasets = str2dict(self.ioctx.get_xattr(self.name, "datasets").decode())
            del datasets[name]
            self.ioctx.set_xattr(self.name, "datasets", dict2str(datasets).encode(_ENCODING))
        else:
            raise DatasetNotFoundError(
                'Dataset `{}` does not exist'.format(name))

    def _get_dataset_object(self, name):
        if not self._has_dataset_object(name):
            raise DatasetNotFoundError('Dataset `%s` does not exist' % name)
        return self._get_dataset(name)

    def _has_dataset_object(self, name):
        try:
            valid = self.ioctx.stat(name)[0] == len(_SIGNATURE.encode(_ENCODING)) and \
                self.ioctx.read(name) == _SIGNATURE.encode(_ENCODING)
        except rados.ObjectNotFound:
            return False
        return valid

    def _del_dataset_object(self, name):
        log.debug("Removing dataset %s", name)
        if self._has_dataset_object(name):
            self.ioctx.remove_object(name)
        else:
            raise DatasetNotFoundError(
                'Dataset `{}` does not exist'.format(name))


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
        if idx > self._idx_from_flat(self.total_chunks - 1):
            raise IndexOutOfRangeError(idx, self._idx_from_flat(self.total_chunks - 1))
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
        if idx > self._idx_from_flat(self.total_chunks-1):
            raise IndexOutOfRangeError(idx, self._idx_from_flat(self.total_chunks-1))
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
