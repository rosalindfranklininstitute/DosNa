#!/usr/bin/env python
"""Backend s3 uses a S3 interface to store the dataset and chunks data"""

import logging

import numpy as np

import boto3
from botocore.exceptions import ClientError
from dosna.backends import Backend
from dosna.backends.base import (
    BackendConnection,
    BackendDataChunk,
    BackendDataset,
    BackendGroup,
    BackendLink,
    ConnectionError,
    DatasetNotFoundError,
    GroupExistsError,
    GroupNotFoundError,
    ParentLinkError,
    DatasetExistsError,
    IndexOutOfRangeError,
)
from dosna.util import dtype2str, shape2str, str2shape, str2dict, dict2str
from dosna.util.data import slices2shape

_PATH_SPLIT = "/"
_SIGNATURE = "DosNa Dataset"
_SIGNATURE_GROUP = "DosNa Group"
_SIGNATURE_LINK = "Dosna Link"
_ENCODING = "utf-8"
_SHAPE = "shape"
_DTYPE = "dtype"
_FILLVALUE = "fillvalue"
_CHUNK_GRID = "chunk-grid"
_CHUNK_SIZE = "chunk-size"
_NAME = "name"
_ATTRS = "attrs"
_ABSOLUTE_PATH = "absolute-path"
_DATASETS = "datasets"
_LINKS = "links"
_PARENT = "parent"

log = logging.getLogger(__name__)

# Sanitise bucket name to conform to AWS conventions
def bucket_name(name):
    return name.replace("_", "-").lower()


class S3Connection(BackendConnection):
    """
    A S3 Connection that wraps boto3 S3 client
    """

    def __init__(
        self,
        name,
        endpoint_url=None,
        verify=True,
        profile_name="default",
        *args,
        **kwargs
    ):
        super(S3Connection, self).__init__(name, *args, **kwargs)

        self._endpoint_url = endpoint_url
        self._verify = verify
        self._client = None
        self._profile_name = profile_name
        self._root_group = None
        super(S3Connection, self).__init__(bucket_name(name), *args, **kwargs)

    def connect(self):

        if self.connected:
            raise ConnectionError("Connection {} is already open".format(self.name))
        session = boto3.session.Session(profile_name=self._profile_name)

        # Use access key and secret_key in call to client?
        self._client = session.client(
            service_name="s3", endpoint_url=self._endpoint_url, verify=self._verify
        )
        self._client.create_bucket(Bucket=self.name)
        super(S3Connection, self).connect()
        if self.has_group_object(_PATH_SPLIT) == False:
            self.create_root_group()
        self._root_group = self._get_root_group()

    def create_root_group(self):
        metadata = {
            _NAME: str(_PATH_SPLIT),
            _ATTRS: str({}),
            _ABSOLUTE_PATH: str(_PATH_SPLIT),
            _DATASETS: str({}),
            _LINKS: str({}),
            _PARENT: str(_PATH_SPLIT),
        }
        self._client.put_object(
            Bucket=self.name,
            Key=_PATH_SPLIT,
            Body=_SIGNATURE_GROUP.encode(_ENCODING),
            Metadata=metadata,
        )

    def _get_root_group(
        self,
    ):
        metadata = self._client.head_object(Bucket=self.name, Key=_PATH_SPLIT)[
            "Metadata"
        ]
        group = S3Group(self, metadata[_NAME], absolute_path=metadata[_ABSOLUTE_PATH])
        return group

    def disconnect(self):

        if self.connected:
            super(S3Connection, self).disconnect()

    @property
    def client(self):
        return self._client

    @property
    def bucket_name(self):
        return self.name

    def create_group(self, path, attrs={}):
        return self._root_group.create_group(path, attrs)

    def get_group(self, name):
        if name == _PATH_SPLIT:
            return self._get_root_group()
        return self._root_group.get_group(name)

    def has_group_object(self, name):
        try:
            valid = self._client.get_object(Bucket=self.name, Key=name)[
                "Body"
            ].read() == _SIGNATURE_GROUP.encode(_ENCODING)
        except Exception:  # Any exception it should return false
            return False
        return valid

    def del_group(self, path):
        return self._root_group.del_group(path)

    def create_dataset(
        self,
        name,
        shape=None,
        dtype=np.float32,
        fillvalue=0,
        data=None,
        chunk_size=None,
    ):
        if not ((shape is not None and dtype is not None) or data is not None):
            raise Exception("Provide `shape` and `dtype` or `data`")
        if self.has_dataset(name):
            raise Exception("Dataset `%s` already exists" % name)

        if data is not None:
            shape = data.shape
            dtype = data.dtype

        if chunk_size is None:
            chunk_size = shape

        chunk_grid = (np.ceil(np.asarray(shape, float) / chunk_size)).astype(int)

        log.debug(
            "creating dataset %s with shape:%s chunk_size:%s " "chunk_grid:%s",
            name,
            shape,
            chunk_size,
            chunk_grid,
        )

        metadata = {
            _SHAPE: shape2str(shape),
            _DTYPE: dtype2str(dtype),
            _FILLVALUE: repr(fillvalue),
            _CHUNK_GRID: shape2str(chunk_grid),
            _CHUNK_SIZE: shape2str(chunk_size),
        }
        self._client.put_object(
            Bucket=self.name,
            Key=name,
            Body=_SIGNATURE.encode(_ENCODING),
            Metadata=metadata,
        )

        dataset = S3Dataset(self, name, shape, dtype, fillvalue, chunk_grid, chunk_size)

        return dataset

    def get_dataset(self, name):

        if not self.has_dataset(name):
            raise DatasetNotFoundError("Dataset `%s` does not exist" % name)

        metadata = self._client.head_object(Bucket=self.name, Key=name)["Metadata"]
        if metadata is None:
            raise DatasetNotFoundError(
                "Dataset `%s` does not have required DosNa metadata" % name
            )

        shape = str2shape(metadata[_SHAPE])
        dtype = metadata[_DTYPE]
        fillvalue = int(metadata[_FILLVALUE])
        chunk_grid = str2shape(metadata[_CHUNK_GRID])
        chunk_size = str2shape(metadata[_CHUNK_SIZE])
        dataset = S3Dataset(self, name, shape, dtype, fillvalue, chunk_grid, chunk_size)

        return dataset

    def has_dataset(self, name):
        try:
            valid = self._client.get_object(Bucket=self.name, Key=name)[
                "Body"
            ].read() == _SIGNATURE.encode(_ENCODING)
        except Exception:  # Any exception it should return false
            return False
        return valid

    def del_dataset(self, name):

        if self.has_dataset(name):
            try:
                self._client.delete_object(Bucket=self.name, Key=name)
            except ClientError as e:
                log.error(
                    "del_dataset: cannot delete %s: %s", name, e.response["Error"]
                )
        else:
            raise DatasetNotFoundError("Dataset `{}` does not exist".format(name))


class S3Link(BackendLink):
    def __init__(self, source, target, name):
        super(S3Link, self).__init__(source, target, name)


class S3Group(BackendGroup):
    def __init__(self, parent, name, absolute_path, path_split="/"):
        super(S3Group, self).__init__(parent, name)
        self.path_split = path_split
        self.absolute_path = absolute_path

    @property
    def client(self):
        return self.parent.client

    @property
    def bucket_name(self):
        return self.parent.bucket_name

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
            for i in range(len(path_elements) - 2, 0, -1):
                parent = "/".join(path_elements[:-i])
                if group.name == parent:
                    group = group
                elif group._has_group_object(parent):
                    group = group._get_group_object(parent)
                else:
                    group = group._create_group_object(parent)
            group = group._create_group_object(path, attrs)
            return group

        if path[0] != "/":
            path = "/" + path
        if self.name != self.path_split:
            path = self.name + path
        if self._has_group_object(path):
            raise GroupExistsError(path)
        group = _create_subgroup(path, self, attrs)
        return group

    def _create_group_link(self, path):
        header = self.client.head_object(Bucket=self.bucket_name, Key=self.name)
        metadata = header["Metadata"]
        links = str2dict(metadata[_LINKS])
        link_name = self.name + "->" + path
        links[path] = {"name": link_name, "source": self.name, "target": path}
        metadata[_LINKS] = str(links)
        self.client.copy_object(
            Bucket=self.bucket_name,
            Key=self.name,
            CopySource={"Bucket": self.bucket_name, "Key": self.name},
            Metadata=metadata,
            MetadataDirective="REPLACE",
            CopySourceIfMatch=header["ETag"],
        )

    def _create_dataset_link(self, path):
        header = self.client.head_object(Bucket=self.bucket_name, Key=self.name)
        metadata = header["Metadata"]
        datasets = str2dict(metadata[_DATASETS])
        datasets[path] = path
        metadata[_DATASETS] = dict2str(datasets)
        self.client.copy_object(
            Bucket=self.bucket_name,
            Key=self.name,
            CopySource={"Bucket": self.bucket_name, "Key": self.name},
            Metadata=metadata,
            MetadataDirective="REPLACE",
            CopySourceIfMatch=header["ETag"],
        )
        return datasets

    def create_link(self, path):
        if self._has_group_object(path):
            self._create_group_link(path)
            return True
        elif self._has_dataset_object(path):
            self._create_dataset_link(path)
            return True
        return False

    def _del_group_link(self, name):
        links = str2dict(
            self.client.head_object(Bucket=self.bucket_name, Key=self.name)["Metadata"][
                _LINKS
            ]
        )
        if name in links:
            header = self.client.head_object(Bucket=self.bucket_name, Key=name)
            link_parent = header["Metadata"][_PARENT]
            if link_parent != self.name:
                del links[name]
                self.set_links(links)
                return links
            raise ParentLinkError(self.name, name)
        raise GroupNotFoundError(self.name)

    def _del_dataset_link(self, name):
        header = self.client.head_object(Bucket=self.bucket_name, Key=self.name)
        metadata = header["Metadata"]
        datasets = str2dict(metadata[_DATASETS])
        if name in datasets:
            if self.name == self.path_split:
                if name[: len(self.name)] == self.name:
                    raise ParentLinkError(self.name, name)
            elif name[: len(self.name) + 1] == self.name + self.path_split:
                raise ParentLinkError(self.name, name)
            del datasets[name]
            metadata[_DATASETS] = dict2str(datasets)
            self.client.copy_object(
                Bucket=self.bucket_name,
                Key=self.name,
                CopySource={"Bucket": self.bucket_name, "Key": self.name},
                Metadata=metadata,
                MetadataDirective="REPLACE",
                CopySourceIfMatch=header["ETag"],
            )
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

        metadata = {
            _NAME: str(path),
            _ATTRS: str(attrs),
            _ABSOLUTE_PATH: str(absolute_path),
            _DATASETS: str({}),
            _LINKS: str({}),
            _PARENT: str(self.name),
        }
        self.client.put_object(
            Bucket=self.bucket_name,
            Key=path,
            Body=_SIGNATURE_GROUP.encode(_ENCODING),
            Metadata=metadata,
        )
        group = S3Group(self, path, absolute_path)
        self.create_link(path)
        return group

    def get_group(self, path):
        def _find_group(path):
            group = self
            for i in range(1, len(path) + 1):
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
            metadata = self.client.head_object(Bucket=self.bucket_name, Key=name)[
                "Metadata"
            ]
            group = S3Group(
                self, metadata[_NAME], absolute_path=metadata[_ABSOLUTE_PATH]
            )
            return group
        raise GroupNotFoundError(name)

    def has_group(self, name):
        try:
            valid = self.client.get_object(Bucket=self.bucket_name, Key=name)[
                "Body"
            ].read() == _SIGNATURE_GROUP.encode(_ENCODING)
        except Exception:  # Any exception it should return false
            return False
        return valid

    def _has_group_object(self, name):
        try:
            valid = self.client.get_object(Bucket=self.bucket_name, Key=name)[
                "Body"
            ].read() == _SIGNATURE_GROUP.encode(_ENCODING)
        except Exception:  # Any exception it should return false
            return False
        return valid

    def _del_group_object(self, path):
        if self._has_group_object(path):
            header = self.client.head_object(Bucket=self.bucket_name, Key=path)
            parent = header["Metadata"][_PARENT]
            if self._has_group_object(parent):
                parent_head = self.client.head_object(
                    Bucket=self.bucket_name, Key=parent
                )
                links = str2dict(parent_head["Metadata"][_LINKS])
                del links[path]
                parent_head["Metadata"][_LINKS] = dict2str(links)
                self.client.copy_object(
                    Bucket=self.bucket_name,
                    Key=parent,
                    CopySource={"Bucket": self.bucket_name, "Key": parent},
                    Metadata=parent_head["Metadata"],
                    MetadataDirective="REPLACE",
                    CopySourceIfMatch=header["ETag"],
                )
            self.client.delete_object(Bucket=self.bucket_name, Key=path)
            return True
        return False

    def del_group(self, path, root=None):
        def del_sub_group(node, root, datasets):
            links = node.get_links()
            for link in links:
                node = self._get_group_object(link)
                del_sub_group(node, root, datasets)
                if (
                    node.absolute_path[: len(root.absolute_path) + 1]
                    == root.name + root.path_split
                ):
                    if node.get_datasets() is not {}:
                        for data in node.get_datasets():
                            if (
                                data[: len(node.name) + 1]
                                == node.name + self.path_split
                            ):
                                datasets.append(data)
                    root._del_group_object(node.name)

        if self._has_group_object(path):
            datasets = []
            root = self._get_group_object(path)
            del_sub_group(root, root, datasets)
            if root.get_datasets() is not {}:
                for key in root.get_datasets():
                    if key[: len(root.name) + 1] == root.name + self.path_split:
                        datasets.append(key)
            self._del_group_object(path)
            return datasets
        raise GroupNotFoundError(path)

    def get_attrs(self):
        return str2dict(
            self.client.head_object(Bucket=self.bucket_name, Key=self.name)["Metadata"][
                _ATTRS
            ]
        )

    def set_attrs(self, attrs):
        header = self.client.head_object(Bucket=self.bucket_name, Key=self.name)
        metadata = header["Metadata"]
        metadata[_ATTRS] = str(attrs)
        self.client.copy_object(
            Bucket=self.bucket_name,
            Key=self.name,
            CopySource={"Bucket": self.bucket_name, "Key": self.name},
            Metadata=metadata,
            MetadataDirective="REPLACE",
            CopySourceIfMatch=header["ETag"],
        )
        return str2dict(
            self.client.head_object(Bucket=self.bucket_name, Key=self.name)["Metadata"][
                _ATTRS
            ]
        )

    def get_links(self):
        links = str2dict(
            self.client.head_object(Bucket=self.bucket_name, Key=self.name)["Metadata"][
                _LINKS
            ]
        )
        for key, value in links.items():
            path = value["name"]
            source = value["source"]
            target = value["target"]
            if self._has_group_object(value["target"]):
                links[key] = S3Link(source, target, path)
            else:
                target = None
                links[key] = S3Link(source, target, path)
        return links

    def set_links(self, links):
        header = self.client.head_object(Bucket=self.bucket_name, Key=self.name)
        metadata = header["Metadata"]
        metadata[_LINKS] = str(links)
        self.client.copy_object(
            Bucket=self.bucket_name,
            Key=self.name,
            CopySource={"Bucket": self.bucket_name, "Key": self.name},
            Metadata=metadata,
            MetadataDirective="REPLACE",
            CopySourceIfMatch=header["ETag"],
        )
        return str2dict(
            self.client.head_object(Bucket=self.bucket_name, Key=self.name)["Metadata"][
                _LINKS
            ]
        )

    def create_dataset(
        self,
        name,
        shape=None,
        dtype=np.float32,
        fillvalue=0,
        data=None,
        chunk_size=None,
    ):
        if not ((shape is not None and dtype is not None) or data is not None):
            raise Exception("Provide `shape` and `dtype` or `data`")
        if self.name == self.path_split:
            if name[0] != self.path_split:
                name = self.name + name
        elif name[: len(self.name) + 1] != self.name + self.path_split:
            name = self.name + self.path_split + name
        if self._has_dataset_object(name):
            raise DatasetExistsError(name)

        if data is not None:
            shape = data.shape
            dtype = data.dtype

        if chunk_size is None:
            chunk_size = shape

        chunk_grid = (np.ceil(np.asarray(shape, float) / chunk_size)).astype(int)

        log.debug(
            "creating dataset %s with shape:%s chunk_size:%s " "chunk_grid:%s",
            name,
            shape,
            chunk_size,
            chunk_grid,
        )
        metadata = {
            _SHAPE: shape2str(shape),
            _DTYPE: dtype2str(dtype),
            _FILLVALUE: repr(fillvalue),
            _CHUNK_GRID: shape2str(chunk_grid),
            _CHUNK_SIZE: shape2str(chunk_size),
        }
        self.client.put_object(
            Bucket=self.bucket_name,
            Key=name,
            Body=_SIGNATURE.encode(_ENCODING),
            Metadata=metadata,
        )

        dataset = S3Dataset(self, name, shape, dtype, fillvalue, chunk_grid, chunk_size)

        header = self.client.head_object(Bucket=self.bucket_name, Key=self.name)
        metadata = header["Metadata"]
        datasets = str2dict(metadata[_DATASETS])
        dataset_name = name
        datasets[dataset_name] = dataset_name
        metadata[_DATASETS] = dict2str(datasets)

        self.client.copy_object(
            Bucket=self.bucket_name,
            Key=self.name,
            CopySource={"Bucket": self.bucket_name, "Key": self.name},
            Metadata=metadata,
            MetadataDirective="REPLACE",
            CopySourceIfMatch=header["ETag"],
        )
        return dataset

    def _get_dataset(self, name):
        metadata = self.client.head_object(Bucket=self.bucket_name, Key=name)[
            "Metadata"
        ]
        if metadata is None:
            raise DatasetNotFoundError(
                "Dataset `%s` does not have required DosNa metadata" % name
            )

        shape = str2shape(metadata[_SHAPE])
        dtype = metadata[_DTYPE]
        fillvalue = int(metadata[_FILLVALUE])
        chunk_grid = str2shape(metadata[_CHUNK_GRID])
        chunk_size = str2shape(metadata[_CHUNK_SIZE])
        dataset = S3Dataset(self, name, shape, dtype, fillvalue, chunk_grid, chunk_size)

        return dataset

    def has_dataset(self, name):
        if name in str2dict(
            self.client.head_object(Bucket=self.bucket_name, Key=self.name)["Metadata"][
                _DATASETS
            ]
        ):
            return True
        return False

    def get_datasets(self):
        datasets = str2dict(
            self.client.head_object(Bucket=self.bucket_name, Key=self.name)["Metadata"][
                _DATASETS
            ]
        )
        return datasets

    def get_dataset(self, name):
        if not self.has_dataset(name):
            raise DatasetNotFoundError("Dataset `%s` does not exist" % name)
        if not self._has_dataset_object(name):
            return None
        return self._get_dataset(name)

    def del_dataset(self, name):
        if self.has_dataset(name):
            self.client.delete_object(Bucket=self.bucket_name, Key=name)
            header = self.client.head_object(Bucket=self.bucket_name, Key=self.name)
            metadata = header["Metadata"]
            datasets = str2dict(metadata[_DATASETS])
            del datasets[name]
            metadata[_DATASETS] = dict2str(datasets)
            self.client.copy_object(
                Bucket=self.bucket_name,
                Key=self.name,
                CopySource={"Bucket": self.bucket_name, "Key": self.name},
                Metadata=metadata,
                MetadataDirective="REPLACE",
                CopySourceIfMatch=header["ETag"],
            )
        else:
            raise DatasetNotFoundError("Dataset `{}` does not exist".format(name))

    def _get_dataset_object(self, name):
        if not self._has_dataset_object(name):
            raise DatasetNotFoundError("Dataset `%s` does not exist" % name)
        return self._get_dataset(name)

    def _has_dataset_object(self, name):
        try:
            valid = self.client.get_object(Bucket=self.bucket_name, Key=name)[
                "Body"
            ].read() == _SIGNATURE.encode(_ENCODING)
        except Exception:  # Any exception it should return false
            return False
        return valid

    def _del_dataset_object(self, name):
        if self._has_dataset_object(name):
            try:
                self.client.delete_object(Bucket=self.bucket_name, Key=name)
            except ClientError as e:
                log.error(
                    "del_dataset: cannot delete %s: %s", name, e.response["Error"]
                )
        else:
            raise DatasetNotFoundError("Dataset `{}` does not exist".format(name))


class S3Dataset(BackendDataset):
    """
    S3Dataset
    """

    @property
    def client(self):
        return self.connection.client

    @property
    def bucket_name(self):
        return self.connection.bucket_name

    def _idx2name(self, idx):
        return "{}/{}".format(self.name, ".".join(map(str, idx)))

    def create_chunk(self, idx, data=None, slices=None):
        if idx > self._idx_from_flat(self.total_chunks - 1):
            raise IndexOutOfRangeError(idx, self._idx_from_flat(self.total_chunks - 1))
        if self.has_chunk(idx):
            raise Exception("DataChunk `{}{}` already exists".format(self.name, idx))
        name = self._idx2name(idx)
        dtype = self.dtype
        shape = self.chunk_size
        fillvalue = self.fillvalue
        datachunk = S3DataChunk(self, idx, name, shape, dtype, fillvalue)
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
            return S3DataChunk(self, idx, name, shape, dtype, fillvalue)
        return self.create_chunk(idx)

    def has_chunk(self, idx):

        has_chunk = False
        name = self._idx2name(idx)
        try:
            self.client.head_object(Bucket=self.bucket_name, Key=name)
            has_chunk = True
        except ClientError as e:
            logging.debug("ClientError: %s", e.response["Error"]["Code"])

        return has_chunk

    def del_chunk(self, idx):
        if self.has_chunk(idx):
            self.client.delete_object(Bucket=self.bucket_name, Key=self._idx2name(idx))
            return True
        return False


class S3DataChunk(BackendDataChunk):
    @property
    def client(self):
        return self.dataset.client

    @property
    def bucket_name(self):
        return self.dataset.bucket_name

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

        self.client.put_object(Bucket=self.bucket_name, Key=self.name, Body=data)

    def read(self, length=None, offset=0):
        if length is None:
            length = self.byte_count

        byteRange = "bytes={}-{}".format(offset, offset + length - 1)
        return self.client.get_object(
            Bucket=self.bucket_name, Key=self.name, Range=byteRange
        )["Body"].read()


_backend = Backend("s3", S3Connection, S3Dataset, S3DataChunk)
