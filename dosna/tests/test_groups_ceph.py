import time
import dosna as dn
import unittest

from dosna.engines.cpu import CpuGroup
from dosna.backends.ceph import CephGroup

from dosna.util import str2dict
from dosna.backends.base import (
    DatasetNotFoundError,
    GroupNotFoundError,
    GroupExistsError,
)
_SIGNATURE = "DosNa Dataset"
_SIGNATURE_GROUP = "DosNa Group"
_SIGNATURE_LINK =  "Dosna Link"
_ENCODING = "utf-8"
PATH_SPLIT = "/"


class GroupTest(unittest.TestCase):
    """
    Test dataset actions
    """

    connection_handle = None

    @classmethod
    def setUpClass(cls):
        dn.use(backend="ceph", engine="cpu")
        cls.connection_handle = dn.Connection("dosna", conffile="ceph.conf")
        cls.connection_handle.connect()
        cls.iocxt = cls.connection_handle.instance.ioctx

    def setUp(self):
        if self.connection_handle.connected == False:
            self.connection_handle = dn.Connection("dosna", conffile="ceph.conf")
            self.connection_handle.connect()
            self.iocxt = self.connection_handle.instance.ioctx
        self._started_at = time.time()

    @classmethod
    def tearDownClass(cls):
        cls.connection_handle.disconnect()

    def tearDown(self):
        elapsed = time.time() - self._started_at
        print('{} ({}s)'.format(self.id(), round(elapsed, 5)))
        if dn.get_backend().name == "ceph":
            for obj in self.connection_handle.ioctx.list_objects():
                self.connection_handle.ioctx.remove_object(obj.key)
        self.connection_handle.disconnect()

    def check_group(
        self, group, name, absolute_path, attrs={}, links={}, datasets={}
    ):
        self.assertEqual(_SIGNATURE_GROUP, str(self.connection_handle.ioctx.read(PATH_SPLIT).decode()))
        self.assertEqual(group.name, str(self.iocxt.get_xattr(name, "name").decode()))
        self.assertEqual(group.name, name)
        self.assertEqual(
            group.absolute_path, str(self.iocxt.get_xattr(name, "absolute_path").decode())
        )
        self.assertEqual(group.absolute_path, absolute_path)
        self.assertDictEqual(
            group.attrs, str2dict(str(self.iocxt.get_xattr(name, "attrs").decode()))
        )
        self.assertEqual(group.attrs, attrs)
        self.assertDictEqual(
            group.links, str2dict(str(self.iocxt.get_xattr(name, "links").decode()))
        )
        self.assertDictEqual(group.links, links)
        self.assertDictEqual(
            group.datasets, str2dict(str(self.iocxt.get_xattr(name, "datasets").decode()))
        )
        self.assertDictEqual(group.datasets, datasets)

    def test_root_group_exists(self):
        self.assertEqual(_SIGNATURE_GROUP, str(self.connection_handle.ioctx.read(PATH_SPLIT).decode()))

    def test__create_group_object(self):
        name = "/A"
        root = self.connection_handle.get_group(PATH_SPLIT)
        self.assertEqual(type(root), CpuGroup)

        A = root._create_group_object(name)
        # This should be a ceph group as it's access via a private method within CephGroup
        self.assertEqual(type(A), CephGroup)
        self.check_group(A, name, name)
        attrs = {"A1": "V1"}

        A = root._create_group_object(name, attrs)
        # This should be a ceph group as it's access via a private method within CephGroup
        self.assertEqual(type(A), CephGroup)
        self.check_group(A, name, name, attrs)

        # Check overwrites
        A = root._create_group_object(name)
        self.check_group(A, name, name)
