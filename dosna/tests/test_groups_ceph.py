import time
import unittest

import rados

import numpy as np
from numpy.testing import assert_array_equal

import dosna as dn
from dosna.engines.cpu import CpuGroup, CpuLink, CpuDataset
from dosna.backends.ceph import CephGroup, CephLink
from dosna.backends.base import (
    DatasetNotFoundError,
    GroupNotFoundError,
    GroupExistsError, DatasetExistsError,
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
        cls.ioctx = cls.connection_handle.instance.ioctx

    def setUp(self):
        if self.connection_handle.connected == False:
            self.connection_handle = dn.Connection("dosna", conffile="ceph.conf")
            self.connection_handle.connect()
            self.ioctx = self.connection_handle.instance.ioctx
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

    def check_group(self, group, name, absolute_path):
        self.assertEqual(_SIGNATURE_GROUP, str(self.connection_handle.ioctx.read(PATH_SPLIT).decode()))
        self.assertEqual(group.name, str(self.ioctx.get_xattr(name, "name").decode()))
        self.assertEqual(group.name, name)
        self.assertEqual(
            group.absolute_path, str(self.ioctx.get_xattr(name, "absolute_path").decode())
        )
        self.assertEqual(group.absolute_path, absolute_path)


    def test_root_group_exists(self):
        self.assertEqual(_SIGNATURE_GROUP, str(self.ioctx.read(PATH_SPLIT).decode()))

    def test_create_root_group(self):
        with self.assertRaises(GroupExistsError):
            self.connection_handle.create_group("/")

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
        self.check_group(A, name, name)

        # Check overwrites
        A = root._create_group_object(name)
        self.check_group(A, name, name)

    def test_create_group(self):
        root = self.connection_handle.get_group(PATH_SPLIT)
        group_name = "/FakeGroup"
        group_obj = self.connection_handle.create_group(group_name)
        self.assertEqual(type(group_obj), CpuGroup)
        self.check_group(group_obj, group_name, group_name)
        self.assertEqual(type(root), CpuGroup)
        self.assertIn(group_name, root.get_links())

    def test_create_existing_group(self):
        root = self.connection_handle.get_group(PATH_SPLIT)
        group_name = "/FakeGroup"
        self.connection_handle.create_group(group_name)
        with self.assertRaises(GroupExistsError):
            group_obj = self.connection_handle.create_group(group_name)


    def test_create_group_with_attrs(self):
        root = self.connection_handle.get_group(PATH_SPLIT)
        group_name = "/FakeGroup"
        attrs = {"A1": "V1"}
        group_obj = self.connection_handle.create_group(group_name, attrs)
        self.assertEqual(type(group_obj), CpuGroup)
        self.check_group(group_obj, group_name, group_name)
        self.assertEqual(attrs, group_obj.get_attrs())
        self.assertEqual(type(root), CpuGroup)
        self.assertIn(group_name, root.get_links())

    def test_create_subgroups(self):
        root = self.connection_handle.get_group(PATH_SPLIT)
        attrs = {"C1": "V1"}
        groups = "/A/B/C"  # Expected /A -> /A/B -> /A/B/C
        group_obj = self.connection_handle.create_group(groups, attrs)  # Group /A/B/C as last group
        self.assertEqual(type(group_obj), CpuGroup)
        self.check_group(group_obj, groups, "/A/A/B/A/B/C")
        self.assertNotIn(groups, root.get_links())
        self.assertIn("/A", root.get_links())

    def test_create_subgroup_with_existing_groups(self):
        root = self.connection_handle.get_group(PATH_SPLIT)
        attrs = {"C1": "V1"}

        group_name = "/A"
        A_attrs = {"A1": "V1"}
        A = self.connection_handle.create_group(group_name, A_attrs)
        self.assertEqual(type(A), CpuGroup)
        self.check_group(A, group_name, "/A")
        self.assertIn("/A", root.get_links())
        self.assertDictEqual(A.get_attrs(), A_attrs)
        group_name = "/A/B"
        B = self.connection_handle.create_group(group_name)
        self.assertEqual(type(B), CpuGroup)
        self.check_group(B, group_name, "/A/A/B")
        self.assertIn("/A/B", A.get_links())

        groups = "/A/B/C"  # Expected /A -> /A/B -> /A/B/C
        group_obj = self.connection_handle.create_group(groups, attrs)  # Group /A/B/C as last group
        self.assertEqual(type(group_obj), CpuGroup)
        self.check_group(group_obj, groups, "/A/A/B/A/B/C")
        self.assertNotIn(groups, root.get_links())
        self.assertIn("/A/B/C", B.get_links())

    def test_create_subgroup_with_group(self):
        root = self.connection_handle.get_group(PATH_SPLIT)
        attrs = {"C1": "V1"}
        group_name = "/A"
        A_attrs = {"A1": "V1"}
        A = self.connection_handle.create_group(group_name, A_attrs)
        self.assertEqual(type(A), CpuGroup)
        group_name = "/B/C"
        C = A.create_group(group_name)
        self.assertEqual(type(C), CpuGroup)
        self.check_group(C, "/A/B/C", "/A/A/B/A/B/C")
        self.assertNotIn(C.name, root.get_links())
        self.assertIn("/A/B", A.get_links())
        self.assertNotIn("/A/B/C", A.get_links())
        with self.assertRaises(GroupExistsError):
            group_obj = A.create_group(group_name)

    def test__has_group_object(self):
        groups = "A/B/C"
        self.connection_handle.create_group(groups)
        root = self.connection_handle.get_group(PATH_SPLIT)
        self.assertTrue(root._has_group_object("/A"))
        self.assertTrue(root._has_group_object("/A/B"))
        self.assertTrue(root._has_group_object("/A/B/C"))
        self.assertFalse(root._has_group_object("/A/B/C/D"))
        self.assertFalse(root._has_group_object("/B/C"))

    def test__del_group_object(self):
        groups = "A/B/C"
        self.connection_handle.create_group(groups)
        root = self.connection_handle.get_group(PATH_SPLIT)
        self.assertTrue(root._del_group_object("/A/B/C"))
        with self.assertRaises(rados.ObjectNotFound):
            self.ioctx.read("/A/B/C")
        self.assertFalse(root._del_group_object("/A/B/C"))

        self.assertTrue(root._del_group_object("/A"))
        with self.assertRaises(rados.ObjectNotFound):
            self.ioctx.read("/A")
        self.assertFalse(root._del_group_object("/A"))

        self.assertTrue(root._del_group_object("/A/B"))
        with self.assertRaises(rados.ObjectNotFound):
            self.ioctx.read("/A/B")
        self.assertFalse(root._del_group_object("/A/B"))

    def test_del_group(self):  # TODO: Added del_group with dataset check
        groups = "/A/B/C"
        root = self.connection_handle.get_group(PATH_SPLIT)

        root.create_group(groups)
        root.del_group("/A/B/C")
        self.assertTrue(root.has_group("/A"))
        self.assertTrue(root.has_group("/A/B"))
        self.assertFalse(root.has_group("/A/B/C"))
        with self.assertRaises(GroupNotFoundError):
            root.del_group("/A/B/C")

        root.create_group(groups)
        root.del_group("/A")
        self.assertFalse(root.has_group("/A"))
        self.assertFalse(root.has_group("/A/B"))
        self.assertFalse(root.has_group("/A/B/C"))
        with self.assertRaises(GroupNotFoundError):
            root.del_group("/A/B/C")

    def test_get_group(self):
        root = self.connection_handle.get_group(PATH_SPLIT)
        group_name = "/A/B"
        root.create_group(group_name)
        B = root.get_group("/A/B")
        self.assertEqual(type(B), CpuGroup)
        self.check_group(B, "/A/B", "/A/A/B")
        A = root.get_group("/A")
        self.assertEqual(type(A), CpuGroup)
        self.check_group(A, "/A", "/A")
        with self.assertRaises(GroupNotFoundError):
            root.get_group("/A/B/C")
        B = A.get_group("/A/B")
        self.assertEqual(type(B), CpuGroup)
        self.check_group(B, "/A/B", "/A/A/B")


    def test_get_group_object(self):
        name = "/A"
        root = self.connection_handle.get_group(PATH_SPLIT)
        self.assertEqual(type(root), CpuGroup)
        A = root.create_group(name)
        self.assertEqual(type(A), CpuGroup)
        A_get = root._get_group_object(name)
        self.assertEqual(type(A_get), CephGroup)
        self.check_group(A_get, name, name)
        with self.assertRaises(GroupNotFoundError):  # TODO: Is this check required?
            root._get_group_object("/B")



    def test_get_links(self):
        name = "/A"
        root = self.connection_handle.get_group(PATH_SPLIT)
        root.create_group(name)
        root_link = CpuLink(CephLink("/", "/A", "/->/A"))
        links = {"/A": root_link}
        for link in links:
            self.assertEqual(type(links[link]), type(root.get_links()[link]))
            self.assertEqual(links[link].name, root.get_links()[link].name)
            self.assertEqual(links[link].source, root.get_links()[link].source)
            self.assertEqual(links[link].target, root.get_links()[link].target)

    def test_create_link2grp(self):
        group_a = "/A"
        group_b = "/B"
        root = self.connection_handle.get_group(PATH_SPLIT)
        root.create_group(group_a)
        root.create_group(group_b)
        A = root.get_group(group_a)
        B = root.get_group(group_b)
        self.assertNotIn(group_b, A.get_links())
        A.create_link(group_b)
        self.assertIn(group_b, A.get_links())
        B_through_A = A.get_group("/B")
        self.assertEqual(type(B_through_A), CpuGroup)
        self.assertEqual(B.name, B_through_A.name)
        self.assertEqual(B.absolute_path, B_through_A.absolute_path)
        with self.assertRaises(GroupNotFoundError):
            B.get_group("/A")

    def test_get_link2_del_group(self):
        group_a = "/A"
        group_b = "/B"
        root = self.connection_handle.get_group(PATH_SPLIT)
        root.create_group(group_a)
        root.create_group(group_b)
        A = root.get_group(group_a)
        root.get_group(group_b)
        self.assertNotIn(group_b, A.get_links())
        A.create_link(group_b)
        self.assertIn(group_b, A.get_links())
        A.del_group("/B")
        self.assertEqual(A.get_links()["/B"].target, None)
        with self.assertRaises(GroupNotFoundError):
            A.get_group("/B")
        # Adding "/B" back link should be maintained
        root.create_group(group_b)
        B = root.get_group(group_b)
        self.assertEqual(A.get_group("/B").name, B.name)

    def test_get_attrs(self):
        root = self.connection_handle.get_group(PATH_SPLIT)
        group_name = "/FakeGroup"
        attrs = {"A1": "V1"}
        group_obj = self.connection_handle.create_group(group_name, attrs)
        self.assertEqual(type(group_obj), CpuGroup)
        self.check_group(group_obj, group_name, group_name)
        self.assertEqual(group_obj.get_attrs(), attrs)

    def test_group_create_dataset(self):
        grp = "/A"
        data = np.random.randn(100, 100, 100)
        root = self.connection_handle.get_group(PATH_SPLIT)
        data1 = root.create_dataset("data", data=data, chunk_size=(32, 32, 32))
        path = root.name + "data"
        self.assertEqual(type(data1), CpuDataset)
        self.assertEqual(_SIGNATURE, str(self.ioctx.read(path).decode()))
        self.assertIsNone(assert_array_equal(data, data1[:]))
        data_path = "/dset1"
        dset1 = root.create_dataset(data_path, data=data)
        self.assertEqual(type(dset1), CpuDataset)
        self.assertEqual(_SIGNATURE, str(self.ioctx.read(data_path).decode()))
        self.assertIsNone(assert_array_equal(data, dset1[:]))
        A = root.create_group(grp)
        A_data = A.create_dataset("data", data=data)
        path = A.name + PATH_SPLIT + "data"
        self.assertEqual(type(A_data), CpuDataset)
        self.assertEqual(_SIGNATURE, str(self.ioctx.read(path).decode()))
        self.assertIsNone(assert_array_equal(data, A_data[:]))
        with self.assertRaises(DatasetExistsError):
            A.create_dataset("data", data=data)
        with self.assertRaises(Exception):
            A.create_dataset("data")

        data_path = "/A/dset1"
        A_dset1 = A.create_dataset(data_path, data=data)
        self.assertEqual(type(A_dset1), CpuDataset)
        self.assertEqual(_SIGNATURE, str(self.ioctx.read(data_path).decode()))
        self.assertIsNone(assert_array_equal(data, A_dset1[:]))

    def test__has_dataset_object(self):
        grp = "/A"
        root = self.connection_handle.get_group(PATH_SPLIT)
        A = root.create_group(grp)

        data = np.random.randn(100, 100, 100)
        root.create_dataset("data", data=data, chunk_size=(32, 32, 32))
        path = root.name + "data"
        self.assertTrue(root._has_dataset_object(path))
        self.assertFalse(root._has_dataset_object("/B"))
        self.assertTrue(A._has_dataset_object(path))

    def test_has_dataset(self):
        grp = "/A"
        root = self.connection_handle.get_group(PATH_SPLIT)
        A = root.create_group(grp)

        data = np.random.randn(100, 100, 100)
        root.create_dataset("data", data=data, chunk_size=(32, 32, 32))

        self.assertTrue(root.has_dataset("/data"))
        self.assertFalse(A.has_dataset("/data"))
