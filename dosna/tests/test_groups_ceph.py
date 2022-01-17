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
        self.check_group(A, name, name, attrs)

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
        self.assertNotIn(group_name, root.links)  # TODO: make links a private field.
        self.assertIn(group_name, root.get_links())

    def test_create_existing_group(self):
        root = self.connection_handle.get_group(PATH_SPLIT)
        group_name = "/FakeGroup"
        group_obj = self.connection_handle.create_group(group_name)
        with self.assertRaises(GroupExistsError):
            group_obj = self.connection_handle.create_group(group_name)


    def test_create_group_with_attrs(self):
        root = self.connection_handle.get_group(PATH_SPLIT)
        group_name = "/FakeGroup"
        attrs = {"A1": "V1"}
        group_obj = self.connection_handle.create_group(group_name, attrs)
        self.assertEqual(type(group_obj), CpuGroup)
        self.check_group(group_obj, group_name, group_name, attrs)
        self.assertEqual(type(root), CpuGroup)
        self.assertNotIn(group_name, root.links)
        self.assertIn(group_name, root.get_links())

    def test_create_subgroups(self):
        root = self.connection_handle.get_group(PATH_SPLIT)
        attrs = {"C1": "V1"}
        groups = "/A/B/C"  # Expected /A -> /A/B -> /A/B/C
        group_obj = self.connection_handle.create_group(groups, attrs)  # Group /A/B/C as last group
        self.assertEqual(type(group_obj), CpuGroup)
        self.check_group(group_obj, groups, "/A/A/B/A/B/C", attrs)
        self.assertNotIn(groups, root.get_links())
        self.assertIn("/A", root.get_links())
        # Check /A doesn't have attrs of /A/B/C
        self.assertNotEqual(root.get_links()["/A"].target.attrs, attrs)
        # Check /A/B doesn't have attrs of /A/B/C
        self.assertNotEqual(root.get_links()["/A"].target.get_links()["/A/B"].target.attrs, attrs)

    def test_create_subgroup_with_existing_groups(self):
        root = self.connection_handle.get_group(PATH_SPLIT)
        attrs = {"C1": "V1"}

        group_name = "/A"
        A_attrs = {"A1": "V1"}
        A = self.connection_handle.create_group(group_name, A_attrs)
        self.assertEqual(type(A), CpuGroup)
        self.check_group(A, group_name, "/A", A_attrs)
        self.assertIn("/A", root.get_links())
        self.assertDictEqual(A.attrs, A_attrs)
        group_name = "/A/B"
        B = self.connection_handle.create_group(group_name)
        self.assertEqual(type(B), CpuGroup)
        self.check_group(B, group_name, "/A/A/B", )
        self.assertIn("/A/B", A.get_links())

        groups = "/A/B/C"  # Expected /A -> /A/B -> /A/B/C
        group_obj = self.connection_handle.create_group(groups, attrs)  # Group /A/B/C as last group
        self.assertEqual(type(group_obj), CpuGroup)
        self.check_group(group_obj, groups, "/A/A/B/A/B/C", attrs)
        self.assertNotIn(groups, root.get_links())
        self.assertIn("/A/B/C", B.get_links())
        # Check /A doesn't have attrs of /A/B/C
        self.assertNotEqual(root.get_links()["/A"].target.attrs, attrs)
        self.assertEqual(root.get_links()["/A"].target.attrs, A_attrs)
        # Check /A/B doesn't have attrs of /A/B/C
        self.assertNotEqual(root.get_links()["/A"].target.get_links()["/A/B"].target.attrs, attrs)
