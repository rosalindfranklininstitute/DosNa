import time
import dosna as dn
import unittest
from dosna.backends.base import (
    DatasetNotFoundError,
    GroupNotFoundError,
    GroupExistsError,
)

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

    def setUp(self):
        if self.connection_handle.connected == False:
            self.connection_handle = dn.Connection("dosna", conffile="ceph.conf")
            self.connection_handle.connect()
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
