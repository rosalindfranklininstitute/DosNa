import logging
import sys
import unittest
import os
import numpy as np
from numpy.testing import assert_array_equal
import h5py

import dosna as dn
from dosna.tests import configure_logger

from dosna.tools.hdf_converter import HdfConverter
import time
log = logging.getLogger(__name__)

BACKEND = 'ram'
ENGINE = 'cpu'
CONNECTION_CONFIG = {}

DATA_SIZE = (100, 100, 100)
DATA_CHUNK_SIZE = (32, 32, 32)

SEQUENTIAL_TEST_PARTS = 3
DATASET_NUMBER_RANGE = (-100, 100)

H5FILE_NAME = 'test_h5_file.h5'
JSON_FILE_NAME = 'test_json_file.json'

# Hardcode for Ceph access so unit tests are easy to run.

# TODO: add to test_hdf_converter.
class HdfConverterTest(unittest.TestCase):
    """
    Test HDF5 to DosNa methods
    """
    dn_connection = None

    @classmethod
    def setUpClass(cls):
        dn.use(backend="ceph", engine=ENGINE)
        cls.dn_connection = dn.Connection("dosna", conffile="ceph.conf")
        cls.dn_connection.connect()
        cls.hdf_converter = HdfConverter()

    @classmethod
    def tearDownClass(cls):
        cls.dn_connection.disconnect()

    def setUp(self):
        if self.dn_connection.connected == False:
            self.dn_connection = dn.Connection("dosna", conffile="ceph.conf")
            self.dn_connection.connect()

        def create_h5file(filename):
            with h5py.File(filename, "w") as f:
                A = f.create_group("A")
                B = A.create_group("B")
                C = A.create_group("C")
                D = B.create_group("D")

                A.attrs["A1"] = "V1"
                A.attrs["A2"] = "V2"
                C.attrs["C1"] = "C1"

                # NOT CHUNKED
                dset1 = B.create_dataset("dset1", shape=DATA_SIZE)
                data = np.random.randint(DATASET_NUMBER_RANGE[0], DATASET_NUMBER_RANGE[1]+1, DATA_SIZE)
                dset1[...] = data

                data = np.random.randint(DATASET_NUMBER_RANGE[0], DATASET_NUMBER_RANGE[1] + 1, DATA_SIZE)
                dset2 = B.create_dataset("dset2", shape=DATA_SIZE, chunks=DATA_CHUNK_SIZE)
                dset2[...] = data

                f.close()

        create_h5file(H5FILE_NAME)
        self._started_at = time.time()


    def tearDown(self):
        elapsed = time.time() - self._started_at
        print('{} ({}s)'.format(self.id(), round(elapsed, 2)))
        if self.dn_connection.has_group_object('/A'):
            self.dn_connection.del_group('/A')
        self.dn_connection.disconnect()
        if os.path.isfile(H5FILE_NAME):
            os.remove(H5FILE_NAME)
        if os.path.isfile(JSON_FILE_NAME):
            os.remove(JSON_FILE_NAME)

    def compare_datasets_hdf(self, dataset1, dataset2):
        self.assertEqual(dataset1.name, dataset2.name)
        self.assertEqual(dataset1.shape, dataset2.shape)
        self.assertEqual(dataset1.dtype, dataset2.dtype)
        self.assertEqual(dataset1.chunks, dataset2.chunks)
        self.assertEqual(dataset1.nbytes, dataset2.nbytes)
        self.assertEqual(dataset1.ndim, dataset2.ndim)
        self.assertEqual(dataset1.fillvalue, dataset2.fillvalue)
        for d1, d2 in zip(dataset1, dataset2):
            self.assertIsNone(assert_array_equal(d1, d2))
        if dataset1.chunks is not None:
            self.assertEqual(dataset1.chunks, dataset2.chunks)
            for chunk in dataset1.iter_chunks():
                self.assertIsNone(assert_array_equal(dataset1[chunk], dataset2[chunk]))

    def compare_datasets_dosna(self, hdf_dset, dn_dset):
        self.assertEqual(hdf_dset.shape, dn_dset.shape)
        self.assertEqual(hdf_dset.dtype, dn_dset.dtype)
        self.assertEqual(hdf_dset.ndim, dn_dset.ndim)
        self.assertEqual(hdf_dset.fillvalue, dn_dset.fillvalue)
        for hdf, dn in zip(hdf_dset, dn_dset):
            self.assertIsNone(assert_array_equal(hdf, dn))
        if hdf_dset.chunks is not None:
            self.assertEqual(hdf_dset.chunks, dn_dset.chunk_size)
            for chunk in hdf_dset.iter_chunks():
                self.assertIsNone(assert_array_equal(hdf_dset[chunk], dn_dset[chunk]))

    def compare_datasets_json(self, hdf_dset, json_dset):
        self.assertEqual(hdf_dset.shape, json_dset["shape"])
        self.assertEqual(hdf_dset.dtype, json_dset["dtype"])
        self.assertEqual(hdf_dset.ndim, json_dset["ndim"])
        self.assertEqual(hdf_dset.fillvalue, json_dset["fillvalue"])
        self.assertEqual(hdf_dset.chunks, json_dset["chunk_size"])
        for d1, d2 in zip(hdf_dset, json_dset['dataset_value']):
            self.assertIsNone(assert_array_equal(d1, d2))


    def test_hdf2dict(self):
        hdf_dict = self.hdf_converter.hdf2dict(H5FILE_NAME)
        hdf_file = h5py.File(H5FILE_NAME)
        hdf_file_attrs = dict(attr for attr in hdf_file['A'].attrs.items())
        self.assertDictEqual(hdf_file_attrs, hdf_dict['A']['attrs'])
        hdf_file_attrs = dict(attr for attr in hdf_file['A']['B'].attrs.items())
        self.assertEqual(hdf_file_attrs, hdf_dict['A']['B']['attrs'])
        hdf_file_attrs = dict(attr for attr in hdf_file['A']['B']['D'].attrs.items())
        self.assertEqual(hdf_file_attrs, hdf_dict['A']['B']['D']['attrs'])
        hdf_file_attrs = dict(attr for attr in hdf_file['A']['C'].attrs.items())
        self.assertEqual(hdf_file_attrs, hdf_dict['A']['C']['attrs'])

        hdf_file_dset1 = hdf_file['A']['B']['dset1']
        hdf_dict_dset1 = hdf_dict['A']['B']['dset1']
        self.compare_datasets_hdf(hdf_file_dset1, hdf_dict_dset1)

        hdf_file_dset2 = hdf_file['A']['B']['dset2']
        hdf_dict_dset2 = hdf_dict['A']['B']['dset2']
        self.compare_datasets_hdf(hdf_file_dset2, hdf_dict_dset2)

        hdf_file.close()

    def test_hdf2dosna(self):
        dn_cluster = self.hdf_converter.hdf2dosna(H5FILE_NAME, self.dn_connection)
        hdf_file = h5py.File(H5FILE_NAME)
        hdf_file_attrs = dict(attr for attr in hdf_file['/A'].attrs.items())
        self.assertDictEqual(hdf_file_attrs, dn_cluster['/A'].get_attrs())
        hdf_file_attrs = dict(attr for attr in hdf_file['/A']['/A/B'].attrs.items())
        self.assertEqual(hdf_file_attrs, dn_cluster['/A']['/A/B'].get_attrs())
        hdf_file_attrs = dict(attr for attr in hdf_file['/A']['/A/B']['/A/B/D'].attrs.items())
        self.assertEqual(hdf_file_attrs, dn_cluster['/A']['/A/B']['/A/B/D'].get_attrs())
        hdf_file_attrs = dict(attr for attr in hdf_file['/A']['/A/C'].attrs.items())
        self.assertEqual(hdf_file_attrs, dn_cluster['/A']['/A/C'].get_attrs())

        hdf_file_dset1 = hdf_file['/A']['/A/B']['/A/B/dset1']
        dn_cluster_dset1 = dn_cluster['/A']['/A/B']['/A/B/dset1']
        self.compare_datasets_dosna(hdf_file_dset1, dn_cluster_dset1)

        hdf_file_dset2 = hdf_file['/A']['/A/B']['/A/B/dset2']
        dn_cluster_dset2 = dn_cluster['/A']['/A/B']['/A/B/dset2']
        self.compare_datasets_dosna(hdf_file_dset2, dn_cluster_dset2)

        hdf_file.close()

    def test_hdf2json(self):
        json_dict = self.hdf_converter.hdf2json(H5FILE_NAME, JSON_FILE_NAME)
        hdf_file = h5py.File(H5FILE_NAME)
        hdf_file_attrs = dict(attr for attr in hdf_file['A'].attrs.items())
        self.assertDictEqual(hdf_file_attrs, json_dict['A']['attrs'])
        hdf_file_attrs = dict(attr for attr in hdf_file['A']['B'].attrs.items())
        self.assertEqual(hdf_file_attrs, json_dict['A']['B']['attrs'])
        hdf_file_attrs = dict(attr for attr in hdf_file['A']['B']['D'].attrs.items())
        self.assertEqual(hdf_file_attrs, json_dict['A']['B']['D']['attrs'])
        hdf_file_attrs = dict(attr for attr in hdf_file['A']['C'].attrs.items())
        self.assertEqual(hdf_file_attrs, json_dict['A']['C']['attrs'])

        hdf_file_dset1 = hdf_file['A']['B']['dset1']
        json_dict_dset1 = json_dict['A']['B']['dset1']
        self.compare_datasets_json(hdf_file_dset1, json_dict_dset1)

        hdf_file_dset2 = hdf_file['A']['B']['dset2']
        json_dict_dset2 = json_dict['A']['B']['dset2']
        self.compare_datasets_json(hdf_file_dset2, json_dict_dset2)

        hdf_file.close()

def main():
    unittest.main(verbosity=2)


if __name__ == "__main__":
    main()
