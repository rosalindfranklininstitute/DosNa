#!/usr/bin/env python
"""Helper functions to store and get the selected backend"""

import logging

from dosna.utils import named_module

log = logging.getLogger(__name__)

_current = None
AVAILABLE = ['ram', 'hdf5', 'ceph']


def use_backend(backend):
    backend = backend.lower()
    global _current
    if backend in AVAILABLE:
        module_ = named_module('dosna.backends.{}'.format(backend))
        if hasattr(module_, '_backend'):
            log.debug('Switching backend to `%s`', module_._backend.name)
            _current = module_._backend
        else:
            raise Exception(
                'Module `{}` is not a proper backend.'.format(backend))
    else:
        raise Exception('Backend `{}` not available! Choose from: {}'
                        .format(backend, AVAILABLE))


def get_backend(name=None):
    if name is not None:
        use_backend(name)
    if _current is None:
        use_backend(AVAILABLE[0])
    return _current
