from __future__ import print_function
import sys
import os
import errno


def mkdir_p(path):
    """
    It creates directory recursively if it does not already exist
    """
    try:
        os.makedirs(path)
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise
