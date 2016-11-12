import time
import logging
import os


def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()

        logging.debug('%r (%r, %r) %2.2f sec' % \
                      (method.__name__, args, kw, te - ts))
        return result

    return timed


def scan_midifiles(dirpath, recursive=True, prefix=""):
    files = []
    if not os.path.isdir(dirpath):
        base_path = os.path.dirname(os.path.realpath(__file__))
        dirpath = os.path.join(base_path, dirpath)
    for fname in os.listdir(dirpath):
        full_path = os.path.join(dirpath, fname)
        if recursive and os.path.isdir(full_path):
            dir_prefix = os.path.relpath(full_path,
                                         os.path.dirname(full_path))
            if prefix:
                dir_prefix = prefix + "_" + dir_prefix
            local_files = scan_midifiles(full_path,
                                         recursive=True,
                                         prefix=dir_prefix)
            files.extend(local_files)

        if fname.lower().endswith('.mid'):
            files.append(os.path.join(dirpath, fname))
    return files
