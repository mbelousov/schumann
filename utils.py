import time
import logging
import msgpack


def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()

        logging.debug('%r (%r, %r) %2.2f sec' % \
                      (method.__name__, args, kw, te - ts))
        return result

    return timed


def object_dump(obj, fpath):
    data = msgpack.packb(obj)
    with open(fpath, 'wb') as f:
        f.write(data)


def object_load(fpath):
    with open(fpath, 'rb') as f:
        return msgpack.unpack(f)
