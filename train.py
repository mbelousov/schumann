import os
from convert import midi2nsmatrix
import logging

batch_width = 10  # number of sequences in a batch
batch_len = 16 * 8  # length of each sequence
division_len = 16  # interval between possible start locations


def loadMusicPieces(dirpath, recursive=True, prefix=""):
    pieces = {}

    for fname in os.listdir(dirpath):
        full_path = os.path.join(dirpath, fname)
        if recursive and os.path.isdir(full_path):
            dir_prefix = os.path.relpath(full_path,
                                     os.path.dirname(full_path))
            if prefix:
                dir_prefix = prefix + "_" + dir_prefix
            dir_pieces = loadMusicPieces(full_path, recursive=True,
                                         prefix=dir_prefix)
            pieces.update(dir_pieces)

        if fname[-4:] not in ('.mid', '.MID'):
            continue

        name = fname[:-4]
        if prefix:
            name = prefix + "_" + name

        statematrix = midi2nsmatrix(os.path.join(dirpath, fname))
        if len(statematrix) < batch_len:
            continue
        if name in pieces.keys():
            logging.error("Piece with name %s is already exist! replacing.")

        pieces[name] = statematrix
        print "Loaded {}".format(name)

    return pieces


if __name__ == '__main__':
    m = loadMusicPieces('music')
