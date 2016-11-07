import os
from convert import midi2nsmatrix

batch_width = 10  # number of sequences in a batch
batch_len = 16 * 8  # length of each sequence
division_len = 16  # interval between possible start locations


def loadMusicPieces(dirpath):
    pieces = {}

    for fname in os.listdir(dirpath):
        if fname[-4:] not in ('.mid', '.MID'):
            continue

        name = fname[:-4]

        statematrix = midi2nsmatrix(os.path.join(dirpath, fname))
        if len(statematrix) < batch_len:
            continue

        pieces[name] = statematrix
        print "Loaded {}".format(name)

    return pieces

if __name__ == '__main__':
    m = loadMusicPieces('music')
