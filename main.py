import logging
from convert import nsmatrxi2midi, midi2nsmatrix
from train import loadMusicPieces
if __name__ == '__main__':
    logging.basicConfig(level=logging.WARN)

    pieces = loadMusicPieces('music')
    print("Loaded %d music pieces" % len(pieces))



