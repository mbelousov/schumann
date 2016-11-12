import logging
from convert import LazyMidiCollection, DurationMidiMatrix, MidiMatrix
from train import TrainData

if __name__ == '__main__':
    logging.basicConfig(level=logging.WARN)
    target = open("output/training.txt", 'w')
    collection = LazyMidiCollection('music_16_21-108.bin')
    for piece in collection.iterpieces():
        durMat = DurationMidiMatrix.from_midimatrix(piece)
        trainData = TrainData(durMat, context_length=10, step=1)
        print durMat.name + " " + str(durMat.lower_bound) + " " + str(
            durMat.upper_bound)
        # for met in durMat.duration_matrix:
        #    print met
        #
        for x_line in range(0, len(trainData.x)):
            target.write("x:" + str(trainData.x[x_line]) + '\r\n')
            target.write("y:" + str(trainData.y[x_line]) + '\r\n')
            #
    target.close()
    print("Loaded %d music pieces" % collection.num_instances)



