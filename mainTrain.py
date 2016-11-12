import logging
from convert import MIDIConverter,DurationMidiMatrix,MidiMatrix,MidiCollection
from Train import TrainData
from utils import object_dump, object_load
if __name__ == '__main__':
    logging.basicConfig(level=logging.WARN)

    converter = MidiCollection("music", lower_bound =21, upper_bound= 108)
    pieces = converter.from_directory('music', lower_bound =21, upper_bound= 108)
    target = open("training.txt", 'w')
    for piece in pieces.pieces:
        durMat = DurationMidiMatrix(piece)
        trainData = TrainData(durMat,context_length=10,step=1)
        print durMat.name + " "+str(durMat.lower_bound)+" "+str(durMat.upper_bound)
        #for met in durMat.durationmatrix:
        #    print met
        #
        for x_line in range(0,len(trainData.x)):
            target.write("x:"+str(trainData.x[x_line])+'\r\n')
            target.write("y:"+str(trainData.y[x_line])+'\r\n')
        #
    target.close()
    print("Loaded %d music pieces" % len(pieces.pieces))