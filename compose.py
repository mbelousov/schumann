from keras.models import Sequential
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.layers import LSTM, GRU
from keras.optimizers import RMSprop
import numpy as np
from convert import LazyMidiCollection, DurationMidiMatrix

from train import TrainData, normalise, unnormalise, generateMelody
if __name__ == '__main__':
    collection_file = raw_input("Collection file: ")
    num_nodes = int(raw_input("Number of nodes: "))
    collection = LazyMidiCollection(collection_file)
    context_length = collection.sample_rate * 4 * 2
    print "Context: %d" % context_length

    X = []
    y = []
    for piece in collection.iterpieces():
        durMat = DurationMidiMatrix.from_midimatrix(piece)
        trainData = TrainData(durMat, context_length=context_length, step=1)
        X.extend(trainData.x)
        y.extend(trainData.y)

    X, y, norm_upper = normalise(X, y)
    n_notes = collection.upper_bound - collection.lower_bound + 1
    l_subsequence = context_length
    n_examples = len(y)
    input_shape = (l_subsequence, n_notes)

    model = Sequential()
    model.add(GRU(num_nodes, input_shape=input_shape))
    model.add(Dense(n_notes))
    model.add(Activation('tanh'))

    optimizer = RMSprop(lr=0.01)
    model.compile(loss='mse', optimizer=optimizer)

    model.load_weights('weights/model_weights.h5')
    startSequence = X[0]
    melody = generateMelody(model, startSequence, 16 * context_length)
    startSequence = X[0]
    m = DurationMidiMatrix('start', lower_bound=collection.lower_bound,
                           upper_bound=collection.upper_bound,
                           duration_matrix=unnormalise(startSequence,
                                                       norm_upper))

    m.to_midi('start.mid')

    melody = generateMelody(model, startSequence, 16 * context_length)

    m = DurationMidiMatrix('output', lower_bound=collection.lower_bound,
                           upper_bound=collection.upper_bound,
                           duration_matrix=unnormalise(melody, norm_upper))

    m.to_midi('output.mid')

    for state in melody:
        print state
        raw_input()
