from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.layers import LSTM, GRU
from keras.optimizers import RMSprop
import numpy as np

from convert import LazyMidiCollection, DurationMidiMatrix


class TrainData(object):
    x = []
    y = []

    def __init__(self, durationMatrix, context_length, step=1):
        self.x = []
        self.y = []
        for i in xrange(0, len(durationMatrix.duration_matrix) - context_length,
                        step):
            context = []
            for j in xrange(0, context_length):
                context.append(durationMatrix.duration_matrix[i + j])
            self.x.append(context)
            self.y.append(durationMatrix.duration_matrix[i + context_length])


def normalise(x, y):
    upper = max([note for window in x for state in window for note
                 in state])
    norm_x = x
    norm_y = y
    for i in xrange(len(x)):
        for j in xrange(len(x[i])):
            norm_x[i][j] = [1.0 * norm_x[i][j][k] / upper for k in
                            xrange(len(norm_x[i][j]))]
    for j in xrange(len(y)):
        norm_y[j] = [1.0 * norm_y[j][k] / upper for k in
                     xrange(len(norm_y[j]))]
    return norm_x, norm_y, upper


def unnormalise(x, upper):
    conv_x = []
    for state in x:
        result_state = []
        for note in state:
            result = 0
            if note > 0:
                result = int(note * upper)
            result_state.append(result)
        conv_x.append(result_state)
    return conv_x


def generateMelody(model, startSequence, addLength):
    completeSequence = startSequence
    length = len(startSequence)
    for n in range(addLength):
        lastElems = completeSequence[-length:]
        prediction = model.predict(np.array([lastElems]))
        completeSequence.append(prediction[0].tolist())
    return completeSequence


if __name__ == '__main__':
    collection_file = raw_input("Collection file: ")
    nb_epochs = int(raw_input("Number of epochs: "))
    batch_size = int(raw_input("Batch size: "))
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
    u_startSequence = X[0]
    X, y, norm_upper = normalise(X, y)

    startSequence = X[0]
    m = DurationMidiMatrix('start', lower_bound=collection.lower_bound,
                           upper_bound=collection.upper_bound,
                           duration_matrix=unnormalise(startSequence, norm_upper))

    m.to_midi('start.mid')

    n_notes = collection.upper_bound - collection.lower_bound + 1
    l_subsequence = context_length
    n_examples = len(y)
    input_shape = (l_subsequence, n_notes)
    print("Num examples: %d" % n_examples)
    print('Build model...')
    model = Sequential()
    model.add(GRU(num_nodes, input_shape=input_shape))
    model.add(Dense(n_notes))
    model.add(Activation('tanh'))

    optimizer = RMSprop(lr=0.01)
    model.compile(loss='mse', optimizer=optimizer)

    # for i in range(100):
    #
    model.fit(X, y, batch_size=batch_size, nb_epoch=nb_epochs, verbose=2)
    model.save_weights('weights/model_weights.h5')

    melody = generateMelody(model, startSequence, 16 * context_length)

    m = DurationMidiMatrix('output', lower_bound=collection.lower_bound,
                           upper_bound=collection.upper_bound,
                           duration_matrix=unnormalise(melody, norm_upper))

    m.to_midi('output.mid')
