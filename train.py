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
    for n in range(len(startSequence)):
        model.predict(np.array([startSequence[n]]))

    for n in range(addLength):
        lastElems = completeSequence[-1]
        prediction = model.predict(np.array([lastElems]))
        completeSequence.append(prediction.tolist())
    return completeSequence


if __name__ == '__main__':
    default_collection_file = 'music_8_21-108.bin'
    collection_file = raw_input(
        "Collection file[%s]: " % default_collection_file)
    if collection_file == "":
        collection_file = default_collection_file
    nb_epochs = int(raw_input("Number of epochs: "))
    # batch_size = int(raw_input("Batch size: "))
    num_nodes = int(raw_input("Number of nodes: "))
    collection = LazyMidiCollection(collection_file)
    # context_length = collection.sample_rate * 4 * 2
    start_sequence_length = collection.sample_rate * 4 * 2
    context_length = 1
    batch_size = 1
    # print "Context: %d" % context_length

    X = []
    y = []
    pieces_idx = []
    for piece in collection.iterpieces():
        durMat = DurationMidiMatrix.from_midimatrix(piece)
        trainData = TrainData(durMat, context_length=context_length, step=1)
        X.extend(trainData.x)
        y.extend(trainData.y)
        pieces_idx.append(len(y))
    X, y, norm_upper = normalise(X, y)

    print np.array(X).shape
    startSequence = X[:start_sequence_length]
    print np.array(startSequence).shape
    nStartSeq = [x[0] for x in startSequence]
    m = DurationMidiMatrix('start', lower_bound=collection.lower_bound,
                           upper_bound=collection.upper_bound,
                           duration_matrix=unnormalise(nStartSeq,
                                                       norm_upper))

    m.to_midi('start.mid')

    n_notes = collection.upper_bound - collection.lower_bound + 1
    l_subsequence = context_length
    n_examples = len(y)
    input_shape = (l_subsequence, n_notes)
    print("Num examples: %d" % n_examples)
    print('Build model...')
    model = Sequential()
    model.add(GRU(num_nodes, batch_input_shape=(1, l_subsequence, n_notes),
                  stateful=True))
    model.add(Dense(n_notes))
    model.add(Activation('tanh'))

    optimizer = RMSprop(lr=0.01)
    model.compile(loss='mse', optimizer=optimizer)

    # for i in range(100):
    #
    for epoch in xrange(nb_epochs):
        old_idx = 0
        for idx in pieces_idx:
            model.fit(X[old_idx:idx], y[old_idx:idx], batch_size=batch_size,
                      nb_epoch=1,
                      shuffle=False,
                      verbose=2)
            old_idx = idx
        model.save_weights('weights/model_weights_%d.h5' % (epoch + 1))

    model.save_weights('weights/model_weights.h5')

    melody = generateMelody(model, startSequence, 2 * start_sequence_length)
    nMelody = [x[0] for x in melody]
    m = DurationMidiMatrix('output', lower_bound=collection.lower_bound,
                           upper_bound=collection.upper_bound,
                           duration_matrix=unnormalise(nMelody, norm_upper))

    m.to_midi('output.mid')
