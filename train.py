from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.layers import LSTM, GRU
from keras.optimizers import RMSprop, Adam
import numpy as np
from convert import LazyMidiCollection, DurationMidiMatrix, MidiMatrix
import math


class TrainData(object):
    x = []
    y = []

    def __init__(self, x, y):
        self.x = x
        self.y = y

    @staticmethod
    def from_duration_matrix(durationMatrix, context_length, step=1):
        x = []
        y = []
        for i in xrange(0, len(durationMatrix.duration_matrix) - context_length,
                        step):
            context = []
            for j in xrange(0, context_length):
                context.append(durationMatrix.duration_matrix[i + j])
            x.append(context)
            y.append(durationMatrix.duration_matrix[i + context_length])
        return TrainData(x, y)

    @staticmethod
    def from_midimatrix(midimatrix, context_length, step=1):
        x = []
        y = []
        for i in xrange(0, len(midimatrix.statematrix) - context_length,
                        step):
            context = []
            for j in xrange(0, context_length):
                context.append([a[0] for a in midimatrix.statematrix[i + j]])
            x.append(context)
            y.append([a[0] for a in midimatrix.statematrix[i + context_length]])
        return TrainData(x, y)


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
        melody = prediction.tolist()
        completeSequence.append([int(math.ceil(m)) for m in melody[0]])
        # completeSequence.append(prediction[0].tolist())
    return completeSequence


# def generateMelody(model, startSequence, addLength):
#     completeSequence = startSequence
#     for n in range(len(startSequence)):
#         model.predict(np.array([startSequence[n]]))
#
#     for n in range(addLength):
#         lastElems = completeSequence[-1]
#         prediction = model.predict(np.array([lastElems]))
#         completeSequence.append(prediction.tolist())
#         # melody = prediction.tolist()
#         # completeSequence.append([[int(round(m)) for m in melody[0]]])
#     return completeSequence



if __name__ == '__main__':
    default_collection_file = 'music_8_21-108.bin'
    collection_file = raw_input(
        "Collection file[%s]: " % default_collection_file)
    if collection_file == "":
        collection_file = default_collection_file
    num_nodes = int(raw_input("Number of nodes: "))
    weight_file = raw_input("Weight file: ")
    nb_epochs = int(raw_input("Number of epochs: "))

    # batch_size = int(raw_input("Batch size: "))
    collection = LazyMidiCollection(collection_file)
    # context_length = collection.sample_rate * 4 * 2
    start_sequence_length = collection.sample_rate * 4 * 2
    context_length = collection.sample_rate * 4 * 2

    # context_length = 1
    batch_size = 10
    # print "Context: %d" % context_length

    X = []
    y = []
    pieces_idx = []
    for piece in collection.iterpieces():
        # durMat = DurationMidiMatrix.from_midimatrix(piece)
        # trainData = TrainData(durMat, context_length=context_length, step=1)
        trainData = TrainData.from_midimatrix(piece,
                                              context_length=context_length,
                                              step=1)
        X.extend(trainData.x)
        y.extend(trainData.y)
        pieces_idx.append(len(y))
    # X, y, norm_upper = normalise(X, y)
    # startSequence = list(X[:start_sequence_length])
    startSequence = X[0]
    nStartSeq = [list(state) for state in startSequence]
    # nStartSeq = list([list(x[0]) for x in startSequence])
    for i in xrange(len(nStartSeq)):
        for j in xrange(len(nStartSeq[i])):
            if nStartSeq[i][j] == 1:
                nStartSeq[i][j] = [nStartSeq[i][j], 1]
            else:
                nStartSeq[i][j] = [nStartSeq[i][j], 0]
    m = MidiMatrix('start', lower_bound=collection.lower_bound,
                   upper_bound=collection.upper_bound,
                   # duration_matrix=unnormalise(nStartSeq, norm_upper),
                   statematrix=nStartSeq)
    #
    m.to_midi('start.mid')

    n_notes = collection.upper_bound - collection.lower_bound + 1
    l_subsequence = context_length
    n_examples = len(y)
    input_shape = (l_subsequence, n_notes)
    print("Num examples: %d" % n_examples)
    print('Build model...')
    model = Sequential()
    # model.add(GRU(num_nodes, batch_input_shape=(1, l_subsequence, n_notes),
    #               stateful=True))
    model.add(GRU(num_nodes, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(Dense(n_notes))
    model.add(Activation('relu'))
    # model.add(Activation('linear'))

    # optimizer = RMSprop(lr=0.01)
    optimizer = Adam()
    model.compile(loss='mse', optimizer=optimizer)

    if weight_file != "":
        model.load_weights('weights/' + weight_file)

    # for epoch in xrange(nb_epochs):
    #     old_idx = 0
    #     for idx in pieces_idx:
    #         model.fit(X[old_idx:idx], y[old_idx:idx], batch_size=batch_size,
    #                   nb_epoch=1,
    #                   verbose=2)
    #         old_idx = idx
    #         model.reset_states()
    #         model.save_weights('weights/model_weights_%d.h5' % (epoch + 1))
    #     print "Saved epoch-%d" % (epoch + 1)
    model.fit(X, y, batch_size=batch_size, nb_epoch=nb_epochs, verbose=2)
    if nb_epochs > 0:
        model.save_weights('weights/model_weights.h5')

    melody = generateMelody(model, startSequence, 50 * start_sequence_length)
    nMelody = melody
    # nMelody = [x[0] for x in melody]
    # nMelody = [int(note) for state in nMelody for note in state]
    # for state in nMelody[len(startSequence):]:
    #     for note in state:
    #         if int(round(note)) > 0:
    #             print "%.2f => %d" % (note, int(round(note)))
    # nMelody = [[int(round(note)) for note in state] for state in nMelody]
    for i in xrange(len(nMelody)):
        for j in xrange(len(nMelody[i])):
            signal = int(round(nMelody[i][j]))
            # print "%.5f => %d" % (nMelody[i][j], signal)
            if signal == 1:
                nMelody[i][j] = [signal, 1]
            else:
                nMelody[i][j] = [signal, 0]

    m = MidiMatrix('output', lower_bound=collection.lower_bound,
                   upper_bound=collection.upper_bound,
                   # duration_matrix=unnormalise(nMelody, norm_upper),
                   # duration_matrix=nMelody
                   statematrix=nMelody
                   )

    m.to_midi('output.mid')
    print "Done."
