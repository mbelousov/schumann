from convert import LazyMidiCollection, DurationMidiMatrix


class TrainData(object):
    x = []
    y = []

    def __init__(self, durationMatrix, context_length, step=1):
        x = []
        y = []
        for i in xrange(0, len(durationMatrix.duration_matrix) - context_length,
                        step):
            context = []
            for j in xrange(0, context_length):
                context.append(durationMatrix.duration_matrix[i + j])
            self.x.append(context)
            self.y.append(durationMatrix.duration_matrix[i + context_length])
        self.norm_x, self.norm_y = self.normalise(self.x, self.y)

    def normalise(self, x, y):
        upper = max([note for window in x for state in window for note in \
                     state])
        norm_x = x
        norm_y = y
        for i in xrange(len(x)):
            for j in xrange(len(x[i])):
                norm_x[i][j] = [1.0 * norm_x[i][j][k] / upper for k in
                                xrange(len(norm_x[i][j]))]
        for j in xrange(len(y)):
            norm_y[j] = [1.0 * norm_y[j][k] / upper for k in
                         xrange(len(norm_y[j]))]
        return norm_x, norm_y


if __name__ == '__main__':

    from keras.models import Sequential
    from keras.layers import Dense, Activation, Dropout
    from keras.layers import LSTM, GRU
    from keras.optimizers import RMSprop
    import numpy as np

    context_length = 128

    collection = LazyMidiCollection('music_21-108.bin')
    X = []
    y = []
    for piece in collection.iterpieces():
        durMat = DurationMidiMatrix.from_midimatrix(piece)
        trainData = TrainData(durMat, context_length=context_length, step=1)
        X.extend(trainData.norm_x)
        y.extend(trainData.norm_y)

    n_notes = collection.upper_bound - collection.lower_bound + 1
    l_subsequence = context_length
    examples = len(y)
    input_shape = (l_subsequence, n_notes)

    print('Build model...')
    model = Sequential()
    model.add(GRU(10, input_shape=input_shape))
    model.add(Dense(n_notes))
    model.add(Activation('tanh'))

    optimizer = RMSprop(lr=0.01)
    model.compile(loss='mse', optimizer=optimizer)


    def generateMelody(model, startSequence, addLength):
        completeSequence = startSequence
        length = len(startSequence)
        for n in range(addLength):
            lastElems = completeSequence[-length:]
            prediction = model.predict(np.array([lastElems]))
            completeSequence.append(prediction[0].tolist())
        return completeSequence


    # for i in range(100):
    #
    model.fit(X, y, batch_size=1, nb_epoch=5, verbose=2)

    startSequence = X[0]
    melody = generateMelody(model, startSequence, 2 * context_length)
    print(melody)
