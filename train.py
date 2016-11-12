class TrainData(object):
    x = []
    y = []

    def __init__(self, durationMatrix, context_length, step=1):
        for i in xrange(0, len(durationMatrix.duration_matrix) - context_length,
                        step):
            context = []
            for j in xrange(0, context_length):
                context.append(durationMatrix.duration_matrix[i + j])
            self.x.append(context)
            self.y.append(durationMatrix.duration_matrix[i + context_length])
