
class TrainData(object):
    x = []
    y = []
    def __init__(self,durationMatrix,context_length,step=1):
        curr = 0
        for i in xrange(0,len(durationMatrix.durationmatrix),step):
            context = []
            for j in xrange(0,context_length):
                context.append(durationMatrix.durationmatrix[i+j])
            self.x.append(context)
            self.y.append(durationMatrix.durationmatrix[i+context_length])
