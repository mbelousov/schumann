from convert import MidiCollection
from collections import Counter
from operator import itemgetter
import matplotlib.pyplot as plt
import seaborn as sns
from pprint import pprint

if __name__ == '__main__':
    collection = MidiCollection.load_model('music_24-102.json')

    print("Loaded %d music pieces" % collection.num_pieces)

    x = []
    for piece in collection.pieces:
        for state in piece.statematrix:
            for j in xrange(len(state)):
                if state[j][0] == 1:
                    x.append(j)

    loc_lower_bound = min(x)
    loc_upper_bound = max(x)
    print "Lower bound: %d, upper bound: %d" % (
        loc_lower_bound, loc_upper_bound)
    bins = range(collection.lower_bound, collection.upper_bound + 1)
    sns.distplot(x,
                 hist=bins)
    c = Counter(x)
    pprint(sorted(c.items(), key=itemgetter(0)))
    plt.show()
