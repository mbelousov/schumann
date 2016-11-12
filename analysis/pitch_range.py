from convert import LazyMidiCollection, MidiMatrix
from collections import Counter
from operator import itemgetter
import matplotlib.pyplot as plt
import seaborn as sns
from pprint import pprint
import os
import json

if __name__ == '__main__':
    print "Reading.."
    x = []
    collection = LazyMidiCollection('music_21-108.bin')
    for midimatrix in collection.iterpieces():
        for state in midimatrix.statematrix:
            for j in xrange(len(state)):
                if state[j][0] == 1:
                    x.append(j + collection.lower_bound)

    loc_lower_bound = min(x)
    loc_upper_bound = max(x)

    print "Local lower bound: %d, Local upper bound: %d" % (
        loc_lower_bound, loc_upper_bound)

    bins = range(collection.lower_bound,
                 collection.upper_bound + 1)
    sns.distplot(x,
                 hist=bins)
    c = Counter(x)
    pprint(sorted(c.items(), key=itemgetter(0)))
    plt.show()
