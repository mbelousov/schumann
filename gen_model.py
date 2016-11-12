from convert import MIDIConverter, MidiCollection, MidiMatrix

if __name__ == '__main__':
    lower_bound = int(raw_input("Lower bound: "))
    upper_bound = int(raw_input("Upper bound: "))
    conv = MIDIConverter(lower_bound=lower_bound, upper_bound=upper_bound)
    fname = 'models/music-db_%d-%d.bin' % (lower_bound, upper_bound)
    conv.directory2bin('music-db',
                       fname)




    # collection = MidiCollection.from_directory(
    #     'music',
    #     lower_bound=lower_bound,
    #     upper_bound=upper_bound)
    # print("Loaded %d music pieces" % collection.num_pieces)
    # collection.to_file()
    # # collection.save_model()
    # print "Model saved."
