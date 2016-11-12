from convert import MIDIConverter, MidiCollection, MidiMatrix

if __name__ == '__main__':
    sample_rate = 16
    collection_name = 'music'

    lower_bound = int(raw_input("Lower bound: "))
    upper_bound = int(raw_input("Upper bound: "))
    conv = MIDIConverter(lower_bound=lower_bound, upper_bound=upper_bound)
    fname = 'models/%s_%d_%d-%d.bin' % (collection_name, sample_rate,
                                        lower_bound, upper_bound)

    conv.directory2bin(collection_name, fname, sample_rate=sample_rate)




    # collection = MidiCollection.from_directory(
    #     'music',
    #     lower_bound=lower_bound,
    #     upper_bound=upper_bound)
    # print("Loaded %d music pieces" % collection.num_pieces)
    # collection.to_file()
    # # collection.save_model()
    # print "Model saved."
