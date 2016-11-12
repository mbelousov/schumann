from convert import MIDIConverter, MidiCollection

if __name__ == '__main__':
    lower_bound = int(raw_input("Lower bound: "))
    upper_bound = int(raw_input("Upper bound: "))
    # pieces = converter.from_directory('music-db', recursive=True)

    collection = MidiCollection.from_directory('music',
                                               lower_bound=lower_bound,
                                               upper_bound=upper_bound)
    print("Loaded %d music pieces" % collection.num_pieces)
    collection.save_model()
    print "Model saved."
