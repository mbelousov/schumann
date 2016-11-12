import logging
from convert import MIDIConverter
from utils import object_dump, object_load
if __name__ == '__main__':
    logging.basicConfig(level=logging.WARN)

    converter = MIDIConverter()
    pieces = converter.from_directory('music-db', recursive=True)
    converter.save_model(pieces, 'music-db')
    pieces = converter.from_model('music-db')
    print("Loaded %d music pieces" % len(pieces))
