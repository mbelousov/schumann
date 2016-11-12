import midi
import numpy as np
import logging
import os
from pprint import pprint
from utils import timeit, scan_midifiles
import json


class Serializable(object):
    def to_json(self):
        return json.dumps(self, default=lambda o: o.__dict__, )


class LazyMidiCollection(object):
    def __init__(self, model_name):
        self.model_name = model_name
        self.__read_header()

    def __read_header(self):
        fp = self.__open_model_file()
        line = fp.next().strip()
        lower_bound, upper_bound, num_instances = line.split()
        self.lower_bound = int(lower_bound)
        self.upper_bound = int(upper_bound)
        self.num_instances = int(num_instances)
        fp.close()

    def __open_model_file(self):
        base_path = os.path.dirname(os.path.realpath(__file__))
        dirpath = os.path.join(base_path, 'models')
        model_path = os.path.join(dirpath, self.model_name)
        return open(model_path, 'r')

    def iterpieces(self):
        with self.__open_model_file() as f:
            i = 0
            for line in f:
                i += 1
                if i % 10 == 0:
                    print i
                line = line.strip()
                if i == 1:
                    lower_bound, upper_bound, num_instances = line.split()
                    lower_bound = int(lower_bound)
                    upper_bound = int(upper_bound)
                    num_instances = int(num_instances)
                    continue
                midimatrix = MidiMatrix.from_bin(line, lower_bound, upper_bound)
                yield midimatrix


class MidiCollection(Serializable):
    name = None
    pieces = None

    def __init__(self, name, lower_bound, upper_bound):
        self.name = name
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        if self.pieces is None:
            self.pieces = []

    def add(self, midimatrix):
        self.pieces.append(midimatrix)

    @property
    def num_pieces(self):
        return len(self.pieces)

    @staticmethod
    def from_directory(dirpath, lower_bound, upper_bound):
        collection = MidiCollection(dirpath.replace('/', '_'),
                                    lower_bound, upper_bound)
        pieces = MidiCollection.__load_pieces_from_dir(dirpath,
                                                       lower_bound,
                                                       upper_bound,
                                                       recursive=True)
        collection.pieces = pieces
        return collection

    @staticmethod
    def __load_pieces_from_dir(dirpath, lower_bound, upper_bound,
                               recursive=True, prefix=""):
        """Loads all midi files into dictionary of state-matrices
        """

        # batch_width = 10  # number of sequences in a batch
        # batch_len = 16 * 8  # length of each sequence
        # division_len = 16  # interval between possible start locations
        conv = MIDIConverter(lower_bound=lower_bound, upper_bound=upper_bound)
        pieces = []
        if not os.path.isdir(dirpath):
            base_path = os.path.dirname(os.path.realpath(__file__))
            dirpath = os.path.join(base_path, dirpath)
        for fname in os.listdir(dirpath):
            full_path = os.path.join(dirpath, fname)
            if recursive and os.path.isdir(full_path):
                dir_prefix = os.path.relpath(full_path,
                                             os.path.dirname(full_path))
                if prefix:
                    dir_prefix = prefix + "_" + dir_prefix
                dir_pieces = MidiCollection.__load_pieces_from_dir(full_path,
                                                                   lower_bound,
                                                                   upper_bound,
                                                                   recursive=True,
                                                                   prefix=dir_prefix)
                pieces.extend(dir_pieces)

            name = fname[:-4]
            if prefix:
                name = prefix + "_" + name

            midimatrix = conv.midi2nsmatrix(os.path.join(dirpath, fname))
            if midimatrix is None:
                continue
            pieces.append(midimatrix)
            print "Loaded {}".format(name)

        return pieces

    def save_model(self):
        base_path = os.path.dirname(os.path.realpath(__file__))
        dirpath = os.path.join(base_path, 'models')
        model_name = "%s_%d-%d.json" % (
            self.name, self.lower_bound, self.upper_bound)
        model_path = os.path.join(dirpath, model_name)
        with open(model_path, 'w') as f:
            f.write(self.to_json())

    @staticmethod
    def load_model(model_name):
        base_path = os.path.dirname(os.path.realpath(__file__))
        dirpath = os.path.join(base_path, 'models')

        model_path = os.path.join(dirpath, model_name)
        with open(model_path, 'r') as f:
            obj = json.load(f)
            collection = MidiCollection(name=obj['name'],
                                        lower_bound=obj['lower_bound'],
                                        upper_bound=obj['upper_bound'])
            for piece in obj['pieces']:
                midimatrix = MidiMatrix(name=piece['name'],
                                        lower_bound=piece['lower_bound'],
                                        upper_bound=piece['upper_bound'],
                                        statematrix=piece['statematrix'])
                collection.add(midimatrix)
        return collection


class MidiMatrixBase(Serializable):
    upper_bound = 0
    lower_bound = 0
    name = ''

    def __init__(self, name, lower_bound, upper_bound):
        self.name = name
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound


class MidiMatrix(MidiMatrixBase):
    statematrix = []

    def __init__(self, name, lower_bound, upper_bound, statematrix=None):
        self.name = name
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.statematrix = statematrix

    def to_midi(self, output_file):
        conv = MIDIConverter(lower_bound=self.lower_bound,
                             upper_bound=self.upper_bound)
        conv.nsmatrix2midi(self.statematrix, output_file)

    @staticmethod
    def from_bin(content, lower_bound, upper_bound):
        name, num_states, line = content.split('|')
        num_states = int(num_states)
        num_notes = upper_bound - lower_bound + 1
        states = []
        c = 0
        for i in xrange(num_states):
            state = []
            for j in xrange(num_notes):
                note = [int(line[c]), int(line[c + 1])]
                state.append(note)
                c += 2
            states.append(state)
        return MidiMatrix(name, lower_bound, upper_bound, states)

    @property
    def num_states(self):
        return len(self.statematrix)


class DurationMidiMatrix(MidiMatrixBase):
    durationmatrix = []
    midimatrix = []

    def __init__(self, midimatrix):
        acc = [0 for i in midimatrix.statematrix[0]]
        self.durationmatrix = [
            [0 for j in xrange(len(midimatrix.statematrix[i]))] for i in
            xrange(len(midimatrix.statematrix))]
        for i in xrange(len(midimatrix.statematrix) - 1, 0, -1):
            state = midimatrix.statematrix[i]
            for j in xrange(0, len(state)):
                if state[j][0] == 0:
                    continue
                if state[j][1] == 0:
                    acc[j] = acc[j] + 1
                else:
                    acc[j] = acc[j] + 1
                    self.durationmatrix[i][j] = acc[j]
                    acc[j] = 0
        self.name = midimatrix.name
        self.lower_bound = midimatrix.lower_bound
        self.upper_bound = midimatrix.upper_bound
        self.midimatrix = midimatrix


class MIDIConverter(object):
    def __init__(self, lower_bound=24, upper_bound=102):
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    def midimatrix2bin(self, midimatrix):
        content = "%s|%s|" % (midimatrix.name, midimatrix.num_states)
        for state in midimatrix.statematrix:
            for notes in state:
                content += ("".join([str(n) for n in notes]))
        content += "\n"
        return content

    def directory2bin(self, dirpath,
                      output_file):
        """Loads all midi files into dictionary of state-matrices
        """
        files = scan_midifiles(dirpath)
        fp = open(output_file, 'w')
        fp.write("%d %d %d\n" % (self.lower_bound, self.upper_bound,
                                 len(files)))

        # batch_width = 10  # number of sequences in a batch
        # batch_len = 16 * 8  # length of each sequence
        # division_len = 16  # interval between possible start locations
        for fpath in files:
            fname = fpath.split('/')[-1]
            name = fname[:-4]

            midimatrix = self.midi2nsmatrix(fpath)

            if midimatrix is None:
                continue
            fp.write(self.midimatrix2bin(midimatrix))
            print "Loaded {}".format(name)

        return

    def nsmatrix2midi(self, statematrix, output_file, tickscale=20,
                      velocity=85):
        """
        Converts state-matrix to MIDI File and saves it to the disk
        """
        statematrix = np.asarray(statematrix)
        pattern = midi.Pattern()
        track = midi.Track()
        pattern.append(track)

        span = self.upper_bound - self.lower_bound + 1

        lastcmdtime = 0
        prevstate = [[0, 0] for x in range(span)]
        for time, state in enumerate(statematrix + [prevstate[:]]):
            offNotes = []
            onNotes = []
            for i in range(span):
                n = state[i]
                p = prevstate[i]
                if p[0] == 1:
                    if n[0] == 0:
                        offNotes.append(i)
                    elif n[1] == 1:
                        offNotes.append(i)
                        onNotes.append(i)
                elif n[0] == 1:
                    onNotes.append(i)
            for note in offNotes:
                track.append(
                    midi.NoteOffEvent(tick=(time - lastcmdtime) * tickscale,
                                      pitch=note + self.lower_bound))
                lastcmdtime = time
            for note in onNotes:
                track.append(
                    midi.NoteOnEvent(tick=(time - lastcmdtime) * tickscale,
                                     velocity=velocity,
                                     pitch=note + self.lower_bound))
                lastcmdtime = time

            prevstate = state

        eot = midi.EndOfTrackEvent(tick=1)
        track.append(eot)

        midi.write_midifile(output_file, pattern)

    @timeit
    def midi2nsmatrix(self, midifile, sample_rate=16):
        """Converts MIDI file to state-matrix
        """
        fname = midifile.split('/')[-1]
        if not fname.lower().endswith('.mid'):
            logging.info("Skip %s" % fname)
            return None
            # raise ValueError("Invalid file: %s!" % fname)
        fname = fname[:-4]
        pattern = midi.read_midifile(midifile)
        timeleft = [track[0].tick for track in pattern]
        posns = [0 for track in pattern]

        span = self.upper_bound - self.lower_bound + 1
        time = 0
        logging.info("Span size: %d" % span)
        state = [[0, 0] for x in range(span)]
        # statematrix = np.array([state])
        statematrix = [state]
        logging.info("Pattern resolution: %s" % pattern.resolution)
        logging.info("Timeleft: %s" % timeleft)
        while True:
            # print "TIME: {}".format(time)
            if time % (pattern.resolution / sample_rate) == (
                        pattern.resolution /
                        (sample_rate * 2)):
                # Crossed a note boundary. Create a new state, defaulting to holding notes
                oldstate = state
                state = [[oldstate[x][0], 0] for x in range(span)]
                # statematrix = np.concatenate((statematrix, [np.asarray(state)]))
                statematrix.append(state)
            for i in range(len(timeleft)):
                while timeleft[i] == 0:
                    track = pattern[i]
                    pos = posns[i]
                    # print "POS:%d [%d]" % (pos, timeleft[i])

                    evt = track[pos]
                    if isinstance(evt, midi.NoteEvent):
                        # logging.info("NOTE: %s" % evt)
                        if (evt.pitch < self.lower_bound) or (
                                    evt.pitch >= self.upper_bound):
                            logging.warn("Ignoring note {} at time {}".format(
                                evt.pitch, time))
                            pass
                        else:
                            if isinstance(evt,
                                          midi.NoteOffEvent) or evt.velocity == 0:
                                state[evt.pitch - self.lower_bound] = [0, 0]
                            else:
                                logging.info(
                                    "Note {} at {} (velocity: {})".format(
                                        evt.pitch,
                                        (evt.pitch - self.lower_bound),
                                        evt.velocity))
                                state[evt.pitch - self.lower_bound] = [1, 1]
                    elif isinstance(evt, midi.TimeSignatureEvent):
                        logging.info("TIME SIGN: %s" % evt)

                    try:
                        timeleft[i] = track[pos + 1].tick
                        posns[i] += 1
                    except IndexError:
                        timeleft[i] = None

                if timeleft[i] is not None:
                    timeleft[i] -= 1

            if all(t is None for t in timeleft):
                break

            time += 1

        return MidiMatrix(fname, self.lower_bound, self.upper_bound,
                          statematrix=statematrix)


if __name__ == '__main__':
    logging.basicConfig(level=logging.ERROR)

    converter = MIDIConverter()
    print "Reading file.."

    midimatrix = converter.midi2nsmatrix('music/elise.mid',
                                         sample_rate=16)
    print "States: %d" % midimatrix.num_states

    s_i = 0
    for state in midimatrix.statematrix:
        p_i = 0
        seq = []
        for pair in state:
            if pair != [0, 0]:
                if pair == [1, 0]:
                    seq.append("%d (c)" % p_i)
                else:
                    seq.append(p_i)
            p_i += 1
        print "State %d: [%s]" % (s_i, " ".join([str(e) for e in seq]))
        s_i += 1
    print "Generating the output.."
    # converter.nsmatrix2midi(midimatrix.statematrix, "output.mid")
    midimatrix.to_midi("output.mid")

    print "DONE"
