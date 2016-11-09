import midi, numpy
import logging
from pprint import pprint

lower_bound = 24
upper_bound = 102


def nsmatrxi2midi(statematrix, name="example", tickscale=20):
    """
    Converts NoteStateMatrix to MIDI File and saves it to the disk
    :param statematrix: NoteStateMatrix
    :param name: Name of the result file
    :param tickscale: Tick scale (default 20)
    :return:
    """

    statematrix = numpy.asarray(statematrix)
    pattern = midi.Pattern()
    track = midi.Track()
    pattern.append(track)

    span = upper_bound - lower_bound

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
                                  pitch=note + lower_bound))
            lastcmdtime = time
        for note in onNotes:
            track.append(midi.NoteOnEvent(tick=(time - lastcmdtime) * tickscale,
                                          velocity=85,
                                          pitch=note + lower_bound))
            lastcmdtime = time

        prevstate = state

    eot = midi.EndOfTrackEvent(tick=1)
    track.append(eot)

    midi.write_midifile("{}.mid".format(name), pattern)


def midi2nsmatrix(midifile, sample_rate=16):
    """Converts MIDI file to NoteStateMatrix

    :param midifile: Path to MIDI File
    :param sample_rate: Sample rate (default 16)
    :return: NoteStateMatrix
    """
    pattern = midi.read_midifile(midifile)
    timeleft = [track[0].tick for track in pattern]
    posns = [0 for track in pattern]

    statematrix = []
    span = upper_bound - lower_bound
    time = 0
    logging.info("Span size: %d" % span)
    state = [[0, 0] for x in range(span)]
    statematrix.append(state)
    logging.info("Pattern resolution: %s" % pattern.resolution)
    logging.info("Timeleft: %s" % timeleft)
    while True:
        # print "TIME: {}".format(time)
        if time % (pattern.resolution / sample_rate) == (pattern.resolution /
                                                             (sample_rate * 2)):
            # Crossed a note boundary. Create a new state, defaulting to holding notes
            oldstate = state
            state = [[oldstate[x][0], 0] for x in range(span)]
            # logging.info("APPEND STATE %s" % state)
            # print "CROSSED"
            statematrix.append(state)
        for i in range(len(timeleft)):
            while timeleft[i] == 0:
                track = pattern[i]
                pos = posns[i]
                # print "POS:%d [%d]" % (pos, timeleft[i])

                evt = track[pos]
                if isinstance(evt, midi.NoteEvent):
                    # logging.info("NOTE: %s" % evt)
                    if (evt.pitch < lower_bound) or (evt.pitch >= upper_bound):
                        logging.warn("Ignoring note {} at time {}".format(
                            evt.pitch, time))
                        pass
                    else:
                        if isinstance(evt,
                                      midi.NoteOffEvent) or evt.velocity == 0:
                            state[evt.pitch - lower_bound] = [0, 0]
                        else:
                            logging.info("Note {} at {} (velocity: {})".format(
                                evt.pitch,
                                (evt.pitch - lower_bound),
                                evt.velocity))
                            state[evt.pitch - lower_bound] = [1, 1]
                elif isinstance(evt, midi.TimeSignatureEvent):
                    logging.info("TIME SIGN: %s" % evt)
                    # if evt.numerator not in (2, 4):
                    #     # We don't want to worry about non-4 time signatures. Bail early!
                    #     # print "Found time signature event {}. Bailing!".format(evt)
                    #     logging.error("Invalid signature event {}".format(evt))
                    #     return statematrix

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

    return statematrix


if __name__ == '__main__':
    logging.basicConfig(level=logging.ERROR)
    print "Reading file.."
    statematrix = midi2nsmatrix('music/mozart_turkish_march.mid',
                                sample_rate=16)
    print "States: %d" % len(statematrix)

    s_i = 0
    for state in statematrix:
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
    nsmatrxi2midi(statematrix, "output")

    print "DONE"
