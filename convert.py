import midi, numpy
import logging
from pprint import pprint

lowerBound = 24
upperBound = 102


def noteStateMatrixToMidi(statematrix, name="example", tickscale=20):
    statematrix = numpy.asarray(statematrix)
    pattern = midi.Pattern()
    track = midi.Track()
    pattern.append(track)

    span = upperBound - lowerBound

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
                                  pitch=note + lowerBound))
            lastcmdtime = time
        for note in onNotes:
            track.append(midi.NoteOnEvent(tick=(time - lastcmdtime) * tickscale,
                                          velocity=85, pitch=note + lowerBound))
            lastcmdtime = time

        prevstate = state

    eot = midi.EndOfTrackEvent(tick=1)
    track.append(eot)

    midi.write_midifile("{}.mid".format(name), pattern)


def midiToNoteStateMatrix(midifile, sample_rate=16):
    pattern = midi.read_midifile(midifile)
    timeleft = [track[0].tick for track in pattern]
    posns = [0 for track in pattern]

    statematrix = []
    span = upperBound - lowerBound
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
                    if (evt.pitch < lowerBound) or (evt.pitch >= upperBound):
                        logging.warn("Ignoring note {} at time {}".format(
                            evt.pitch, time))
                        pass
                    else:
                        if isinstance(evt,
                                      midi.NoteOffEvent) or evt.velocity == 0:
                            state[evt.pitch - lowerBound] = [0, 0]
                        else:
                            print "Note {} at {} (velocity: {})".format(
                                evt.pitch,
                                (evt.pitch - lowerBound),
                                evt.velocity)
                            state[evt.pitch - lowerBound] = [1, 1]
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
            print "BREAK"
            break

        time += 1

    return statematrix


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    statematrix = midiToNoteStateMatrix('music/mond_1.mid')
    print "Result"
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

    # raw_input()
    # pprint(statematrix)
    noteStateMatrixToMidi(statematrix, "output")
    print "DONE"
