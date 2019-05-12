from mido import Message, MidiFile, MidiTrack, MetaMessage, bpm2tempo
import pandas as pd
import numpy as np
import os


class MidiProcessor:

    def __init__(self, midi_dir):

        self.midi_dir = midi_dir
        self.midi = MidiFile(self.midi_dir)
        self.drum_track_ind = [i for i in range(
            len(self.midi.tracks)) if self.midi.tracks[i][0].channel == 9][0]
#         self.meta_track = MidiFile(self.midi_dir).tracks[0]
        self.drum_track = self.midi.tracks[self.drum_track_ind]
#         self.meta_ind = [i for i, m in enumerate(self.meta_track) if 'numerator' in m.dict()][0]
#         self.numerator = self.meta_track[self.meta_ind].numerator
#         self.denominator = self.meta_track[self.meta_ind].denominator
        self.ticks_per_beat = self.midi.ticks_per_beat
        self.ticks_per_32nt = self.ticks_per_beat/8

    def midi_to_df(self):

        df = pd.DataFrame([m.dict() for m in self.drum_track])

        # get time passed since the first message and quantize
        df.time = [round(sum(df.time[0:i])/self.ticks_per_32nt)
                   for i in range(1, len(df)+1)]
        df = df[df.type == 'note_on']
        df = df.pivot_table(index='time', columns='note',
                            values='velocity', fill_value=0)

        # Fill empty notes
        df = df.reindex(pd.RangeIndex(df.index.max()+1)).fillna(0).sort_index()

        # if velocity > 0, change it to 1
        df = (df > 0).astype(int)
        df.columns = df.columns.astype(int)

        return df


def prepare_data(df, input_window_len=32, pred_steps=1, overlaps=0, train_test_split=None,
                 tracks_len_list=None, max_instruments=None):
    '''
    tracks_len_list: if the provided df is a concatenation of several midis, 
        a list of tracks length should be provided to segment encoding results
    max_instruments: Some percussion instruments are not that frequently appear, 
        one can set the maximum instruments to lower the complexity.
    '''

    # choose top max_instruments
    if max_instruments != None:
        most_frequent_inst = sorted(
            df.sum().to_dict().items(), key=lambda kv: kv[1], reverse=True)
        most_frequent_inst = [instrument[0]
                              for instrument in most_frequent_inst][0:max_instruments]
        df = df[most_frequent_inst]
    df = df.reset_index(drop=True)
    # remember the encoding scheme
    instruments = df.columns.tolist()

    def split_tracks(df_values, tracks_len_list=tracks_len_list):

        segment_indices = [sum(tracks_len_list[:i])
                           for i in range(len(tracks_len_list) + 1)]
        encoded_tracks_list = [df_values[segment_indices[i]:segment_indices[i+1], :]
                               for i in range(len(segment_indices)-1)]

        return encoded_tracks_list

    def get_Xy(df_values, input_window_len=input_window_len,
               pred_steps=pred_steps, overlaps=overlaps):
        X = []
        y = []
        for i in range(len(df_values)-input_window_len-pred_steps+1):
            input_start = i
            input_end = i + input_window_len
            output_start = input_end - overlaps
            output_end = output_start + overlaps + pred_steps

            X.append(df_values[input_start:input_end])
            y.append(df_values[output_start:output_end])

        y = list(np.array(y).reshape(-1, np.array(y).shape[-1]))
        return X, y

    if tracks_len_list == None:
        X, y = get_Xy(df.values)
    else:
        encoded_tracks_list = split_tracks(
            df.values, tracks_len_list=tracks_len_list)
        X_list = []
        y_list = []
        for track in encoded_tracks_list:
            X, y = get_Xy(track)
            X_list.append(X)
            y_list.append(y)
        X = np.concatenate(X_list, axis=0)
        y = np.concatenate(y_list, axis=0)

    if train_test_split == None:
        return X, y, instruments
    else:
        split_point = int(len(X)*train_test_split)
        X_train, y_train = X[:-split_point], y[:-split_point]
        X_test, y_test = X[-split_point:], y[-split_point:]
        return X_train, y_train, X_test, y_test, instruments


def array_to_midi(encoding_array, instruments_list, bpm=180):
    new_song = MidiFile()
    new_song.ticks_per_beat = 960
    meta_track = MidiTrack()
    new_song.tracks.append(meta_track)

    # Create meta_track, add neccessary settings.
    meta_track.append(MetaMessage(
        type='track_name', name='meta_track', time=0))
    meta_track.append(MetaMessage(type='time_signature', numerator=4, denominator=4,
                                  clocks_per_click=24, notated_32nd_notes_per_beat=8, time=0))
    meta_track.append(MetaMessage(type='set_tempo',
                                  tempo=bpm2tempo(bpm), time=0))

    # drum_track
    drum_track = MidiTrack()
    new_song.tracks.append(drum_track)

    ticks_per_32note = 120

    time_indices = []
    for i, note in enumerate(encoding_array*instruments_list):
        if sum(note) == 0:
            pass
        else:
            time_indices.append(i)

            if len(time_indices) <= 1:
                notes_from_last_message = 0
            else:
                notes_from_last_message = time_indices[-1] - time_indices[-2]

            same_note_count = 0
            for inst in note:

                if inst == 0:
                    pass
                elif same_note_count == 0:
                    drum_track.append(Message('note_on', channel=9, note=inst, velocity=80,
                                              time=notes_from_last_message*ticks_per_32note))
                    same_note_count += 1
                else:
                    drum_track.append(Message('note_on', channel=9, note=inst, velocity=80,
                                              time=0))
                    same_note_count += 1
    return new_song


def concat_all_midi_to_df(root_dir, return_tracks_len_list=True):

    def get_all_midi_dir(root_dir=root_dir):
        all_midi = []
        for dirName, _, fileList in os.walk(root_dir):
            for fname in fileList:
                if '.mid' in fname:
                    all_midi.append(dirName + '/' + fname)

        return all_midi

    # loop through all the midis in provided root_dir and create df
    df_lists = []
    for file_name in get_all_midi_dir(root_dir=root_dir):
        midiprocessor = MidiProcessor(file_name)
        # numerator = midiprocessor.numerator
        # denominator = midiprocessor.denominator
        # if (numerator == 4) and (denominator == 4):
        df = midiprocessor.midi_to_df()
        df_lists.append(df)
    df = pd.concat(df_lists).fillna(0).astype(int)

    tracks_len_list = [len(df) for df in df_lists]
    print("{} drum loops".format(len(df_lists)))
    print("{} percussion instruments".format(len(df.columns)))
    print("{} 32-notes".format(len(df)))

    if return_tracks_len_list:
        return df, tracks_len_list
    else:
        return df
