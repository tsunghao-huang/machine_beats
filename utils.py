from mido import Message, MidiFile, MidiTrack, MetaMessage, bpm2tempo
import pandas as pd
import numpy as np
import os

class MidiProcessor:
    
    def __init__(self, midi_dir):
        
        self.midi_dir = midi_dir
        self.midi = MidiFile(self.midi_dir)
        self.drum_track_ind = [i for i in range(len(self.midi.tracks)) if self.midi.tracks[i][0].channel == 9][0]
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
        df.time = [round(sum(df.time[0:i])/self.ticks_per_32nt) for i in range(1, len(df)+1)]
        df = df[df.type == 'note_on']
        df = df.pivot_table(index='time', columns='note', values='velocity', fill_value=0)

        # Fill empty notes
        df = df.reindex(pd.RangeIndex(df.index.max()+1)).fillna(0).sort_index()

        # if velocity > 0, change it to 1
        df = (df > 0).astype(int)
        df.columns = df.columns.astype(int)
        
        return df
    
def encode_midi_df(df, loops_len_list=None, max_instruments=None):
    '''
    loops_len_list: if the provided df is a concatenation of several midis, 
        a list of loops length should be provided to segment encoding results
    max_instruments: Some percussion instruments are not that frequently appear, 
        one can set the maximum instruments to lower the complexity.
    '''

    if max_instruments != None:
        most_frequent_inst = sorted(df.sum().to_dict().items(), key=lambda kv: kv[1], reverse=True)
        most_frequent_inst = [instrument[0] for instrument in most_frequent_inst][0:max_instruments]
        df = df[most_frequent_inst]
    
    df = df.reset_index(drop=True)

    columns_v = df.columns.tolist()
    encoding_list = df.values*columns_v
    encoding_list = [list(n[n > 0]) for n in encoding_list]
    unique_note = [list(n) for n in set(tuple(n) for n in encoding_list)]
    encoding_dict = {key: value for key, value in enumerate(unique_note)}

    def encode_by_dict(encoding_dict, value):

        items_list = encoding_dict.items()
        for item in items_list:
            if item[1] == value:
                key = item[0]
                break
        return key

    encoding_result = np.array([encode_by_dict(encoding_dict,n) for n in encoding_list])
    if loops_len_list == None:
        return encoding_result, encoding_dict
    else:
        segment_indices = [sum(loops_len_list[:i]) for i in range(len(loops_len_list) + 1)]
        encoding_result = [encoding_result[segment_indices[i]:segment_indices[i+1]] for i in range(len(segment_indices)-1)]
        return encoding_result, encoding_dict

def array_to_midi(encoding_array, encoding_dict, bpm=120):
    new_song = MidiFile()
    new_song.ticks_per_beat = 240
    meta_track = MidiTrack()
    new_song.tracks.append(meta_track)

    # Create meta_track, add neccessary settings.
    meta_track.append(MetaMessage(type='track_name', name='meta_track', time=0))
    meta_track.append(MetaMessage(type='time_signature', numerator=4, denominator=4, 
                                    clocks_per_click=24, notated_32nd_notes_per_beat=8, time=0))
    meta_track.append(MetaMessage(type='set_tempo', tempo=bpm2tempo(bpm), time=0))

    # drum_track
    drum_track = MidiTrack()
    new_song.tracks.append(drum_track)

    decoding_list = [encoding_dict[i] for i in encoding_array]

    ticks_per_32note = 30
    time_list = []

    for time_index, note in enumerate(decoding_list):
        if len(note) > 0:
            time_list.append(time_index)

            if len(time_list) <= 1:
                notes_from_last_message = 0

            else:
                notes_from_last_message = time_list[-1] - time_list[-2]

            for index, instrument in enumerate(note):
                if index == 0:
                    drum_track.append(Message('note_on', channel=9, note=instrument, velocity=60, 
                                                time=notes_from_last_message*ticks_per_32note))

                else:
                    drum_track.append(Message('note_on', channel=9, note=instrument, velocity=60, time=0))
        else:
            pass
    return new_song

def concat_all_midi_to_df(root_dir = './groove/', return_loops_len_list=True):

    def get_all_midi_dir(root_dir = root_dir):
        all_midi = []
        for dirName, _, fileList in os.walk(root_dir):
            for fname in fileList:
                if '.mid' in fname:
                    all_midi.append(dirName + '/' + fname)
        
        return all_midi
    
    # loop through all the midis in provided root_dir and create df
    df_lists = []
    for file_name in get_all_midi_dir(root_dir = root_dir):
        midiprocessor = MidiProcessor(file_name)
        # numerator = midiprocessor.numerator
        # denominator = midiprocessor.denominator
        # if (numerator == 4) and (denominator == 4):
        df = midiprocessor.midi_to_df()
        df_lists.append(df)
    df = pd.concat(df_lists).fillna(0).astype(int)
    
    loops_len_list = [len(df) for df in df_lists]
    print("{} drum loops".format(len(df_lists)))
    print("{} percussion instruments".format(len(df.columns)))
    print("{} 32-notes".format(len(df)))


    if return_loops_len_list:
        return df, loops_len_list
    else:
        return df

def prepare_input(encoding_loops_list, input_window_len=16, pred_steps = 1, 
                  overlaps = 15, train_test_split=None):
    output_len = pred_steps + overlaps
    X = []
    y = []
    for loop in encoding_loops_list:
        for i in range(len(loop)-input_window_len-pred_steps+1):
            input_start = i
            input_end = i + input_window_len
            output_start = input_end - overlaps
            output_end = output_start + output_len

            X.append(loop[input_start:input_end])
            y.append(loop[output_start:output_end])
    if train_test_split == None:
        return X, y
    else:
        split_point = int(len(X)*train_test_split)
        X_train, y_train = X[:-split_point], y[:-split_point]
        X_test, y_test = X[-split_point:], y[-split_point:]
        return X_train, y_train, X_test, y_test
