from utils import MidiProcessor, prepare_data, array_to_midi, concat_all_midi_to_df
from model import build_stateless_lstm
import numpy as np
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from keras.utils import np_utils

input_window_len = 32
pred_steps = 1

df, tracks_len_list = concat_all_midi_to_df(root_dir = './LOG_4_4', return_tracks_len_list=True)

X_train, y_train, X_test, y_test, instruments_list = prepare_data(df, input_window_len=input_window_len, pred_steps=1, overlaps=0, train_test_split=0.1,
                                                                  tracks_len_list=tracks_len_list, max_instruments=6)

n_feature = X_train.shape[2]
output_shape = y_train.shape[1]
model = build_stateless_lstm(input_window_len=input_window_len, n_feature=n_feature, output_shape=output_shape, dropout=0.5, neurons_per_layer=[32,32,32])

# Experiment with different batch_size
# batch_size = 256
es = EarlyStopping(monitor='val_loss', patience=20)
mc = ModelCheckpoint(filepath='./new_encode_1st_try.h5', monitor='val_loss', verbose=1,save_best_only=True)
#tb = TensorBoard(log_dir="./logs/lstm_{}".format(time()))
callbacks = [es, mc]

history = model.fit(X_train, y_train, epochs=50, callbacks=callbacks, 
                    validation_split=0.1, verbose=1, shuffle=False)