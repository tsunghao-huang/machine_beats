from utils import MidiProcessor, array_to_midi, concat_all_midi_to_df, prepare_data
from model import build_CNN
import numpy as np
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from keras.utils import np_utils

input_window_len = 128
pred_steps = 1

df, tracks_len_list = concat_all_midi_to_df(root_dir = './LOG_4_4', return_tracks_len_list=True)

X_train, y_train, X_test, y_test, instruments_list = prepare_data(df, input_window_len=input_window_len, pred_steps=1, overlaps=0, train_test_split=0.1,
                                                                  tracks_len_list=tracks_len_list, max_instruments=6)

# Build CNN
input_window_len=X_train.shape[1]
n_feature = X_train.shape[2]
output_shape = y_train.shape[1]

model = build_CNN(input_window_len, n_feature, output_shape)

# train CNN
es = EarlyStopping(monitor='val_loss', patience=20)
mc = ModelCheckpoint(filepath='./CNN_LoG_1st_try.h5', monitor='val_loss', verbose=1,save_best_only=True)
# tb = TensorBoard(log_dir="./logs/CNN_LoG_1st_try_{}".format(time()))
callbacks = [es, mc]

history = model.fit(X_train, y_train, epochs=50, callbacks=callbacks,
                    validation_split=0.1, verbose=1, shuffle=True)