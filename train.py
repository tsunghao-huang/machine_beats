from utils import MidiProcessor, encode_midi_df, array_to_midi, concat_all_midi_to_df, prepare_input
from model import build_stateless_lstm
import numpy as np
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from keras.utils import np_utils

input_window_len=128
pred_steps = 1

df, loops_len_list = concat_all_midi_to_df(root_dir = './LOG_4_4', return_loops_len_list=True)
encoding_result, encoding_dict = encode_midi_df(df, loops_len_list=loops_len_list, max_instruments=4)

X_train, y_train, X_test, y_test = prepare_input(encoding_result, input_window_len=input_window_len,
                                                pred_steps = pred_steps, 
                                                overlaps = 0, train_test_split=0.2)

nunique_instruments = len(encoding_dict)
X_train = np.reshape(X_train, (len(X_train), input_window_len, 1))
X_train = X_train / float(nunique_instruments)
y_train = np_utils.to_categorical(y_train)

output_shape = y_train.shape[1]
model = build_stateless_lstm(input_window_len=input_window_len, n_feature=1, output_shape=output_shape, dropout=0.3, neurons_per_layer=[32,32,32,32])

batch_size = 256
es = EarlyStopping(monitor='val_loss', patience=20)
mc = ModelCheckpoint(filepath='1st_try.h5', monitor='val_loss', verbose=1,save_best_only=True)
#tb = TensorBoard(log_dir="/content/gdrive/My Drive/Colab Notebooks/logs/lstm_{}".format(time()))
callbacks = [es, mc]

history = model.fit(X_train, y_train, epochs=50, callbacks=callbacks, 
                    batch_size=batch_size, validation_split=0.1, verbose=1, shuffle=False)