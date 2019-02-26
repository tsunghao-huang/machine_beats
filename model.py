from keras.models import Sequential
from keras.layers import Dense, Dropout, Embedding
from keras.layers import LSTM
from keras.utils import np_utils
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard

def build_stateless_lstm(input_window_len, n_feature, output_shape, dropout=0.3, neurons_per_layer=None):
    model = Sequential()

    if neurons_per_layer == None:
        model.add(LSTM(input_window_len, input_shape=(input_window_len, n_feature), dropout=dropout))
    elif len(neurons_per_layer) == 1:
        model.add(LSTM(neurons_per_layer[0], input_shape=(input_window_len, n_feature), dropout=dropout))
    elif len(neurons_per_layer) > 1:
        for i, neurons in enumerate(neurons_per_layer):
            if i == 0:
                model.add(LSTM(neurons_per_layer[0], input_shape=(input_window_len, n_feature), return_sequences=True, dropout=dropout))
            else:
                model.add(LSTM(neurons, return_sequences=True, dropout=dropout))
    model.add(Dense(output_shape, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam')

    return model