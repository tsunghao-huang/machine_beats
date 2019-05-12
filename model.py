from keras.models import Sequential
from keras.layers import Dense, Dropout, Embedding, LSTM, Conv1D, MaxPooling1D, Flatten
from keras.utils import np_utils


def build_stateless_lstm(input_window_len, n_feature, output_shape, dropout=0.5, neurons_per_layer=None):
    model = Sequential()

    if neurons_per_layer == None:
        model.add(LSTM(input_window_len, input_shape=(
            input_window_len, n_feature), dropout=dropout))
    elif len(neurons_per_layer) == 1:
        model.add(LSTM(neurons_per_layer[0], input_shape=(
            input_window_len, n_feature), dropout=dropout))
    elif len(neurons_per_layer) > 1:
        for i, neurons in enumerate(neurons_per_layer):
            if i == 0:
                model.add(LSTM(neurons_per_layer[0], input_shape=(
                    input_window_len, n_feature), return_sequences=True, dropout=dropout))
            elif i == len(neurons_per_layer) - 1:
                model.add(LSTM(neurons, dropout=dropout))
            else:
                model.add(LSTM(neurons, return_sequences=True, dropout=dropout))
    model.add(Dense(output_shape, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam')

    return model


def build_CNN(input_window_len, n_feature, output_shape):

    model = Sequential()
    model.add(Conv1D(filters=32, kernel_size=32, activation='relu',
                     input_shape=(input_window_len, n_feature)))
    model.add(MaxPooling1D(pool_size=2))
#   model.add(Conv1D(filters=32, kernel_size=16, activation='relu'))
#   model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(16, activation='relu'))
    model.add(Dense(output_shape, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam')

    return model
