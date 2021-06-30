from keras.utils import plot_model
from input import get_data
from keras.models import Input, Model, load_model
from keras.layers import Conv1D, MaxPooling1D, Dropout, Dense, Flatten, LSTM, RepeatVector, Permute, Lambda, Multiply, \
    CuDNNLSTM
import keras.backend as K
import numpy as np
import matplotlib.pyplot as plt
from keras import optimizers

data, scaled = get_data()  # 电、冷、热
print("data.shape, scaled.shape: ", data.shape, scaled.shape)

# 划分训练集和测试集
data_train = scaled[: int(len(data) * 0.8)]
data_test = scaled[int(len(data) * 0.8):]
seq_len = 24
X_train = np.array([data_train[i: i + seq_len, :] for i in range(data_train.shape[0] - seq_len)])
y_train = np.array([data_train[i + seq_len, 0] for i in range(data_train.shape[0] - seq_len)])
X_test = np.array([data_test[i: i + seq_len, :] for i in range(data_test.shape[0] - seq_len)])
y_test = np.array([data_test[i + seq_len, 0] for i in range(data_test.shape[0] - seq_len)])
print("X_train.shape, y_train.shape, X_test.shape, y_test.shape: ",
      X_train.shape, y_train.shape, X_test.shape, y_test.shape)
# (2920, 24, 3) (2920,) (712, 24, 3) (712,)

SINGLE_ATTENTION_VECTOR = False


def attention_block(inputs):
    """
    加入注意力机制
    :param inputs: 网络上一层的输出
    :return: 返回加入注意力机制的特征层
    """
    input_dim = int(inputs.shape[2])
    a = inputs
    a = Dense(input_dim, activation='softmax')(a)
    if SINGLE_ATTENTION_VECTOR:
        a = Lambda(lambda x: K.mean(x, axis=1), name='dim_reduction')(a)
        a = RepeatVector(input_dim)(a)
    a_probs = Permute((1, 2), name='attention_vec')(a)
    output_attention_mul = Multiply()([inputs, a_probs])
    return output_attention_mul


def cnn_model():
    time_steps = 24
    input_dim = 3
    epoch = 300
    batch_size = 48

    inputs = Input(shape=(time_steps, input_dim))
    hidden_1 = Conv1D(filters=64, kernel_size=1, activation='relu', name="cnn1")(inputs)
    hidden_2 = MaxPooling1D(pool_size=3)(hidden_1)
    print(hidden_2.shape)
    hidden_2 = Flatten()(hidden_2)
    hidden_3 = Dropout(0.2)(hidden_2)

    output = Dense(1, activation='sigmoid')(hidden_3)

    model = Model(inputs, output)
    model.summary()

    model.compile(loss='mae', optimizer='adam')
    plot_model(model, to_file='model.png')
    model.fit(X_train, y_train, epochs=epoch, batch_size=batch_size, shuffle=False)

    middle = Model(inputs, model.get_layer('cnn1').output)

    return model, middle


def lstm_model():
    time_steps = 24
    input_dim = 3
    epoch = 5
    batch_size = 48
    # model = Sequential()
    # model.add(LSTM(48, input_shape=(time_steps, input_dim), return_sequences=True, name='lstm1'))
    # model.add(Dropout(0.3))
    # model.add(Dense(1, activation='sigmoid'))

    inputs = Input(shape=(time_steps, input_dim))
    hidden_1 = LSTM(48, return_sequences=True, name='lstm1')(inputs)
    dropout_1 = Dropout(0.3)(hidden_1)
    hidden_2 = LSTM(24, return_sequences=True, name='lstm2')(dropout_1)
    dropout_2 = Dropout(0.3)(hidden_2)
    # x = Dense(1, activation='sigmoid', name='middle')(dropout_2)
    flat = Flatten()(dropout_2)
    output = Dense(1, activation='sigmoid')(flat)

    lstm = Model(inputs, output)

    opt = optimizers.Adam(lr=0.001)
    lstm.compile(loss='mae', optimizer=opt)
    lstm.fit(X_train, y_train, epochs=epoch, batch_size=batch_size, shuffle=False)

    middle = Model(inputs, lstm.get_layer('lstm2').output)

    return lstm, middle


def attention_model():
    time_steps = 24
    input_dim = 3
    lstm_units = 64
    epoch = 50
    batch_size = 48

    inputs = Input(shape=(time_steps, input_dim))

    x = Conv1D(filters=64, kernel_size=1, activation='relu', name='cnn1')(inputs)
    # padding = 'same'
    x = Dropout(0.3)(x)

    # lstm_out = Bidirectional(LSTM(lstm_units, activation='relu'), name='bilstm')(x)
    # 对于GPU可以使用CuDNNLSTM
    # lstm_out = Bidirectional(LSTM(lstm_units, return_sequences=True))(x)
    lstm_out = CuDNNLSTM(lstm_units, return_sequences=True, name='lstm')(x)
    lstm_out = Dropout(0.3)(lstm_out)
    attention_mul = attention_block(lstm_out)
    attention_mul = Flatten()(attention_mul)

    output = Dense(1, activation='sigmoid')(attention_mul)
    model = Model(inputs=[inputs], outputs=output)
    model.summary()

    opt = optimizers.Adam(lr=0.001)
    model.compile(loss='mae', optimizer=opt)
    plot_model(model, to_file='attention_model.png')
    model.fit(X_train, y_train, epochs=epoch, batch_size=batch_size, shuffle=False)

    middle = Model(inputs, model.get_layer('attention_vec').output)
    model.save("./attention_model.h5")
    middle.save("./attention.h5")
    return model, middle


def get_activations(model, inputs, print_shape_only=False, layer_name=None):
    # Documentation is available online on Github at the address below.
    # From: https://github.com/philipperemy/keras-visualize-activations
    print('----- activations -----')
    activations = []
    inp = model.input
    if layer_name is None:
        outputs = [layer.output for layer in model.layers]
    else:
        outputs = [layer.output for layer in model.layers if layer.name == layer_name]  # all layer outputs
    funcs = [K.function([inp] + [K.learning_phase()], [out]) for out in outputs]  # evaluation functions
    layer_outputs = [func([inputs, 1.])[0] for func in funcs]
    for layer_activations in layer_outputs:
        activations.append(layer_activations)
        if print_shape_only:
            print(layer_activations.shape)
        else:
            print('shape为', layer_activations.shape)
            print(layer_activations)
    return activations


# model, mid = attention_model()
# model = load_model('attention_model.h5')
# feature = mid.predict(X_test)
# print(feature)
# print("feature.shape: ", feature.shape)

# attention_vector = np.mean(get_activations(model, X_test, print_shape_only=False, layer_name='attention_vec')[0],
#                            axis=2).squeeze()
# print('------attention_vector-----')
# print(attention_vector)
# print("attention_vector.shape: ", attention_vector.shape)

mid = load_model('attention.h5')
weight = mid.predict(X_test)
print("weight.shape: ", weight.shape)  # (712, 24, 64)
print(weight)

# 取第一组测试数据注意力机制的结果
attention_vector = np.mean(weight[0, :, :], axis=0)
print('attention =', attention_vector)

# 画权重柱状图
plt.bar(range(len(attention_vector)), attention_vector, width=3, label='weight')
plt.legend()
plt.show()
