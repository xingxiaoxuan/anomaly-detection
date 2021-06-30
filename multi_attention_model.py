from input import get_data
import numpy as np
from keras.layers import Conv1D, MaxPooling1D, Dropout, Dense, Flatten, LSTM, RepeatVector, Permute, Lambda, Multiply, \
    CuDNNLSTM, Concatenate
from keras.models import Input, Model, load_model
import keras.backend as K
from keras.utils import plot_model
import matplotlib.pyplot as plt

data, scaled = get_data()  # 电、冷、热
print("data.shape, scaled.shape: ", data.shape, scaled.shape)
# (3680, 3) (3680, 3)

# 划分训练集和测试集
data_train = scaled[: int(len(data) * 0.8)]
data_test = scaled[int(len(data) * 0.8):]
seq_len = 24
X_train = np.array([data_train[i: i + seq_len, :] for i in range(data_train.shape[0] - seq_len)])
y_train = np.array([data_train[i + seq_len, :] for i in range(data_train.shape[0] - seq_len)])
X_test = np.array([data_test[i: i + seq_len, :] for i in range(data_test.shape[0] - seq_len)])
y_test = np.array([data_test[i + seq_len, :] for i in range(data_test.shape[0] - seq_len)])
print("X_train.shape, y_train.shape, X_test.shape, y_test.shape: ",
      X_train.shape, y_train.shape, X_test.shape, y_test.shape)
# (2920, 24, 3) (2920, 3) (712, 24, 3) (712, 3)
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


def elec_model():
    time_steps = 24
    input_dim = 1
    lstm_units = 48
    epoch = 200
    batch_size = 48

    input_tensor = Input(shape=(time_steps, input_dim), name='elec_input')
    cnn_out = Conv1D(filters=64, kernel_size=1, activation='relu', name="elec_cnn")(input_tensor)
    hidden_1 = Dropout(0.3, name='elec_dropout1')(cnn_out)
    lstm_out = CuDNNLSTM(lstm_units, return_sequences=True, name='elec_feature')(hidden_1)
    hidden_2 = Dropout(0.3, name='elec_dropout2')(lstm_out)
    hidden_3 = Flatten(name='elec_flatten')(hidden_2)
    output = Dense(1, activation='sigmoid', name='elec_dense')(hidden_3)

    model = Model(input_tensor, output)
    model.summary()

    model.compile(loss='mae', optimizer='adam')
    # plot_model(model, to_file='model.png')
    model.fit(X_train[:, :, 0].reshape((2920, 24, 1)), y_train[:, 0], epochs=epoch, batch_size=batch_size, shuffle=False)

    middle = Model(input_tensor, model.get_layer('elec_feature').output)

    model.save("./elec.h5")
    middle.save("./elec_feature.h5")
    return model, middle


# # 单独训练电负荷模型，提取电负荷深层特征
# model, middle = elec_model()
# model = load_model("./elec.h5")
# model.summary()
# middle = load_model("./elec_feature.h5")
# elec_feature = middle.predict(X_test[0, :, 0].reshape((1, 24, 1)))
# print("------------------------")
# print("elec_feature.shape: ", elec_feature.shape)  # (712, 24, 48)
# print(elec_feature)
#
# # weight_Dense_1 = model.get_layer('elec_feature').get_weights()
# # weight_Dense_1 = np.array(weight_Dense_1)
# # print("weight_Dense_1[0].shape: ", weight_Dense_1[0].shape)
# # print("weight_Dense_1: ", weight_Dense_1)
# # # print("bias_Dense_1: ", bias_Dense_1)


def cool_model():
    time_steps = 24
    input_dim = 1
    lstm_units = 48
    epoch = 200
    batch_size = 48

    input_tensor = Input(shape=(time_steps, input_dim), name='cool_input')
    cnn_out = Conv1D(filters=64, kernel_size=1, activation='relu', name="cool_cnn")(input_tensor)
    hidden_1 = Dropout(0.3, name='cool_dropout1')(cnn_out)
    lstm_out = CuDNNLSTM(lstm_units, return_sequences=True, name='cool_feature')(hidden_1)
    hidden_2 = Dropout(0.3, name='cool_dropout2')(lstm_out)
    hidden_3 = Flatten(name='cool_flatten')(hidden_2)
    output = Dense(1, activation='sigmoid', name='cool_dense')(hidden_3)

    model = Model(input_tensor, output)
    model.summary()

    model.compile(loss='mae', optimizer='adam')
    # plot_model(model, to_file='model.png')
    model.fit(X_train[:, :, 1].reshape((2920, 24, 1)), y_train[:, 1], epochs=epoch, batch_size=batch_size, shuffle=False)

    middle = Model(input_tensor, model.get_layer('cool_feature').output)

    model.save("./cool.h5")
    middle.save("./cool_feature.h5")
    return model, middle


def steam_model():
    time_steps = 24
    input_dim = 1
    lstm_units = 48
    epoch = 200
    batch_size = 48

    input_tensor = Input(shape=(time_steps, input_dim), name='steam_input')
    cnn_out = Conv1D(filters=64, kernel_size=1, activation='relu', name="steam_cnn")(input_tensor)
    hidden_1 = Dropout(0.3, name='steam_dropout1')(cnn_out)
    lstm_out = CuDNNLSTM(lstm_units, return_sequences=True, name='steam_feature')(hidden_1)
    hidden_2 = Dropout(0.3, name='steam_dropout2')(lstm_out)
    hidden_3 = Flatten(name='steam_flatten')(hidden_2)
    output = Dense(1, activation='sigmoid', name='steam_dense')(hidden_3)

    model = Model(input_tensor, output)
    model.summary()

    model.compile(loss='mae', optimizer='adam')
    # plot_model(model, to_file='model.png')
    model.fit(X_train[:, :, 2].reshape((2920, 24, 1)), y_train[:, 2], epochs=epoch, batch_size=batch_size, shuffle=False)

    middle = Model(input_tensor, model.get_layer('steam_feature').output)

    model.save("./steam.h5")
    middle.save("./steam_feature.h5")
    return model, middle


# 模型融合
def multi_attention_Model():
    epoch = 50
    batch_size = 48
    elec_extract = load_model('./elec.h5')
    cool_extract = load_model('./cool.h5')
    steam_extract = load_model('./steam.h5')

    ext1 = elec_extract.get_layer('elec_feature').output
    ext2 = cool_extract.get_layer('cool_feature').output
    ext3 = steam_extract.get_layer('steam_feature').output

    m = Concatenate(axis=1)([ext1, ext2, ext3])

    attention_mul = attention_block(m)
    attention_mul = Flatten()(attention_mul)

    output = Dense(1, activation='sigmoid')(attention_mul)
    model = Model(inputs=[elec_extract.input, cool_extract.input, steam_extract.input], outputs=output)
    model.summary()

    model.compile(loss='mae', optimizer='adam')
    plot_model(model, to_file='multi_attention_model.png')
    model.fit([X_train[:, :, 0].reshape((2920, 24, 1)),
                                        X_train[:, :, 1].reshape((2920, 24, 1)),
                                                                                X_train[:, :, 2].reshape((2920, 24, 1))],
              y_train[:, 0], epochs=epoch, batch_size=batch_size, shuffle=False)

    middle = Model(inputs=[elec_extract.input, cool_extract.input, steam_extract.input],
                   outputs=model.get_layer('attention_vec').output)
    model.save("./multi_attention_model.h5")
    middle.save("./multi_attention.h5")
    return model, middle


# model, middle = multi_attention_Model()
mid = load_model('multi_attention.h5')
weight = mid.predict([X_test[:, :, 0].reshape((712, 24, 1)),
                                        X_test[:, :, 1].reshape((712, 24, 1)),
                                                                                X_test[:, :, 2].reshape((712, 24, 1))])
print("weight.shape: ", weight.shape)  # (712, 72, 48)
print(weight)

# 取第一组测试数据注意力机制的结果
attention_vector = np.mean(weight[0, :, :], axis=0)
print('attention =', attention_vector)

# 画权重柱状图
plt.bar(range(len(attention_vector)), attention_vector, width=0.3, label='weight')
plt.legend()
plt.show()
