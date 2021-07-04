import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt


def get_data():
    df = pd.read_csv('./data/campus.csv')
    df = df.iloc[:, 4:]
    print(df.head())

    data = df.values
    print(data.shape)

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(data)
    print(scaled)
    return data, scaled


def main():
    data, scaled = get_data()

    plt.subplot(3, 1, 1)
    plt.plot(range(1500), scaled[:1500, 0], label='elec', color='g')
    plt.legend(['elec'], loc='upper right')

    plt.subplot(3, 1, 2)
    plt.plot(range(1500), scaled[:1500, 1], label='cool', color='b')
    plt.legend(['cool'], loc='upper right')

    plt.subplot(3, 1, 3)
    plt.plot(range(1500), scaled[:1500, 2], label='heat', color='r')
    plt.legend(['heat'], loc='upper right')
    plt.show()


if __name__ == '__main__':
    main()
