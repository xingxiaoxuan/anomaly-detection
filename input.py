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
    plt.plot(range(len(scaled)), scaled[:, 0], label='elec', color='g')
    plt.plot(range(len(scaled)), scaled[:, 1], label='cool', color='b')
    plt.plot(range(len(scaled)), scaled[:, 2], label='heat', color='r')
    plt.legend(['elec', 'cool', 'heat'], loc='upper right')
    plt.show()


if __name__ == '__main__':
    main()
