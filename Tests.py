from keras import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

data = pd.read_csv('Gold_svej.csv', parse_dates=['Date'], index_col='Date')
data['Price'] = data['Price'].str.replace(',', '').astype(float)
data.sort_values(by=['Date'], inplace=True)

series = data['Price'].dropna()

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(series.values.reshape(-1, 1))

def create_dataset(dataset, look_back=1):
    X, Y = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back), 0]
        X.append(a)
        Y.append(dataset[i + look_back, 0])
    return np.array(X), np.array(Y)

look_back = 60
X, y = create_dataset(scaled_data, look_back)

train_size = int(len(series) * 0.95)

X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))

model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(1, look_back)))
model.add(LSTM(50))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')

model.fit(X_train, y_train, epochs=20, batch_size=1, verbose=2)

train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

train_predict = scaler.inverse_transform(train_predict)
y_train = scaler.inverse_transform([y_train])
test_predict = scaler.inverse_transform(test_predict)
y_test = scaler.inverse_transform([y_test])

mse = mean_squared_error(y_test[0], test_predict[:,0])
print(f'Mean Squared Error: {mse}')

plt.figure(figsize=(12, 6))
plt.plot(series.index[look_back:train_size+look_back], y_train[0], label='Actual Train')
plt.plot(series.index[train_size+look_back:len(series)-1], y_test[0], label='Actual Test')
plt.plot(series.index[train_size+look_back:len(series)-1], test_predict, label='Predicted')
plt.legend()
plt.show()

model.save("Gold_sample.keras")
