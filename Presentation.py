from keras import models
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_percentage_error

model = models.load_model('Gold.keras')
scaler = joblib.load('Gold.scaler')

df = pd.read_csv('Gold2013-2023.csv', parse_dates=[0])

df.drop(['Vol.', 'Change %'], axis=1, inplace=True)
df.sort_values(by=['Date'], inplace=True)
df.reset_index(drop=True, inplace=True)

numColumns = df.columns.drop(['Date'])
df[numColumns] = df[numColumns].replace(',', '', regex=True)
df[numColumns] = df[numColumns].astype('float64')

window_size = 60
test_size = int(df.shape[0] * 0.1)

train_data = df.Price[:-test_size]
train_data = scaler.transform(train_data.values.reshape(-1, 1))

test_data = df.Price[-test_size - window_size:]
test_data = scaler.transform(test_data.values.reshape(-1, 1))

X_test = []
Y_test = []

for i in range(window_size, len(test_data)):
    X_test.append(test_data[i - window_size:i, 0])
    Y_test.append(test_data[i, 0])

X_test = np.array(X_test)
Y_test = np.array(Y_test)

X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
Y_test = np.reshape(Y_test, (-1, 1))

loss = model.evaluate(X_test, Y_test)[0]
Y_pred = model.predict(X_test)
MAPE = mean_absolute_percentage_error(Y_test, Y_pred)
accuracy = 1 - MAPE

print("Test loss: ", loss)
print("Test MAPE: ", MAPE)
print("Test accuracy: ", accuracy)

Y_test_true = scaler.inverse_transform(Y_test)
Y_test_pred = scaler.inverse_transform(Y_pred)

plt.figure(figsize=(15, 6), dpi=150)
plt.rcParams['axes.facecolor'] = 'lightgray'
plt.plot(df['Date'].iloc[:-test_size], scaler.inverse_transform(train_data), color='blue')
plt.plot(df['Date'].iloc[-test_size:], Y_test_true, color='red')
plt.title('Model on Gold Price Prediction', fontsize=15)
plt.xlabel('Date', fontsize=14)
plt.ylabel('Price', fontsize=14)
plt.legend(['Training set', 'Test set'], loc='upper left', prop={'size': 15})
plt.grid(color='white')
plt.show()

plt.figure(figsize=(15, 6), dpi=150)
plt.rcParams['axes.facecolor'] = 'lightgray'
plt.plot(df['Date'].iloc[:-test_size], scaler.inverse_transform(train_data), color='blue')
plt.plot(df['Date'].iloc[-test_size:], Y_test_true, color='red')
plt.plot(df['Date'].iloc[-test_size:], Y_test_pred, color='green')
plt.title('Model on Gold Price Prediction', fontsize=15)
plt.xlabel('Date', fontsize=14)
plt.ylabel('Price', fontsize=14)
plt.legend(['Training set', 'Test set', 'Prediction'], loc='upper left', prop={'size': 15})
plt.grid(color='white')
plt.show()

plt.figure(figsize=(15, 6), dpi=150)
plt.title('Model on Gold Price Prediction', fontsize=15)
plt.plot(df['Date'].iloc[-test_size:], Y_test_true, color='red')
plt.plot(df['Date'].iloc[-test_size:], Y_test_pred, color='green')
plt.xlabel('Date', fontsize=14)
plt.ylabel('Price', fontsize=14)
plt.legend(['Test set', 'Prediction'], loc='upper left', prop={'size': 15})
plt.grid(color='white')
plt.show()
