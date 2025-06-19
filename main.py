import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_percentage_error
import keras
from keras import layers
from keras import callbacks

df = pd.read_csv('Gold2013-2023.csv', parse_dates=[0])

df.drop(['Vol.', 'Change %'], axis=1, inplace=True)
df.sort_values(by=['Date'], inplace=True)
df.reset_index(drop=True, inplace=True)

numColumns = df.columns.drop(['Date'])
df[numColumns] = df[numColumns].replace(',', '', regex=True)
df[numColumns] = df[numColumns].astype('float64')

print('Количество дублирующихся строк: ', df.duplicated().sum())
print('Количество NULL значений в датафрейме: ', df.isnull().sum().sum())

print(df.head())
df.info()

fig = px.line(y=df.Price, x=df.Date)
fig.update_layout(xaxis_title='Date',
                  yaxis_title='Price',
                  title={'text': 'Gold Price History', 'y': 0.97, 'x': 0.5})

# fig.show()

test_size = int(df.shape[0] * 0.1)

plt.figure(figsize=(15, 6), dpi=150)
plt.rcParams['axes.facecolor'] = 'lightgray'
plt.plot(df.Date[:-test_size], df.Price[:-test_size], color='blue', lw=2)
plt.plot(df.Date[-test_size:], df.Price[-test_size:], color='red', lw=2)
plt.title('Gold Price Training and Test', fontsize=15)
plt.xlabel('Date', fontsize=14)
plt.ylabel('Price', fontsize=14)
plt.legend(['Training set', 'Test set'], loc='upper left', prop={'size': 15})
plt.grid(color='white')
plt.show()

scaler = MinMaxScaler()
scaler.fit(df.Price.values.reshape(-1, 1))

window_size = 60

train_data = df.Price[:-test_size]
train_data = scaler.transform(train_data.values.reshape(-1, 1))

X_train = []
Y_train = []

for i in range(window_size, len(train_data)):
    X_train.append(train_data[i - window_size:i, 0])
    Y_train.append(train_data[i, 0])

test_data = df.Price[-test_size - window_size:]
test_data = scaler.transform(test_data.values.reshape(-1, 1))

X_test = []
Y_test = []

for i in range(window_size, len(test_data)):
    X_test.append(test_data[i - window_size:i, 0])
    Y_test.append(test_data[i, 0])

X_train = np.array(X_train)
X_test = np.array(X_test)
Y_train = np.array(Y_train)
Y_test = np.array(Y_test)

X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
Y_train = np.reshape(Y_train, (-1, 1))
Y_test = np.reshape(Y_test, (-1, 1))

print('X_train Shape: ', X_train.shape)
print('Y_train Shape: ', Y_train.shape)
print('X_test Shape: ', X_test.shape)
print('Y_test Shape: ', Y_test.shape)


def define_model():
    input1 = layers.Input(shape=(window_size, 1))
    x = layers.LSTM(units=64, return_sequences=True)(input1)
    x = layers.Dropout(rate=0.2)(x)
    x = layers.LSTM(units=64, return_sequences=True)(x)
    x = layers.Dropout(rate=0.2)(x)
    x = layers.LSTM(units=64)(x)
    x = layers.Dropout(rate=0.2)(x)
    x = layers.Dense(32, activation='relu')(x)
    dnn_output = layers.Dense(1)(x)

    model = keras.Model(inputs=input1, outputs=[dnn_output])
    model.compile(loss='mean_squared_error', optimizer='Nadam', metrics=['mae'])
    model.summary()

    return model


model = define_model()

callback = callbacks.EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True,
    min_delta=1e-4
)

history = model.fit(X_train, Y_train, epochs=20, batch_size=32, validation_split=0.1)

# -----------------------------

# График потерь
fig, axs = plt.subplots(2, 1, figsize=(10, 10))
f1 = axs[0]
f2 = axs[1]

#fig.tight_layout(h_pad=4)
f1.set_yscale('log')
f2.set_yscale('log')

f1.plot(history.history['loss'], label='train')
f1.plot(history.history['val_loss'], label='val')
f1.set_title('Loss over epochs')
f1.set_xlabel('Epoch')
f1.set_ylabel('MSE Loss')
f1.legend()

f2.plot(history.history['mae'], label='train')
f2.plot(history.history['val_mae'], label='val')
f2.set_title('MAE over epochs')
f2.set_xlabel('Epoch')
f2.set_ylabel('MAE')
f2.legend()

#fig.savefig('plot.png')
#os.startfile('plot.png')
fig.show()
# -----------------------------

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

# model.save('Gold.keras')
# joblib.dump(scaler, 'Gold.scaler')
