from keras import models
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

model = models.load_model('Gold_sample.keras')
scaler = joblib.load('Gold.scaler')

df = pd.read_csv('Gold.csv', parse_dates=[0])

df.drop(['Vol.', 'Change %'], axis=1, inplace=True)
df.sort_values(by=['Date'], inplace=True)
df.reset_index(drop=True, inplace=True)

numColumns = df.columns.drop(['Date'])
df[numColumns] = df[numColumns].replace(',', '', regex=True)
df[numColumns] = df[numColumns].astype('float64')


def predict_for_date(target_date, df, model, scaler, window_size=100):
    # Преобразуем целевую дату в datetime
    target_date = pd.to_datetime(target_date)
    print()

    # Проверяем, есть ли в датафрейме данные до целевой даты
    if target_date <= df['Date'].max():
        print("Ошибка: Дата должна быть позже последней даты в датафрейме.")
        return None

    # Берем последние window_size значений из датафрейма
    last_values = df['Price'].values[-window_size:]

    # Масштабируем данные
    last_values_scaled: np.ndarray = scaler.transform(last_values.reshape(-1, 1))

    values = []
    for i in range((target_date - df['Date'].max()).days):
        X_pred = np.array([last_values_scaled[:, 0]])
        X_pred = np.reshape(X_pred, (X_pred.shape[0], 1, X_pred.shape[1]))

        prediction_scaled = model.predict(X_pred)

        last_values_scaled = np.roll(last_values_scaled, -1)
        last_values_scaled[-1] = prediction_scaled
        print(last_values_scaled[-5:].tolist())

        values.append(scaler.inverse_transform(prediction_scaled))

    plt.plot([i[0] for i in values])
    plt.show()
    # Обратное масштабирование
    prediction = scaler.inverse_transform([last_values_scaled[-1]])

    return prediction[0][0]


# Пример использования:
target_date = "2025-06-05"  # Замените на нужную дату
predicted_price = predict_for_date(target_date, df, model, scaler)
print(f"Предсказанная цена на {target_date}: {predicted_price:.2f}")
