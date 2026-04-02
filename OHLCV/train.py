import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

df = pd.read_csv('OHLCV/clean_OHLCV.csv')

for lag in range(1, 6):
    df[f'vol_lag{lag}'] = df['volatility'].shift(lag)

df.dropna(inplace=True)

feature_cols = [
    'Open', 'High', 'Low', 'Close', 'Volume', 'ret', 'hl_range',
    'oc_move', 'vol_change',
    'stock2vec_1', 'stock2vec_2', 'stock2vec_3', 'stock2vec_4'
]
lag_cols = [c for c in df.columns if 'vol_lag' in c]
feature_cols += lag_cols

X = df[feature_cols].values
y = df['volatility'].values.reshape(-1, 1)

split = int(len(X) * 0.8)
X_train_raw, X_test_raw = X[:split], X[split:]
y_train_raw, y_test_raw = y[:split], y[split:]

scaler_x = StandardScaler()
scaler_y = StandardScaler()

X_train = scaler_x.fit_transform(X_train_raw)
X_test = scaler_x.transform(X_test_raw)

y_train = scaler_y.fit_transform(y_train_raw)
y_test = scaler_y.transform(y_test_raw)

X_train_lstm = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
X_test_lstm = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

model = Sequential([
    LSTM(64, activation='tanh', input_shape=(X_train_lstm.shape[1], X_train_lstm.shape[2])),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dropout(0.1),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse')

early_stop = EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True)

history = model.fit(
    X_train_lstm, y_train,
    validation_split=0.1,
    epochs=100,
    batch_size=32,
    callbacks=[early_stop],
    verbose=1
)

y_pred_scaled = model.predict(X_test_lstm)
y_pred = scaler_y.inverse_transform(y_pred_scaled).flatten()
y_test_unscaled = scaler_y.inverse_transform(y_test).flatten()

mse = mean_squared_error(y_test_unscaled, y_pred)
r2 = r2_score(y_test_unscaled, y_pred)

print(f"\nMean Squared Error: {mse}")
print(f"R² Score: {r2}")

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Training Performance')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(y_test_unscaled[:100], label='Actual Volatility', alpha=0.7)
plt.plot(y_pred[:100], label='Predicted', linestyle='--')
plt.title('Prediction vs Actual (First 100 Days)')
plt.legend()
plt.show()

residuals = y_test_unscaled - y_pred
plt.scatter(y_pred, residuals)
plt.axhline(0, color='red')
plt.xlabel('Predicted Volatility')
plt.ylabel('Residuals')
plt.show()