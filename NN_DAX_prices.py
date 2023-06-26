import yfinance as yf
from sklearn import preprocessing
import datetime
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
from sklearn.impute import SimpleImputer

def generate_features(df):
    df['prev_close'] = df['Close'].shift(1)
    df['return'] = (df['Close'] - df['prev_close']) / df['prev_close']
    df['ma_5'] = df['Close'].rolling(window=5).mean()
    df['ma_20'] = df['Close'].rolling(window=20).mean()
    df['rsi'] = calculate_rsi(df['Close'])

def calculate_rsi(prices, window=14):
    delta = prices.diff()
    delta = delta[1:]

    up = delta.copy()
    up[up < 0] = 0

    down = delta.copy()
    down[down > 0] = 0
    down = -down

    avg_gain = up.rolling(window).mean()
    avg_loss = down.rolling(window).mean()

    rs = avg_gain / avg_loss
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi

# Downloading stock market data for the period
start_date = datetime.datetime(1988, 1, 1)
end_date = datetime.datetime(2022, 12, 31)
df = yf.download("^GDAXI", start=start_date, end=end_date)
df.reset_index(inplace=True)

# Generating features for stock data
generate_features(df)

# Preparation of input data
X = df[['Open', 'High', 'Low', 'Close', 'Volume', 'prev_close', 'return', 'ma_5', 'ma_20', 'rsi']].values

# Standardization of input data
scaler = preprocessing.StandardScaler()
X_scaled = scaler.fit_transform(X)

# Preparation of output data.
y = df['Close'].values

# Podział danych na zbiór treningowy i testowy
train_size = int(0.8 * len(X_scaled))
X_train, X_test = X_scaled[:train_size], X_scaled[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Uzupełnienie brakujących wartości
imputer = SimpleImputer(strategy='mean')
X_train_imputed = imputer.fit_transform(X_train)
X_test_imputed = imputer.transform(X_test)

# Trenowanie modelu MLP Regressor
model = MLPRegressor(hidden_layer_sizes=(100, 100), max_iter=1000, random_state=42) #hidden_layer_sizes=(100, 100): Określa strukturę sieci neuronowej, w tym liczbę ukrytych warstw i liczność neuronów w każdej warstwie ukrytej. W tym przypadku, model ma dwie warstwy ukryte, z 100 neuronami w każdej warstwie.max_iter=1000: Określa maksymalną liczbę iteracji (epok) podczas trenowania modelu. Model będzie trenowany przez co najwyżej 1000 iteracji. random_state=42: Ustawia ziarno losowości, co zapewnia powtarzalność wyników przy kolejnych uruchomieniach kodu.
model.fit(X_train_imputed, y_train) # Metoda fit() trenuje model na podanych danych treningowych, dostosowując wagi sieci neuronowej, aby minimalizować błąd regresji między przewidywanymi wartościami a rzeczywistymi wartościami docelowymi.

# Predykcja dla danych testowych
y_pred = model.predict(X_test_imputed)

# Wyliczenie błędu średniokwadratowego (MSE)
mse = mean_squared_error(y_test, y_pred)
print('Mean Squared Error:', mse)

# Predykcja dla danych z roku 2023
df_2023 = yf.download("^GDAXI", start=datetime.datetime(2023, 1, 1), end=datetime.datetime(2023, 5, 6))
df_2023.reset_index(inplace=True)
generate_features(df_2023)

X_2020 = df_2023[['Open', 'High', 'Low', 'Close', 'Volume', 'prev_close', 'return', 'ma_5', 'ma_20', 'rsi']].values
X_2020_scaled = scaler.transform(X_2020)
X_2020_imputed = imputer.transform(X_2020_scaled)
predictions_2020 = model.predict(X_2020_imputed) # Dokonuje predykcji na podstawie wcześniej zdefiniowanego modelu na przetworzonych danych X_2020_imputed i zapisuje wyniki w obiekcie predictions_2020.

df_2023['Predicted_Close'] = predictions_2020

dates = df_2023['Date']
close_prices = df_2023['Close']
predicted_prices = df_2023['Predicted_Close']

plt.figure(figsize=(12, 6))
plt.plot(dates, close_prices, label='Actual Close', color='b')
plt.plot(dates, predicted_prices, label='Predicted Close', color='r')
plt.title('Actual vs. Predicted Close Prices for 2023')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.legend()
plt.xticks(rotation=45)
plt.grid(True)
plt.show()