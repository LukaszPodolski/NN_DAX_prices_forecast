# NN_DAX_prices_forecast

Stock Price Prediction using MLP Regressor
    This Python script utilizes the yfinance library to download historical stock market data for the DAX index (^GDAXI) from Yahoo Finance. The script then generates various features for the stock data, trains a Multi-Layer Perceptron (MLP) Regressor model using the scikit-learn library, and predicts future stock prices.

Features Generated
    prev_close: Previous day's closing price.
    return: Daily return calculated as the percentage change in closing price.
    ma_5: 5-day moving average.
    ma_20: 20-day moving average.
    rsi: Relative Strength Index.
Model Training
    The script prepares input and output data, standardizes the input data, imputes missing values, and splits the data into training and testing sets. It then trains an MLP Regressor model with a specific architecture (two hidden layers with 100 neurons each).

Model Evaluation
    The Mean Squared Error (MSE) is calculated to evaluate the model's performance on the testing set.

Future Price Prediction
    The script downloads data for the year 2023, generates features, scales and imputes the data, and uses the trained model to predict stock prices for 2023. The predicted prices are visualized alongside actual prices using matplotlib.
