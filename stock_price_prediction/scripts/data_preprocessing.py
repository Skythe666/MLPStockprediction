import yfinance as yf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

def load_and_preprocess_data(stock_symbol):
    
    data = yf.download(stock_symbol, start='2010-01-01', end='2020-12-31')

    
    data = data[['Open', 'High', 'Low', 'Volume', 'Close']]

    
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)

    X = scaled_data[:, :-1]
    y = scaled_data[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    return X_train, X_test, y_train, y_test
