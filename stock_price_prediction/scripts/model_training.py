from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_squared_error

def train_and_evaluate_model(model, X_train, X_test, y_train, y_test):
    model.compile(optimizer=Adam(), loss='mse')

    model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.2)

    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    print(f"Mean Squared Error: {mse:.4f}")

    next_day_prediction = model.predict(X_test[-1].reshape(1, -1))
    print(f"Predicted stock price for the next day: {next_day_prediction[0][0]:.4f}")
