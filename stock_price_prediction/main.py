from scripts.data_preprocessing import load_and_preprocess_data
from model.mlp_model import create_mlp_model
from scripts.model_training import train_and_evaluate_model

def main():
    X_train, X_test, y_train, y_test = load_and_preprocess_data('AAPL')

    model = create_mlp_model(input_shape=(X_train.shape[1],))

    train_and_evaluate_model(model, X_train, X_test, y_train, y_test)

if __name__ == "__main__":
    main()
