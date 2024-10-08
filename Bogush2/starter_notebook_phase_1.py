import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import mlflow
import mlflow.keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense
import logging
import h5py
import tensorflow as tf  # Added import

# Step 1: Set up logging to a file (train.log)
logging.basicConfig(filename='train.log', level=logging.INFO, format='%(asctime)s - %(message)s')

# Step 2: Load and preprocess the dataset
def load_data():
    with h5py.File(r'C:\foai\94879-starter-code-Team-Project\metr-la.h5', 'r') as f:
        df_group = f['df']
        
        # Load the column names (features) from 'block0_items'
        columns = list(df_group['block0_items'][:].astype(str))
        
        # Load the actual data from 'block0_values'
        data = df_group['block0_values'][:]
        
        # Convert the data into a DataFrame
        df = pd.DataFrame(data, columns=columns)
        
        return df

def preprocess_data(data):
    # Use MinMaxScaler to normalize the data
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)
    return data_scaled, scaler

# Step 3: Create sequences for time-series forecasting
def create_sequences(data, time_steps=10):
    X, y = [], []
    for i in range(len(data) - time_steps):
        X.append(data[i:i + time_steps])  # Sequence of features
        y.append(data[i + time_steps, 0])  # Predict the next value for the first feature
    return np.array(X), np.array(y)

# Step 4: Define the LSTM and GRU models
def build_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(64, input_shape=input_shape, return_sequences=True))
    model.add(LSTM(32))
    model.add(Dense(1))
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.MeanSquaredError(),
                  metrics=[tf.keras.metrics.MeanAbsoluteError()])
    return model

def build_gru_model(input_shape):
    model = Sequential()
    model.add(GRU(64, input_shape=input_shape, return_sequences=True))
    model.add(GRU(32))
    model.add(Dense(1))
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.MeanSquaredError(),
                  metrics=[tf.keras.metrics.MeanAbsoluteError()])
    return model

# Step 5: Train and evaluate the model, and log to MLflow
def train_and_evaluate(model_type="LSTM"):
    # Load and preprocess the data
    data = load_data()
    data_scaled, scaler = preprocess_data(data)

    # Create sequences
    time_steps = 10
    X, y = create_sequences(data_scaled, time_steps)

    # Check the number of features in the dataset
    num_features = X.shape[2] if len(X.shape) > 2 else 1

    # Reshape the data to (samples, time_steps, num_features)
    X = X.reshape((X.shape[0], X.shape[1], num_features))  # Use actual number of features
    
    input_shape = (X.shape[1], X.shape[2])

    # Start an MLflow run
    with mlflow.start_run():
        # Select and build the model
        if model_type == "LSTM":
            model = build_lstm_model(input_shape)
            logging.info(f"Building LSTM model with input shape {input_shape}")
        elif model_type == "GRU":
            model = build_gru_model(input_shape)
            logging.info(f"Building GRU model with input shape {input_shape}")

        # Log the model type as a parameter
        mlflow.log_param("model_type", model_type)
        logging.info(f"Starting training for {model_type} model...")

        # Train the model
        history = model.fit(X, y, epochs=5, batch_size=32, validation_split=0.2)

        # Make predictions and calculate metrics
        y_pred = model.predict(X)
        mae = mean_absolute_error(y, y_pred)
        mse = mean_squared_error(y, y_pred)
        r2 = r2_score(y, y_pred)

        # Log metrics
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("mse", mse)
        mlflow.log_metric("r2_score", r2)

        logging.info(f"Model training complete. MAE: {mae}, MSE: {mse}, R2: {r2}")

        # Log the trained model to MLflow
        mlflow.keras.log_model(model, "traffic_prediction_model")

        # Save the model locally as 'lstm_model.h5'
        model.save('lstm_model.h5')

        # Upload the train.log file as an artifact
        mlflow.log_artifact("train.log")

        logging.info(f"{model_type} Model - MAE: {mae}, MSE: {mse}, R2 Score: {r2}")

if __name__ == "__main__":
    # Example of running the training for LSTM
    train_and_evaluate(model_type="LSTM")
