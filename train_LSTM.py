import os
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib

# Load preprocessed data
X, y = joblib.load('preprocessed_data.pkl')

# Split into train and test
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

def build_lstm_model(input_shape):
    # Stacked LSTM architecture
    model = Sequential()
    # First LSTM layer (returns sequences to feed to the next LSTM layer)
    model.add(LSTM(64, input_shape=input_shape, return_sequences=True))
    # Second LSTM layer (this one returns only the final output)
    model.add(LSTM(32, return_sequences=False))
    # Dense layer to produce the final output
    model.add(Dense(1))
    # Compile the model with optimizer and loss function
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    
    return model

model = build_lstm_model(input_shape=(X_train.shape[1], X_train.shape[2]))

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# Predict using the trained model
y_pred = model.predict(X_test)

# Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

# Save evaluation metrics to a valid path
output_dir = r'C:\Semester3\FOAI\TeamProject\TrainedModel_LSTM'
os.makedirs(output_dir, exist_ok=True)

with open(os.path.join(output_dir, 'lstm_mae.txt'), 'w') as f:
    f.write(str(mae))

with open(os.path.join(output_dir, 'lstm_mse.txt'), 'w') as f:
    f.write(str(mse))

# Specify the save path for the model
save_path = os.path.join(output_dir, 'lstm_model.h5')

# Save the trained model to the specified path
model.save(save_path)

print(f"Model saved to {save_path}")
print(f"MAE: {mae}, MSE: {mse}")
