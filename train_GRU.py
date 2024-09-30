import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib

# Load preprocessed data
X, y = joblib.load('preprocessed_data.pkl')

# Split into train and test
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Define the GRU model function
def build_gru_model(input_shape):
    # Basic GRU architecture
    model = Sequential()
    model.add(GRU(64, input_shape=input_shape, return_sequences=True))
    model.add(GRU(32))
    model.add(Dense(1))  # The final output layer for regression
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

# Create the model
model = build_gru_model(input_shape=(X_train.shape[1], X_train.shape[2]))

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# Predict using the trained model
y_pred = model.predict(X_test)

# Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

# Save evaluation metrics to a valid path
output_dir = r'C:\Semester3\FOAI\TeamProject\TrainedModel_GRU'
os.makedirs(output_dir, exist_ok=True)

with open(os.path.join(output_dir, 'gru_mae.txt'), 'w') as f:
    f.write(str(mae))

with open(os.path.join(output_dir, 'gru_mse.txt'), 'w') as f:
    f.write(str(mse))

# Specify the save path for the model
save_path = os.path.join(output_dir, 'gru_model.h5')

# Save the trained model to the specified path
model.save(save_path)

print(f"GRU model saved to {save_path}")
print(f"MAE: {mae}, MSE: {mse}")
