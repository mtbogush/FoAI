from flask import Flask, request, jsonify
import numpy as np
from tensorflow.keras.models import load_model
import tensorflow as tf  # Ensure consistent TensorFlow imports

# Load the trained model
def load_trained_model(model_path='lstm_model.h5'):
    return load_model(model_path)

# Initialize Flask app
app = Flask(__name__)
model = load_trained_model()

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Parse input data
        data = request.get_json(force=True)
        input_data = data['input']  # Expecting a 2D list (time_steps x num_features)

        # Convert input data to numpy array
        prediction_input = np.array(input_data)

        # Reshape input to match the model's expected input shape
        # The model expects input shape: (samples, time_steps, num_features)
        prediction_input = prediction_input.reshape((1, prediction_input.shape[0], prediction_input.shape[1]))

        # Make prediction
        prediction = model.predict(prediction_input)

        # Return the prediction as a list
        return jsonify({'prediction': prediction.tolist()})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    # Run the Flask app
    app.run(host='0.0.0.0', port=5000)

