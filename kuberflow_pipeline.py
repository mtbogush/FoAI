import kfp
from kfp import dsl

@dsl.component(base_image='tensorflow/tensorflow:latest-gpu')
def train_lstm_model(output_dir: str):
    import subprocess
    subprocess.run(['python3', 'train_LSTM.py', '--output_dir', output_dir])

@dsl.component(base_image='tensorflow/tensorflow:latest-gpu')
def train_gru_model(output_dir: str):
    import subprocess
    subprocess.run(['python3', 'train_GRU.py', '--output_dir', output_dir])

@dsl.pipeline(
    name='Traffic Prediction Pipeline',
    description='Train and evaluate LSTM and GRU models'
)
def traffic_prediction_pipeline(model_type: str, output_dir: str):
    if model_type == "LSTM":
        train_lstm_model(output_dir=output_dir)
    elif model_type == "GRU":
        train_gru_model(output_dir=output_dir)

# Compile and run the Kubeflow pipeline
if __name__ == "__main__":
    kfp.compiler.Compiler().compile(traffic_prediction_pipeline, 'traffic_prediction_pipeline.yaml')

    client = kfp.Client()
    experiment = client.create_experiment('Traffic_Prediction_Experiment')

    # Start a pipeline run for LSTM or GRU model training
    run = client.run_pipeline(
        experiment.id, 
        'traffic_prediction_run', 
        'traffic_prediction_pipeline.yaml', 
        {"model_type": "LSTM", "output_dir": "/output"}  # Change to "GRU" for GRU model training
    )
