import sys

# Retrieve MAE for both models
lstm_mae = float(sys.argv[1])
gru_mae = float(sys.argv[2])

# Output the comparison result
print(f"LSTM MAE: {lstm_mae}")
print(f"GRU MAE: {gru_mae}")

if lstm_mae < gru_mae:
    print("LSTM performed better.")
else:
    print("GRU performed better.")
