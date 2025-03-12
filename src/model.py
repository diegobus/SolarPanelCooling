import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

# ----------- PREPARE THE DATA FOR THE MODEL ----------- #

# Load the dataset and define features and target
data = pd.read_csv('../solar_data/cooling_data_complete.csv')
X = data[['Tamb', 'I', 'Tcool_in']]  # features
y = data['mass_flowrate']  # target

# Split dataset into training and test subsets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize features and target
scaler_X = StandardScaler()
X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)

scaler_y = StandardScaler()
y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1)).flatten()
y_test_scaled = scaler_y.transform(y_test.values.reshape(-1, 1)).flatten()

# ----------- BUILD THE MODEL AND TRAIN IT ----------- #
# Define the neural network model with the following architecture:
# 1) input layer (use scaled features)
# 2) hidden layer with 3 neurons using:
#     - ReLU activation function
#     - L2 regularization
# 3) dropout layer with 0.1 probability (prevent overfitting)
# 4) output layer with linear activation
inputs = keras.Input(shape=(X_train_scaled.shape[1],))
x = keras.layers.Dense(3, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01))(inputs)
x = keras.layers.Dropout(0.1)(x)
outputs = keras.layers.Dense(1, activation='linear')(x)
model = keras.Model(inputs=inputs, outputs=outputs)

# Train the model with following parameters:
# 1) optimizer  - Adam
# 2) loss       - mean squared error (MSE)
# 3) metrics    - mean absolute error (MAE)
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Train the model with early stopping to prevent overfitting
early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
history = model.fit(
    X_train_scaled, y_train_scaled,
    validation_data=(X_test_scaled, y_test_scaled),
    epochs=200,
    batch_size=16,
    verbose=1,
    callbacks=[early_stopping]
)

# ----------- SAVE MODEL AND LAYER WEIGHTS ----------- #
# Create a dictionary to store layer weights
weights_dict = {}

# Loop through layers and get weights and biases
for i, layer in enumerate(model.layers):
    weights = layer.get_weights()
    
    # if no weights or biases then ignore layer
    if len(weights) == 2:  # Layers with both weights and biases
        w, b = weights
        weights_dict[f"Layer_{i}_Weights"] = w.flatten()
        weights_dict[f"Layer_{i}_Biases"] = b
    elif len(weights) == 1:  # Layers with only weights
        w = weights[0]
        weights_dict[f"Layer_{i}_Weights"] = w.flatten()

# Convert to DataFrame
weights_df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in weights_dict.items()]))

# Save to CSV
weights_df.to_csv("../model/nn_weights.csv", index=False)
print("Neural network weights saved to nn_weights.csv in model/")

# ----------- USE MODEL TO MAKE PREDICTIONS ----------- #
# Use model to make predictions on the test data
y_pred_scaled = model.predict(X_test_scaled).flatten()
y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten() # invert scaling of targets
y_pred = np.maximum(y_pred, 0) # clip negatives since can't have negative flowrate

# Evaluate the model
r2 = r2_score(y_test, y_pred)
print(f"Test R^2 Score: {r2:.4f}")

# Plot actual vs predicted values
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel('Actual mass_flowrate_optimal')
plt.ylabel('Predicted mass_flowrate_optimal')
plt.title('Neural Network: Actual vs Predicted Flowrate')
plt.grid(True)
plt.show()

# Plot loss history
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss (MSE)')
plt.title('Loss Curve')
plt.legend()
plt.grid(True)
plt.show()
