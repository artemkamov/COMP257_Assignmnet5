# Import necessary libraries
from keras.models import Model
from keras.layers import Dense, Input
from keras.callbacks import EarlyStopping
from sklearn.model_selection import KFold
from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split
from keras.optimizers import Adam
from keras.regularizers import l2
import numpy as np
import matplotlib.pyplot as plt

# 1. Retrieve and load the Olivetti faces dataset
faces_data = fetch_olivetti_faces(shuffle=True, random_state=42)
X = faces_data.data  # X contains the flattened image data
y = faces_data.target  # y contains the labels

# 2. Split the Dataset using Stratified Sampling
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

# Normalize data directly
X_train = X_train / X_train.max()
X_val = X_val / X_val.max()
X_test = X_test / X_test.max()

# Define parameters for the autoencoder
input_size = X_train.shape[1]
central_layer_size = 32  

# Function to build the autoencoder with the specified architecture
def build_autoencoder(hidden_units1_size, learning_rate, regularizer_strength):
    input_img = Input(shape=(input_size,))
    
    # Encoding
    top_hidden1 = Dense(hidden_units1_size, activation='relu', kernel_regularizer=l2(regularizer_strength))(input_img)
    central_layer = Dense(central_layer_size, activation='relu', kernel_regularizer=l2(regularizer_strength))(top_hidden1)
    
    # Decoding
    top_hidden3 = Dense(hidden_units1_size, activation='relu', kernel_regularizer=l2(regularizer_strength))(central_layer)
    output_img = Dense(input_size, activation='sigmoid')(top_hidden3)
    
    # Model definition
    autoencoder = Model(input_img, output_img)
    optimizer = Adam(learning_rate=learning_rate)
    autoencoder.compile(optimizer=optimizer, loss='binary_crossentropy')
    return autoencoder

# Hyperparameter grid
learning_rates = [0.001, 0.0005]
regularizer_strengths = [0.0001, 0.001]
hidden_units_values = [128, 256]

# Using K-Fold Cross Validation to evaluate model
kf = KFold(n_splits=3)
best_model = None
best_val_loss = float('inf')
best_config = {}

# Perform cross-validation with hyperparameter tuning
val_losses = []

for learning_rate in learning_rates:
    for reg_strength in regularizer_strengths:
        for hidden_units in hidden_units_values:
            autoencoder = build_autoencoder(hidden_units, learning_rate, reg_strength)
            
            fold_val_losses = []
            
            for train_index, val_index in kf.split(X_train):
                X_train_cv, X_val_cv = X_train[train_index], X_train[val_index]
                
                # Train model with early stopping
                history = autoencoder.fit(
                    X_train_cv, X_train_cv,
                    epochs=50,
                    batch_size=32,
                    validation_data=(X_val_cv, X_val_cv),
                    callbacks=[EarlyStopping(monitor='val_loss', patience=10, min_delta=0.0001)],
                    verbose=0
                )
                
                fold_val_losses.append(history.history['val_loss'][-1])
            
            avg_val_loss = np.mean(fold_val_losses)
            val_losses.append((avg_val_loss, learning_rate, reg_strength, hidden_units))
            
            # Print each configuration's result
            print(f"Learning Rate: {learning_rate}, Regularizer Strength: {reg_strength}, "
                  f"Hidden Units: {hidden_units}, Avg Validation Loss: {avg_val_loss:.4f}")
            
            # Update best model if current configuration is better
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_model = autoencoder
                best_config = {
                    "Learning Rate": learning_rate,
                    "Regularizer Strength": reg_strength,
                    "Hidden Units": hidden_units
                }

# Print the best hyperparameters and corresponding validation loss
print("\nBest Configuration:")
print(f"Learning Rate: {best_config['Learning Rate']}, "
      f"Regularizer Strength: {best_config['Regularizer Strength']}, "
      f"Hidden Units: {best_config['Hidden Units']}, "
      f"Best Validation Loss: {best_val_loss:.4f}")

# Run the best model with the test set
decoded_img = best_model.predict(X_test)
decoded_img = np.clip(decoded_img, 0., 1.)

# Display original and reconstructed images
n = 10
plt.figure(figsize=(20, 4))
for i in range(n):
    # Original Image
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(X_test[i].reshape(64, 64), cmap='gray')
    plt.axis('off')

    # Reconstructed Image
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(decoded_img[i].reshape(64, 64), cmap='gray')
    plt.axis('off')
plt.show()
