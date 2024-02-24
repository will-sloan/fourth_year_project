import numpy as np
import xgboost as xgb
import torch
from HRIRDataset import HRIRDataset
# from torch.utils.data import DataLoader

sofa_file = '/workspace/fourth_year_project/HRTF Models/sofa_hrtfs/RIEC_hrir_subject_001.sofa'
hrir_dataset = HRIRDataset()
for i in range(1,100):
    hrir_dataset.load(sofa_file.replace('001', str(i).zfill(3)))

train_size = int(0.7 * len(hrir_dataset))
val_size = int(0.2 * len(hrir_dataset))
test_size = len(hrir_dataset) - train_size - val_size
train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(hrir_dataset, [train_size, val_size, test_size])

# Placeholder for data loading mechanism
# Assuming `data_loader` is an iterable that yields (input_seq, target_seq, angle)

# Initialize an empty list to store models
models = [xgb.XGBRegressor() for _ in range(1024)]

# Assuming the data loader provides one example at a time for simplicity
for input_seq, target_seq, angle in train_dataset:
    # Flatten the input and target sequences
    input_seq_flat = input_seq.flatten()  # Flatten to [1, 1024]
    target_seq_flat = target_seq.flatten()  # Flatten to [1, 1024]
    
    # Concatenate the angle to the input sequence for each data point
    # Note: This example treats angle as a single feature, so it's repeated for each data point
    # Adjust accordingly if angle is already in a compatible shape or if you're processing batches
    X_with_angle = np.concatenate((input_seq_flat, [angle]))
    print(X_with_angle.shape)
    
    # Train each model on its respective target value
    for i in range(1024):
        # Extract the current target value
        y_target = target_seq_flat[i]
        
        # Fit the model (you may want to add your training data to a larger batch and train after accumulating enough data)
        # This is a simplified example; in practice, consider efficiency and regularization needs
        models[i].fit(X_with_angle.reshape(1, -1), np.array([y_target]))


def predict_sequence(input_sequence, angle, models):
    # Flatten the input_sequence and append the angle
    input_features = np.concatenate((input_sequence.flatten(), [angle]))
    
    # Initialize an array to store predictions
    predictions = np.zeros(1024)
    
    # Generate predictions for each value in the sequence
    for i, model in enumerate(models):
        # Predict the value for the current position
        pred = model.predict(input_features.reshape(1, -1))
        predictions[i] = pred
    
    return predictions.reshape(2, 512)  # Reshape back to [2, 512] format

# Example usage with a single data point from the loader
# Assuming you have a function or mechanism to load a single example for prediction
input_seq, _, angle = next(iter(data_loader))  # Placeholder for getting a prediction example
predicted_sequence = predict_sequence(input_seq, angle, models)
