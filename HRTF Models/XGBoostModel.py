import xgboost as xgb
import numpy as np

class HRIRPredictor:
    def __init__(self, n_estimators=100, max_depth=3, learning_rate=0.1):
        """
        Initialize the HRIR predictor with XGBoost parameters.
        
        Parameters:
        - n_estimators: Number of gradient boosted trees. Equivalent to number of boosting rounds.
        - max_depth: Maximum tree depth for base learners.
        - learning_rate: Boosting learning rate (xgb's "eta")
        """
        self.model = xgb.XGBRegressor(n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate)
    
    def prepare_data(self, hrir_input, angles):
        """
        Prepares the data for training or prediction. This function assumes the HRIR data
        is passed as a numpy array where each row is an HRIR sample, and angles are provided
        as a numpy array of corresponding angles.
        
        Parameters:
        - hrir_input: A 2D numpy array of shape (n_samples, hrir_length).
        - angles: A 1D numpy array of angles corresponding to each HRIR sample.
        
        Returns:
        - A 2D numpy array where each sample is concatenated with its corresponding angle.
        """
        # Assuming angles is a 1D numpy array and needs to be reshaped to concatenate with hrir_input
        angles_expanded = angles[:, np.newaxis]  # Reshape angles to be a 2D array of shape (n_samples, 1)
        data = np.concatenate([hrir_input, angles_expanded], axis=1)  # Concatenate along the columns
        return data
    
    def fit(self, X_train, y_train, angles_train):
        """
        Train the XGBoost model on the provided HRIR data and angles.
        
        Parameters:
        - X_train: A 2D numpy array of input HRIR samples.
        - y_train: A 2D numpy array of target HRIR samples to predict.
        - angles_train: A 1D numpy array of angles corresponding to each HRIR sample in X_train.
        """
        # Prepare the training data
        train_data = self.prepare_data(X_train, angles_train)
        self.model.fit(train_data, y_train)
    
    def predict(self, X_test, angles_test):
        """
        Predict the HRIR given new HRIR data and angles using the trained model.
        
        Parameters:
        - X_test: A 2D numpy array of input HRIR samples for prediction.
        - angles_test: A 1D numpy array of angles corresponding to each HRIR sample in X_test.
        
        Returns:
        - A 2D numpy array of predicted HRIR samples.
        """
        # Prepare the test data
        test_data = self.prepare_data(X_test, angles_test)
        return self.model.predict(test_data)



import sys
sys.path.append('/workspace/fourth_year_project/HRTF Models/')

from BasicDataset import BasicDataset
# from BasicTransformer import BasicTransformer

sofa_file = '/workspace/fourth_year_project/HRTF Models/sofa_hrtfs/RIEC_hrir_subject_001.sofa'
# Basic Dataset only loads the HRIRs at 0 degrees and 90 degrees for baseline and 45 degree for testing
hrir_dataset = BasicDataset()
for i in range(1,100):
    hrir_dataset.load(sofa_file.replace('001', str(i).zfill(3)))

hrirs, angles = hrir_dataset.get_all()

from sklearn.model_selection import train_test_split

# Assuming `hrirs` is your dataset of HRIR signals and `angles` are the corresponding angles
X_train, X_test, y_train, y_test = train_test_split(hrirs, angles, test_size=0.2, random_state=42)


X_test_prepared = np.array(X_test)  # Preparing the test HRIR data

# predictions = model.predict(X_test_prepared)

# # Evaluate predictions, for example, using MSE or any other suitable metric
# from sklearn.metrics import mean_squared_error

# mse = mean_squared_error(y_test, predictions)
# print(f'Test MSE: {mse}')
