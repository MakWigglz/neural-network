import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Fetch the ZIP code digit dataset with error handling
try:
    X, y = fetch_openml('zip.train', version=1, return_X_y=True, as_frame=False, n_retries=5, delay=2)
except Exception as e:
    print(f"An error occurred while fetching the dataset: {e}")
    # You can choose to exit or continue, depending on your needs
    exit()

# Preprocessing
X = np.array(X, dtype=np.float32)
y = np.array(y, dtype=np.int64)

# Standardize features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert to PyTorch tensors
X_train = torch.tensor(X_train)
y_train = torch.tensor(y_train)
X_test = torch.tensor(X_test)
y_test = torch.tensor(y_test)

# Your model definition and training code would go here
