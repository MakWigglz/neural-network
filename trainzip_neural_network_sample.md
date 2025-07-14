Absolutely! Here's a lightweight example of a neural network using PyTorch and the classic ZIP code digits dataset from `zip.train`, which you might recognize from older datasets like the USPS digit recognition task. If you're using a notebook, this will run smoothly and give you a compact testbed to play with predictions.

### 🧠 Simple Neural Network with ZIP Train

```python
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load ZIP code digit dataset
# Alternative: Load directly if you have zip.train as a file
X, y = fetch_openml('zip.train', version=1, return_X_y=True, as_frame=False)
X = np.array(X, dtype=np.float32)
y = np.array(y, dtype=np.int64)

# Preprocessing
scaler = StandardScaler()
X = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert to PyTorch tensors
X_train = torch.tensor(X_train)
y_train = torch.tensor(y_train)
X_test = torch.tensor(X_test)
y_test = torch.tensor(y_test)

# Define a simple neural network
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(256, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = SimpleNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
epochs = 10
for epoch in range(epochs):
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()
    print(f'Epoch [{epoch+1}/{epochs}] - Loss: {loss.item():.4f}')

# 🎯 Testing
with torch.no_grad():
    test_outputs = model(X_test)
    _, predicted = torch.max(test_outputs, 1)
    accuracy = (predicted == y_test).float().mean()
    print(f'Test Accuracy: {accuracy:.4f}')
```

---

import matplotlib.pyplot as plt

# Reshape each sample from 256-dimensional vector to 16x16 image
def plot_digits(samples, labels, num_to_show=10):
    plt.figure(figsize=(12, 2))
    for i in range(num_to_show):
        img = samples[i].reshape(16, 16)
        plt.subplot(1, num_to_show, i + 1)
        plt.imshow(img, cmap='gray')
        plt.title(f'Label: {labels[i]}')
        plt.axis('off')
    plt.tight_layout()
    plt.show()

# Show first 10 digits in training set
plot_digits(X_train.numpy(), y_train.numpy())

