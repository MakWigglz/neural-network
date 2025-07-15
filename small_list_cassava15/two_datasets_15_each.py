import os
import cv2 # OpenCV for image loading
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

# --- 1. Define Paths and Parameters ---
DATA_DIR = 'data' # Make sure this path is correct
IMAGE_SIZE = (128, 128) # Resize all images to this resolution (width, height)
NUM_CLASSES = 2 # Bacterial Blight and Brown Streak Disease
BATCH_SIZE = 8 # Number of images to process at once during training
EPOCHS = 20 # Number of times the entire dataset is passed through the network

# Define the class names and their corresponding integer labels
CLASS_NAMES = {
    'bacterial_blight': 0,
    'brown_streak_disease': 1
}
# This will be useful for one-hot encoding:
# Bacterial Blight: [1, 0]
# Brown Streak Disease: [0, 1]

# --- 2. Load and Preprocess Images ---

images = []
labels = []

print("Loading images...")
for class_name, label_int in CLASS_NAMES.items():
    class_dir = os.path.join(DATA_DIR, class_name)
    if not os.path.exists(class_dir):
        print(f"Warning: Directory '{class_dir}' not found. Please check your data organization.")
        continue

    for img_name in os.listdir(class_dir):
        img_path = os.path.join(class_dir, img_name)
        try:
            # Read image in color (BGR by default with OpenCV)
            img = cv2.imread(img_path)
            if img is None:
                print(f"Warning: Could not read image {img_path}. Skipping.")
                continue

            # Resize image
            img = cv2.resize(img, IMAGE_SIZE)

            # Normalize pixel values to be between 0 and 1
            img = img / 255.0

            images.append(img)
            labels.append(label_int)
        except Exception as e:
            print(f"Error processing {img_path}: {e}")

# Convert lists to NumPy arrays
images = np.array(images)
labels = np.array(labels)

print(f"Loaded {len(images)} images.")
print(f"Image shape: {images.shape}") # Should be (num_images, IMAGE_SIZE[0], IMAGE_SIZE[1], 3)
print(f"Labels shape: {labels.shape}")

# --- 3. Split Data into Training and (Small) Test Sets ---
# With 30 images, this split will be very small.
# A typical split is 80% train, 20% test.
X_train, X_test, y_train, y_test = train_test_split(
    images, labels, test_size=0.2, random_state=42, stratify=labels
)

# Convert integer labels to one-hot encoded vectors
y_train_one_hot = to_categorical(y_train, num_classes=NUM_CLASSES)
y_test_one_hot = to_categorical(y_test, num_classes=NUM_CLASSES)

print(f"\nTraining data shape: {X_train.shape}, Labels shape: {y_train_one_hot.shape}")
print(f"Test data shape: {X_test.shape}, Labels shape: {y_test_one_hot.shape}")

# --- 4. Build the Neural Network (Simple CNN) ---
# We'll use a Convolutional Neural Network (CNN) as they are excellent for images.
# A very shallow CNN for this small dataset.

model = Sequential([
    # Convolutional layer 1: learns features from the image
    # 32 filters, 3x3 kernel, ReLU activation
    Conv2D(32, (3, 3), activation='relu', input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3)),
    # Max Pooling layer 1: reduces spatial dimensions, makes the network more robust
    MaxPooling2D((2, 2)),

    # Convolutional layer 2
    Conv2D(64, (3, 3), activation='relu'),
    # Max Pooling layer 2
    MaxPooling2D((2, 2)),

    # Flatten the 3D output to 1D for the dense layers
    Flatten(),

    # Dense hidden layer: standard fully connected layer
    Dense(64, activation='relu'),

    # Output layer: 2 neurons for 2 classes, softmax for probability distribution
    Dense(NUM_CLASSES, activation='softmax')
])

# --- 5. Compile the Model ---
# Optimizer: Adam is a good general-purpose optimizer
# Loss function: categorical_crossentropy for one-hot encoded labels
# Metrics: accuracy to monitor performance
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary() # Print a summary of the model's architecture

# --- 6. Train the Model ---
print("\nTraining the model...")
history = model.fit(
    X_train, y_train_one_hot,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_data=(X_test, y_test_one_hot) # Use the small test set as validation
)

# --- 7. Evaluate the Model ---
print("\nEvaluating the model on the test set:")
loss, accuracy = model.evaluate(X_test, y_test_one_hot)
print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy:.4f}")

# --- 8. Visualize Training History ---
plt.figure(figsize=(12, 5))

# Plot training & validation accuracy values
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

# Plot training & validation loss values
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

# --- 9. Make a Prediction on a Sample Image ---
if len(X_test) > 0:
    sample_image_index = 0 # Take the first image from the test set
    sample_image = X_test[sample_image_index]
    true_label_one_hot = y_test_one_hot[sample_image_index]
    true_label_int = np.argmax(true_label_one_hot)
    true_class_name = list(CLASS_NAMES.keys())[list(CLASS_NAMES.values()).index(true_label_int)]

    # Add a batch dimension (model expects batches, even for single image)
    sample_image_batch = np.expand_dims(sample_image, axis=0)

    prediction_one_hot = model.predict(sample_image_batch)[0]
    predicted_label_int = np.argmax(prediction_one_hot)
    predicted_class_name = list(CLASS_NAMES.keys())[list(CLASS_NAMES.values()).index(predicted_label_int)]

    plt.imshow(sample_image) # Display the normalized image
    plt.title(f"True: {true_class_name}\nPredicted: {predicted_class_name} (Prob: {prediction_one_hot[predicted_label_int]:.2f})")
    plt.axis('off')
    plt.show()

    print(f"\nPrediction for sample image (Index {sample_image_index}):")
    print(f"True Label: {true_class_name} (One-hot: {true_label_one_hot})")
    print(f"Predicted Probabilities: {prediction_one_hot}")
    print(f"Predicted Label: {predicted_class_name} (Integer: {predicted_label_int})")
else:
    print("\nNo test images available for prediction.")
