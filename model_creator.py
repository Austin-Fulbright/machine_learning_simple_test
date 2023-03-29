import numpy as np
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical

# Load and preprocess the MNIST dataset
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
train_images = train_images.reshape((60000, 28, 28, 1)).astype('float32') / 255
test_images = test_images.reshape((10000, 28, 28, 1)).astype('float32') / 255
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

# Define the CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(train_images, train_labels, epochs=5, batch_size=64)

# Evaluate the model on test data
test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f"Test accuracy: {test_acc:.4f}")

# Save the trained model
model.save('mnist_cnn_model.h5')

# Load a custom handwritten digit image (example: 'handwritten_7.png')
image = cv2.imread('handwritten_7.png', cv2.IMREAD_GRAYSCALE)

# Preprocess the image
resized_image = cv2.resize(image, (28, 28))
inverted_image = 255 - resized_image
normalized_image = inverted_image / 255.0
input_image = normalized_image.reshape(1, 28, 28, 1)

# Create a new model with the same layers as the original model but outputs the activations of each layer
layer_outputs = [layer.output for layer in model.layers]
activation_model = tf.keras.models.Model(inputs=model.input, outputs=layer_outputs)

# Obtain the activations for each layer
activations = activation_model.predict(input_image)

# Print the output for each layer with explanations
for i, activation in enumerate(activations):
    print(f"Output of layer {i + 1} ({model.layers[i].name}):")
    print(activation)

    # Print explanations for each layer
    # ...

# Show the preprocessed input image
plt.imshow(normalized_image, cmap='gray')
plt.show()
