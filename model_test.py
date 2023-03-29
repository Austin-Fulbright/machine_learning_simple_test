import numpy as np
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model('mnist_cnn_model.h5')

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

    if "conv" in model.layers[i].name:
        print("This output represents the feature maps in a convolutional layer.")
        print("Each feature map corresponds to the response of a filter applied to the input or the previous layer's output.")
        print("Higher values in the feature maps indicate the filter has detected a specific feature or pattern in the input.")
    elif "max_pool" in model.layers[i].name:
        print("This output represents the downsampling of the previous layer's feature maps using max-pooling.")
        print("The max-pooling operation selects the maximum value from a region in each feature map.")
        print("This process helps to make the model more robust and reduces computational complexity.")
    elif "dense" in model.layers[i].name and i != len(activations) - 1:
        print("This output represents the activations of a fully connected layer after applying the ReLU activation function.")
        print("These activations represent a high-level understanding of the input image.")
    elif "dense" in model.layers[i].name and i == len(activations) - 1:
        print("This output represents the probability distribution over the 10 possible digits.")
        print("The highest probability corresponds to the predicted digit.")

    print()

# Show the preprocessed input image
plt.imshow(normalized_image, cmap='gray')
plt.show()
