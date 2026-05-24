# ============================================
# BASIC NEURAL NETWORK USING PYTORCH
# ============================================

# We first import PyTorch library.
# PyTorch is the main deep learning framework
# used for building neural networks.
#-------------------------------------------------------------

import torch
import torch.nn as nn
# torch.nn contains neural network components.
# Example:
# - Linear layers
# - Activation functions
# - Loss functions
#
# Think of this as a toolbox for neural networks.
#-------------------------------------------------------------

import torch.optim as optim
# torch.optim contains optimizers.
#
# Optimizers help the neural network improve
# its weights during training.
#
# Example optimizers:
# - SGD
# - Adam
# - RMSProp
#-------------------------------------------------------------

from torchvision import datasets, transforms
# torchvision contains popular computer vision datasets. As we want to read images, we will use torchvision.
#
# We are using:
# - MNIST dataset
# - image transformations
#-------------------------------------------------------------

from torch.utils.data import DataLoader
# DataLoader helps us load data in batches.
#
# Instead of loading all images at once,
# we process smaller groups called batches.

#download MNIST dataset for training and testing
#-------------------------------------------------------------


# DATA PREPROCESSING
# ============================================

# Neural networks cannot understand images directly.
#
# Images must be converted into tensors.
#
# Tensor = mathematical structure used by PyTorch

transform = transforms.ToTensor()
# transforms.ToTensor() converts image pixels
# into tensor values between 0 and 1.

# ============================================
# DOWNLOAD MNIST DATASET
# ============================================
# root = where the data will be stored 
# train = True means we want the training set
# transform = the transformation we want to apply to the images
# download = True means we want to download the dataset if it's not already present

train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
# MNIST contains handwritten digits:
# 0,1,2,3,4,5,6,7,8,9
#
# This dataset is used by beginners to learn AI.

# ============================================
# CREATE DATALOADER
# ============================================

# DataLoader feeds data to the neural network.
#
# batch_size=64 means: process 64 images at one time.
# shuffle=True means: randomly shuffle images. Shuffling helps better learning.

train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# ============================================
# BUILD NEURAL NETWORK
# ============================================
# Every PyTorch neural network is usually created by inheriting from nn.Module

class NeuralNetwork(nn.Module):
    # As __init__ runs when we create an instance of the class, we define the layers of the network here.
    def __init__(self):
        # As we are inheriting from nn.Module, we need to call the parent class's __init__ method.
        super().__init__()

        # MNIST images are 28x28 grey scaled pixels however NN expects 1D vectors. So we have to flatten the matrix into a vector of 784 values (28*28=784).
        # Each Pixel in the image is 1 byte = 8 bits = 2^8 - 1 = 0 -255 values.
        # NN works better with smaller values. So instead of 0to 255 values, we consider 255 as 1. Now we have to convert 0 to 255 into 0 to 1.
        # Hence, we divide each pixel value by 255 to get a value between 0 and 1. This is done by transforms.ToTensor() which we applied earlier.
        
        self.flatten = nn.Flatten() # This layer will flatten the 28x28 image into a 784- 1 dimensional vector.

        # nn.Sequential allows us to stack layers. one after another.
        #
        # Think: Input Layer → Hidden Layer → Output Layer
        self.model = nn.Sequential(nn.Linear(28*28, 128),    # Input layer (of 784 values) to hidden layer (of 128 neurons). Neurons are like mini decision makers (small functions) in the network.
                                                             # Number of neurons can be any number. 128 is a common choice for a small network.
                                                             # Each neuron takes ALL the 784 input values, applies a weighted sum, and produces an output.
            nn.ReLU(),                                       # Rectified Linear Unit- ReLU(x)=max(0,x) Activation function (introduces non-linearity). This islike a filter that check if the output of the neuron is positive. If it's positive, it keeps it; if it's negative, it sets it to zero. This helps the network learn complex patterns.
            nn.Linear(128, 10) )                             # Hidden layer LEARNS , to output layer (128 → 10). 128 because we have 128 neurons in the hidden layer. 
                                                             # 10 because we have 10 classes (digits 0-9) to predict. makes FINAL CLASSIFICATION.
    def forward(self, x):
        x = self.flatten(x)
        return self.model(x)

# Create model
model = NeuralNetwork()

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
epochs = 5

for epoch in range(epochs):

    for images, labels in train_dataloader:

        outputs = model(images)

        loss = criterion(outputs, labels)

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

    print(f"Epoch {epoch+1}, Loss: {loss.item()}")

print("Training complete!")