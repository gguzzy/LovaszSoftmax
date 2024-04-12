# Necessary imports for our script
import sys
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import cv2
from tqdm import tqdm
import lovasz_losses as L
import time
from torch.profiler import profile, record_function, ProfilerActivity

# Function to add the directory containing demo_helpers to our system path
def add_demo_helpers_to_path():
    """
    Adds the demo_helpers directory to the system path to allow imports.
    Assumes this script is located two levels inside the target directory.
    """
    file_dir = os.path.dirname(os.path.abspath(__file__))  # Current file directory
    parent_dir = os.path.dirname(os.path.dirname(file_dir))  # Parent directory
    sys.path.insert(0, os.path.join(parent_dir, 'demo_helpers'))

add_demo_helpers_to_path()
from demo_utils import *  # Import utilities from demo_helpers

# Setting up the device for PyTorch operations
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Parameters for the dataset and model
NUM_CLASSES = 3  # Including the "void" class
BATCH_SIZE = 5
IMG_HEIGHT, IMG_WIDTH = 200, 200

def generate_images(height, width, num_images):
    """
    Generates simple images with geometric shapes using OpenCV.
    Replace or extend this logic according to your specific needs.
    """
    images = []
    for _ in range(num_images):
        img = np.zeros((height, width), dtype=np.uint8)
        # Example shape: draw a rectangle representing one class
        # cv2.rectangle(img, (50, 50), (150, 150), 1, -1)
        images.append(img)
    return images

# Generate labels using the function above or your own custom function
labels_ = generate_images(IMG_HEIGHT, IMG_WIDTH, BATCH_SIZE)
labels = torch.stack([torch.from_numpy(img).long() for img in labels_]).to(device)

class Model(nn.Module):
    """
    A simple convolutional model.
    Modify as needed for your application.
    """
    def __init__(self, in_channels, out_channels):
        super(Model, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        return x + self.conv(x)

# Initialize model, optimizer, and other training elements
model = Model(NUM_CLASSES, NUM_CLASSES).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.005)

# Placeholder for input data; replace with your data loading logic as necessary
features = torch.randn(BATCH_SIZE, NUM_CLASSES, IMG_HEIGHT, IMG_WIDTH, device=device)

# Training loop setup and execution
def train_model():
    """
    Main training loop.
    Measures and reports training performance and memory usage.
    """
    start_time = time.time()
    initial_memory = torch.cuda.memory_allocated(device)
    print(f"Initial CUDA memory: {initial_memory} bytes")

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], profile_memory=True) as prof:
        with record_function("Training"):
            for epoch in tqdm(range(10), desc="Training loop"):
                start_epoch_time = time.time()
                optimizer.zero_grad()
                outputs = model(features)
                loss = L.lovasz_softmax(outputs, labels, ignore=255)
                loss.backward()
                optimizer.step()

                epoch_duration = time.time() - start_epoch_time
                print(f"Epoch duration: {epoch_duration:.3f} s")

                current_memory = torch.cuda.memory_allocated(device)
                print(f"CUDA memory after epoch {epoch}: {current_memory} bytes")

    # total_training_time = time.time() - start_time
    # final_memory = torch.cuda.memory_allocated(device)
    # print(f"Final CUDA memory: {final_memory} bytes")
    # print(f"Total training time: {total_training_time:.2f} s")
    print("Training completed using Lovasz Softmax Loss")
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

if __name__ == "__main__":
    train_model()
