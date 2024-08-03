import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torchvision.transforms as T
from torchvision.utils import make_grid

# Define the convolutional layer
class PatchEmbedding(nn.Module):
    def __init__(self, num_patches, vector_dim, patch_size):
        super(PatchEmbedding, self).__init__()
        self.conv = nn.Conv2d(3, vector_dim, kernel_size=patch_size, stride=patch_size)
    
    def forward(self, x):
        x = self.conv(x)
        return x

# Create a dummy input image
input_image = torch.randn(1, 3, 224, 224)  # (batch_size, num_channels, height, width)

# Define parameters
vector_dim = 256
patch_size = 16

# Initialize the PatchEmbedding layer
patch_embedding = PatchEmbedding(3, vector_dim, patch_size)

# Apply the convolutional layer
output = patch_embedding(input_image)

# Function to visualize image patches
def visualize_patches(input_image, patch_size):
    # Unnormalize the image
    input_image = input_image.squeeze(0).permute(1, 2, 0).numpy()
    
    fig, ax = plt.subplots()
    ax.imshow(input_image)
    
    # Add grid lines to visualize patches
    for i in range(0, input_image.shape[0], patch_size):
        ax.axhline(i, color='red')
    for j in range(0, input_image.shape[1], patch_size):
        ax.axvline(j, color='red')
    
    plt.title("Input Image with Patches")
    plt.show()

# Visualize input image with patches
visualize_patches(input_image, patch_size)

# Visualize the feature maps
def visualize_feature_maps(feature_maps, num_maps_to_show=8):
    # Select the first 'num_maps_to_show' feature maps
    maps_to_show = feature_maps[0, :num_maps_to_show, :, :]
    
    # Create a grid of the feature maps
    grid = make_grid(maps_to_show.unsqueeze(1), nrow=4, normalize=True, scale_each=True)
    
    # Plot the grid
    plt.figure(figsize=(15, 15))
    plt.imshow(grid.permute(1, 2, 0).cpu().numpy())
    plt.title("Feature Maps")
    plt.axis('off')
    plt.show()

# Visualize the first few feature maps
visualize_feature_maps(output, num_maps_to_show=8)
