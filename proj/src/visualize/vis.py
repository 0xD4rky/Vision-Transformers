import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torchvision.transforms as T
from torchvision.utils import make_grid

class PatchEmbedding(nn.Module):
    def __init__(self, num_patches, vector_dim, patch_size):
        super(PatchEmbedding, self).__init__()
        self.conv = nn.Conv2d(3, vector_dim, kernel_size=patch_size, stride=patch_size)
    
    def forward(self, x):
        x = self.conv(x)
        return x

input_image = torch.randn(1, 3, 224, 224)  # {creating a dummy image to vis}

vector_dim = 256
patch_size = 16

patch_embedding = PatchEmbedding(3, vector_dim, patch_size)

output = patch_embedding(input_image)

def visualize_patches(input_image, patch_size):
    """
    visualizing patches and attention maps
    """
    input_image = input_image.squeeze(0).permute(1, 2, 0).numpy()
    
    fig, ax = plt.subplots()
    ax.imshow(input_image)
    
    for i in range(0, input_image.shape[0], patch_size):
        ax.axhline(i, color='red')
    for j in range(0, input_image.shape[1], patch_size):
        ax.axvline(j, color='red')
    
    plt.title("Input Image with Patches")
    plt.show()

visualize_patches(input_image, patch_size)

def visualize_feature_maps(feature_maps, num_maps_to_show=8):
    maps_to_show = feature_maps[0, :num_maps_to_show, :, :]
    
    grid = make_grid(maps_to_show.unsqueeze(1), nrow=4, normalize=True, scale_each=True)
    
    plt.figure(figsize=(15, 15))
    plt.imshow(grid.permute(1, 2, 0).cpu().numpy())
    plt.title("Feature Maps")
    plt.axis('off')
    plt.show()

visualize_feature_maps(output, num_maps_to_show=8)
