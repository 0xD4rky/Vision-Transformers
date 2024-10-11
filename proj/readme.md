# Building ViT from scratch

## INFO:

This project implements a Vision Transformer (ViT) from scratch using Python and PyTorch. The implementation is based on the original paper "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale" by Dosovitskiy et al. The model is trained and evaluated on the CIFAR-10 dataset.

## Project Structure:


The project consists of the following main files:

* `base.py`: Contains the GELU activation function implementation.
* `data.py`: Handles data preparation using the CIFAR-10 dataset.
* `ViT.py`: Contains the Vision Transformer model implemented from scratch.
* `trainer.py`: Implements the entire training and evaluation pipeline.
* `utils.py`: Contains utility functions for model and checkpoint management.
* visualization contains `vis.py` to visualize image patches and attention maps.

## Requirements:

```
cd proj/src
pip install -r requirements.txt
```

## Inference:

1. Clone the repo:

   ```
   git clone https://github.com/0xD4rky/Vision-Transformer.git
   cd proj/src
   ```
2. Prepare the data: The `data.py` script handles the CIFAR-10 dataset preparation. You don't need to run this separately as it will be called by the trainer.

3. Training:
   ```
   python trainer.py
   ```
   This script will train the Vision Transformer on the CIFAR-10 dataset and evaluate its performance.


## Model Architecture
The Vision Transformer (ViT) architecture is implemented in `ViT.py`. It follows the original paper's design, including:

* Patch embedding
* Positional embedding
* Transformer encoder with multi-head self-attention and feed-forward layers
* Classification head


## Training and Evaluation:

The `trainer.py` script handles both training and evaluation. It includes:

* Data loading and preprocessing
* Model initialization
* Training loop with gradient updates
* Evaluation on the test set
* Logging of training progress and results

## Utility Functions:

The `utils.py` file contains helper functions for:

* Saving and loading model checkpoints
* Logging training progress
* Any other utility functions used across the project

## Results:

(You can add information about the performance of your model on the CIFAR-10 dataset, including accuracy, training time, and any comparisons with baseline models.)

## Future Work:

* Implement data augmentation techniques to improve generalization
* Experiment with different model sizes (e.g., ViT-Base, ViT-Large)
* Try fine-tuning on different datasets
* Implement visualization tools for attention maps

