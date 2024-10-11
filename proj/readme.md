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
