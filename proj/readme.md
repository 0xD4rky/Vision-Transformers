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

