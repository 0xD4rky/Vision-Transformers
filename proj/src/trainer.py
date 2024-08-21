import torch
import torch.nn as nn
import argparse

from data import *
from utils import save_checkpoint, save_experiment
from ViT import Classification

config = {
    "patch_size": 4,  # Input image size: 32x32 -> 8x8 patches
    "vector_dim": 48,
    "num_hidden_layers": 4,
    "num_attention_heads": 4,
    "hidden_size": 4 * 48, # 4 * hidden_size
    "hidden_dropout_prob": 0.0,
    "attention_probs_dropout_prob": 0.0,
    "initializer_range": 0.02,
    "image_size": 32,
    "num_classes": 10, # num_classes of CIFAR10
    "num_channels": 3,
    "qkv_bias": True,
}

assert config["vector_dim"] % config["num_attention_heads"] == 0
assert config['hidden_size'] == 4 * config['vector_dim']
assert config['image_size'] % config['patch_size'] == 0

class Trainer:
    """
    simple trainer block
    """
    
    def __init__(self,model,optimizer,loss_fn,exp_name,device):
        self.model = model.to(device)
        self.optim = optimizer
        self.loss  = loss_fn
        self.exp_name = exp_name
        self.device = device
        
    def train(self,train_loader,test_loader,epochs,save_exp_every_n_epochs = 0):
        
        train_losses, test_losses, accuracies = [],[],[]
        
        for i in range(epochs):
            train_loss = self.train_epoch(train_loader)
            accuracy = test_loss = self.evaluate(test_loader)
            train_losses.append(train_loss)
            test_losses.append(test_loss)
            accuracies.append(accuracy)
            print(f"Epoch {i+1}, Train loss: {train_loss:.4f}, Test loss: {test_loss:.4f}, Accuracy: {accuracy:.4f}")
            if save_exp_every_n_epochs > 0 and (i+1) % save_exp_every_n_epochs == 0 and i+1 != epochs:
                print('\tSave checkpoint at epoch',i+1)
                save_checkpoint(self.exp_name, self.model, i+1)
        
        save_experiment(self.exp_name, self.model, i+1)
    
    def train_epoch(self,train_loader):
        
        self.model.train()
        total_loss = 0
        for batch in train_loader:
            
            batch = [t.tp(self.device) for t in batch]
            images, labels = batch
            self.optimizer.zero_grad()
            loss = self.loss(self.model(images)[0], labels)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()*len(images)
            
        return total_loss/ len(train_loader.dataset)
    
    
    @torch.no_grad()
    def evaluate(self,test_loader):
        self.model.eval()
        total_loss = 0
        correct = 0
        
        with torch.no_grad():
            
            for batch in test_loader():
                batch = [t.to(self.device) for t in batch]
                images, labels = batch
                
                logits,_ = self.model(images)
                
                loss = self.loss(logits,labels)
                total_loss += loss.item() * len(images)
                
                predictions = torch.argmax(logits, dim = 1)
                correct = torch.sum(predictions == labels).item()
                
        accuracy = correct/ len(test_loader.dataset)
        avg_loss = total_loss / len(test_loader.dataset)
        return accuracy, avg_loss

        