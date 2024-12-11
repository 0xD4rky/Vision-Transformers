import json, os, math
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.nn import functional as F
import torchvision
import torchvision.transforms as transforms

from ViT import Classification

def save_experiment(experiment_name,config,model,train_losses,test_losses,accuracies,base_dir = "experiments"):
    outdir = os.path.join(base_dir,experiment_name)
    os.makedirs(outdir, exist_ok = True)
    
    configfile = os.path.join(outdir,'config.json')
    with open(configfile, 'w') as f:
        json.dump(config,f,sort_keys = True,indent = 4)
    
    jsonfile = os.path.join(outdir, 'metrics.json')
    with open(jsonfile, 'w') as f:
        data = {
            'train_losses': train_losses,
            'test_losses': test_losses,
            'accuracies': accuracies,
        }
        json.dump(data, f, sort_keys=True, indent=4)

    save_checkpoint(experiment_name,model,"final",base_dir = base_dir)
    
def save_checkpoint(experiment_name, model, epoch, base_dir="experiments"):
    outdir = os.path.join(base_dir, experiment_name)
    os.makedirs(outdir, exist_ok=True)
    cpfile = os.path.join(outdir, f'model_{epoch}.pt')
    torch.save(model.state_dict(), cpfile)
    
def load_experiment(experiment_name,checkpoint_name="model_final.pt",base_dir = "experiments"):
    outdir = os.path.join(base_dir,experiment_name)
    configfile = os.path.join(outdir,'config.json')
    with open(configfile,'r') as f:
        config = json.load(f)
        
    jsonfile = os.path.join(outdir, 'config.json')
    with open(jsonfile,'r') as f:
        data = json.load(f)
    train_losses = data['train_losses']
    test_losses = data['test_losses']
    accuracies = data['accuracies']
    # Load the model
    model = Classfication(config)
    cpfile = os.path.join(outdir, checkpoint_name)
    model.load_state_dict(torch.load(cpfile))
    return config, model, train_losses, test_losses, accuracies

