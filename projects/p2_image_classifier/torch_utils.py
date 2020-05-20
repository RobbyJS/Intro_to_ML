import torch 

from torch import nn
import torch.nn.functional as F
from torchvision import models

class Classifier(nn.Module):
    def __init__(self, input_size, output_size, hidden_layers, drop_p=0.5):
        ''' Builds a feedforward network with arbitrary hidden layers.
        
            Arguments
            ---------
            input_size: integer, size of the input layer
            output_size: integer, size of the output layer
            hidden_layers: list of integers, the sizes of the hidden layers
        
        '''
        super().__init__()
        # Input to a hidden layer
        self.hidden_layers = nn.ModuleList([nn.Linear(input_size, hidden_layers[0])])
        
        # Add a variable number of more hidden layers
        layer_sizes = zip(hidden_layers[:-1], hidden_layers[1:])
        self.hidden_layers.extend([nn.Linear(h1, h2) for h1, h2 in layer_sizes])
        
        self.output = nn.Linear(hidden_layers[-1], output_size)
        
        self.dropout = nn.Dropout(p=drop_p)
        
    def forward(self, x):
        ''' Forward pass through the network, returns the output logits '''
        
        for each in self.hidden_layers:
            x = F.relu(each(x))
            x = self.dropout(x)
        x = self.output(x)
        
        return F.log_softmax(x, dim=1)



# Function to load a checkpoint
def load_checkpoint(filepath,map_location):
    checkpoint = torch.load(filepath,map_location = map_location)
    model = getattr(models, checkpoint['base_model'])(pretrained=True)
    classif_info = checkpoint['classif_info']
    classifier = Classifier(classif_info['input_size'],
                            classif_info['output_size'],
                            classif_info['hidden_layers'],
                            classif_info['drop_p'])
    # We avoid the base model from being trained
    for param in model.parameters():
        param.requires_grad = False
    
    # We change the classifier
    model.classifier = classifier
    
    model.load_state_dict(checkpoint['state_dict'])
    
    model.class_to_idx = checkpoint['data_info']['class_to_idx']

    model.idx_to_class = {v: k for k, v in model.class_to_idx.items()}
    
    return model