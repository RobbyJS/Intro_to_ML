import torch 
import os

from torch import nn
from torch import optim

import torch.nn.functional as F
from torchvision import models, datasets, transforms



from workspace_utils import active_session
import helpers as h

# Function to build the datasets and dataloaders from the folder
def databuild(data_dir,batch_size):
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'



    # Define your transforms for the training, validation, and testing sets
    train_transforms = transforms.Compose([
                                        transforms.RandomRotation(30),
                                        transforms.RandomResizedCrop(224),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])
    ])
    test_transforms = transforms.Compose([
                                        transforms.Resize(255),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])
    ])

    # Load the datasets with ImageFolder
    train_dataset = datasets.ImageFolder(train_dir,transform=train_transforms)
    valid_dataset = datasets.ImageFolder(valid_dir,transform=test_transforms)
    test_dataset = datasets.ImageFolder(test_dir,transform=test_transforms)
    image_datasets = {'train':train_dataset,'valid':valid_dataset,'test':test_dataset}

    # Using the image datasets and the trainforms, define the dataloaders
    # dataloaders = 
    trainloader = torch.utils.data.DataLoader(train_dataset,batch_size=batch_size,shuffle=True)
    validloader = torch.utils.data.DataLoader(valid_dataset,batch_size=batch_size)
    testloader = torch.utils.data.DataLoader(test_dataset,batch_size=batch_size)
    dataloaders = {'train':trainloader,'valid':validloader,'test':testloader}

    return image_datasets, dataloaders


# Classifier class
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

# Function to build a model from a set of specified parameters
def model_construction(arch,out_size,hidden_layers,drop_p=0.5,**kwargs):
    
    model = getattr(models, arch)(pretrained=True)
    
    if 'input_size' in kwargs:
        in_size = kwargs['input_size']
    else:
        for param in model.classifier.parameters():
            in_size = param.shape[1]
            break
    
    classifier = Classifier(in_size,out_size,hidden_layers,drop_p)

    # avoid base model from being trained
    for param in model.parameters():
        param.requires_grad = False
    
    # update classifier of model
    model.classifier = classifier
    return model, in_size

# Function to train a model JUST COPIED
def train_model(model,dataloader,epochs,device,l_r):
    # progress bar: https://stackoverflow.com/questions/4897359/output-to-the-same-line-overwriting-previous-output
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr = l_r)
    
    steps = 0
    

    train_losses , valid_losses , accuracy_l = [], [], []
    len_v_dl_t = len(dataloader['train'])
    len_v_dl = len(dataloader['valid'])
    print_every = max(epochs,int(round(epochs*len_v_dl_t/20)))
    #print_every = 20
    p_bar_l = {'print':min(20,print_every),'epochs':max(20,epochs),'batch':20}

    with active_session():
        running_loss = 0
        
        if os.name == 'posix':
            _ = os.system('clear')
        else:
            _ = os.system('cls')

        for e in range(epochs):
            # print("Iterating in ",list(range(epochs)))
            # print("On epoch number",e)
            h.printProgressBar(e, epochs, prefix = 'Epochs:', suffix = 'Complete', length = p_bar_l['epochs'],printEnd="\n")
            steps = 0
            # images,labels = next(iter(dataloader['train']))
            # if torch.is_tensor(images):

            h.printProgressBar(0, print_every, prefix = 'Next print:', suffix = 'Complete', length = p_bar_l['print'])
            for images,labels in dataloader['train']:
                steps+=1
                

                # Move images and labels tensors to device in use
                images, labels = images.to(device), labels.to(device)
                
                
                optimizer.zero_grad()

                log_ps = model.forward(images)
                loss = criterion(log_ps,labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                

                if steps % print_every == 0:
                    valid_loss = 0
                    accuracy = 0
                    model.eval()
                    with torch.no_grad():
                        for images, labels in dataloader['valid']:
                        # images,labels = next(iter(dataloader['valid']))
                        # if torch.is_tensor(images):
                            # Move images and labels tensors to device in use
                            images, labels = images.to(device), labels.to(device)

                            log_ps = model.forward(images)
                            batch_loss = criterion(log_ps,labels)

                            valid_loss +=batch_loss.item()

                            # Accuracy
                            ps = torch.exp(log_ps)
                            top_p, top_class = ps.topk(1,dim=1)
                            equals = top_class == labels.view(*top_class.shape)
                            accuracy += torch.mean(equals.type(torch.FloatTensor))

                    running_loss /= print_every            
                    valid_loss /= len_v_dl
                    accuracy /= len_v_dl

                    train_losses.append(running_loss)
                    valid_losses.append(valid_loss)
                    accuracy_l.append(accuracy)

                    if os.name == 'posix':
                        _ = os.system('clear')
                    else:
                        _ = os.system('cls')
                    

                    h.pprint_train(e,epochs,steps,len_v_dl_t,print_every,train_losses,valid_losses,accuracy_l)
                    h.printProgressBar(e, epochs, prefix = 'Epochs:', suffix = 'Complete', length = p_bar_l['epochs'],printEnd="\n")
                    h.printProgressBar(steps, len_v_dl_t, prefix = 'Batch:', suffix = 'Complete', length = p_bar_l['batch'],printEnd="\n")
                    h.printProgressBar(0, len_v_dl, prefix = 'Next print:', suffix = 'Complete', length = p_bar_l['print'])


                    # print("Epoch: {}/{}.. ".format(e+1, epochs),
                    #     "Step: {}.. ".format(steps),
                    #     "Training Loss: {:.3f}.. ".format(running_loss),
                    #     "Valid. Loss: {:.3f}.. ".format(valid_loss),
                    #     "Valid. Accuracy: {:.3f}".format(accuracy))
                    running_loss = 0
                    model.train()

                h.printProgressBar(steps % print_every, print_every, prefix = 'Next print:', 
                    suffix = 'Complete', length = min(50,print_every))

        else:
            h.printProgressBar(e+1, epochs, prefix = 'Epochs:', suffix = 'Complete', length = p_bar_l['epochs'],printEnd="\n")
            h.printProgressBar(steps, len_v_dl_t, prefix = 'Batch:', suffix = 'Complete', length = p_bar_l['batch'],printEnd="\n")
            print("Training completed 🚀\n")

                
    
    return model

# Function to save a checkpoint
def save_checkpoint(model,filepath,c_input_size,c_output_size,dropout_p,base_model):
    classif_info = {'input_size': c_input_size,
              'output_size': c_output_size,
              'hidden_layers': [each.out_features for each in model.classifier.hidden_layers],
               'drop_p':dropout_p}

    data_info = {'class_to_idx':model.class_to_idx}


    checkpoint = {'classif_info':classif_info,
                'base_model':base_model,
                'state_dict':model.state_dict(),
                'data_info':data_info,}


    torch.save(checkpoint, filepath)

# Function to load a checkpoint
def load_checkpoint(filepath,map_location):
    checkpoint = torch.load(filepath,map_location = map_location)
    classif_info = checkpoint['classif_info']

    model, _ = model_construction(
        checkpoint['base_model'],
        classif_info['output_size'],
        classif_info['hidden_layers'],
        drop_p = classif_info['drop_p'],
        input_size = classif_info['input_size'])
    
    model.load_state_dict(checkpoint['state_dict'])
    
    model.class_to_idx = checkpoint['data_info']['class_to_idx']

    model.idx_to_class = {v: k for k, v in model.class_to_idx.items()}
    
    return model



if __name__=='__main__':

    # img_path = "flowers/test/5/image_05166.jpg"
    # main(img_path = img_path)
    # run like this: python train.py flowers/test/5/image_05166.jpg --learning_rate 0.02
    # print('Test to concatenate',"","using an empty","","string")
    # hello = "hello"+""+"you"
    # print(hello)
    
    model = model_construction('alexnet',102,[200],drop_p=0.5)
    print(model)