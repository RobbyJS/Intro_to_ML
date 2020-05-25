# Imports
import numpy as np
import matplotlib.pyplot as plt
import argparse
import pathlib

import os

import torch
from torch import nn

import json

import torch_utils as tu


def main():
    print("I'm in")

    parser = argparse.ArgumentParser(description='Short sample app',prefix_chars='-',)

    parser.add_argument("data_dir", help="Insert path to images folder (eg. flowers)")
    
    parser.add_argument('--gpu', action='store_true',
                    default=False,
                    help='Use GPU for training',
                    dest="gpu_on")
    
    parser.add_argument('--epochs', action="store",
                     type=int,
                     default=1,
                     help="Number of epochs to complete in training")
    
    parser.add_argument('--learning_rate', action="store",
                     type=float,
                     default=0.001,
                     dest="l_r",
                     help="Learning rate to use in training")

    parser.add_argument('--save_dir', action="store",
                     default="",
                     help="""Path to folder to save the checkpoint. 
                     eg: vgg16/1
                     Folder will be created if it does not exist.""")

    parser.add_argument('--arch', action="store",
                     default="vgg16",
                     help="Base model that has to be used, eg. vgg16 or alexnet")
                     
                     
    parser.add_argument('--hidden_units', action="append",
                     default=[],
                     type=int,
                     help="""Use this option to store the number of perceptrons in the succesive
                                hidden layers""")                     
    


    results_p = parser.parse_args()

    img_path = results_p.data_dir
    # chkp_path = results_p.checkpoint
    hidden_units = results_p.hidden_units
    print("I got this image path:",img_path)
    print("I got these epochs:",results_p.epochs)
    print("I got this learning rate:",results_p.l_r)    
    print("I got these hidden units:",hidden_units)
    if not hidden_units:
        hidden_units = [512]
    print("So I transformed it to these:",hidden_units)

    
    # chkp_path = results_p.save_dir + '/chk.pth'
    # print("I got this checkpoint path:",chkp_path)
    # directory = os.path.dirname(chkp_path)
    # print("Directory path <"+directory+">")
    # pathlib.Path(directory).mkdir(parents=True, exist_ok=True) 
    
    gpu_on = results_p.gpu_on
    print("GPU set on:",gpu_on)    

    # obtain the datasets folders and the dataloaders
    image_datasets, dataloaders = tu.databuild(img_path,10)

    # obtain number of classes in the provided dataset
    output_size = len(image_datasets['train'].class_to_idx)
    
    # build the model
    drop_p = 0.5
    model, input_size = tu.model_construction(results_p.arch,output_size,hidden_units,drop_p=drop_p)
    print(model)

    # Use GPU if requested    
    if gpu_on and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    
    print(device)

    # Training the model
    model.to(device)

    model = tu.train_model(model,dataloaders,results_p.epochs,device,results_p.l_r)

    # Saving the checkpoint
    model.class_to_idx = image_datasets['train'].class_to_idx

    # create folder if non existing

    chkp_path = results_p.save_dir + '/chk.pth'
    print("I got this checkpoint path:",chkp_path)
    directory = os.path.dirname(chkp_path)
    print("Directory path <"+directory+">")
    pathlib.Path(directory).mkdir(parents=True, exist_ok=True)

    tu.save_checkpoint(model,chkp_path,input_size,output_size,drop_p,results_p.arch) 

    




if __name__=='__main__':

    # img_path = "flowers/test/5/image_05166.jpg"
    # main(img_path = img_path)
    # run like this: python train.py flowers --learning_rate 0.02 --arch vgg13
    # print('Test to concatenate',"","using an empty","","string")
    # hello = "hello"+""+"you"
    # print(hello)
    #print("Training completed 🚀\n")
    main()
    