# Imports
import numpy as np
import matplotlib.pyplot as plt
import argparse

import torch
import json
# from numba import jit
# from numba import cuda

import torch_utils as tu
import image_utils as iu
import helpers as h


def_img_path = "flowers/test/5/image_05186.jpg"



# def main(img_path = def_img_path):
def main():
    print("I'm in")



    parser = argparse.ArgumentParser(description='Short sample app',prefix_chars='-',)

    parser.add_argument("path", help="Insert path to image")
    parser.add_argument("checkpoint", help = "Insert path to model checkpoint")
    parser.add_argument('--gpu', action='store_true',
                    default=False,
                    help='Use GPU for inference',
                    dest="gpu_on")
    parser.add_argument('--top_k', action="store",
                     type=int,
                     default=1,
                     help="Number of classes and probabilities to be provided in prediction")

    parser.add_argument('--category_names', action="store",
                     default="cat_to_name.json",
                     help="Path to file containing the mapping of labels to category names")
                     

    # parser.add_argument('--path', action="store",
    #                 dest="path")

    # print(parser.parse_args(["path","checkpoint","gpu"]))

    results_p = parser.parse_args()

    img_path = results_p.path
    chkp_path = results_p.checkpoint
    
    print("I got this image path:",img_path)
    print("I got this checkpoint path:",chkp_path)
    
    if results_p.gpu_on:
        gpu_on = results_p.gpu_on
        print("GPU set on:",gpu_on)
    
    # if results_p.top_k:
        # top_k = results_p.top_k
    print("Number of classes selected for prediction:",results_p.top_k)
    print("File selected for mapping:",results_p.category_names)
    print("\n\n")
    # print(torch.cuda.current_device())
    # print(torch.cuda.get_device_name(0))
    

    # Conditional to set the import to gpu or cpu
    if torch.cuda.is_available():
        map_location=lambda storage, loc: storage.cuda()
    else:
        map_location='cpu'

    print(map_location)

    # loading a torch checkpoint
    model = tu.load_checkpoint(chkp_path,map_location)

    # Transform the input image
    # torch_img = iu.process_image(img_path)

    # fig, ax = plt.subplots(figsize = (7,7))
    # ax.grid(None)
    # ax = iu.imshow(torch_img,ax)
    # plt.show()
    # time.sleep(10)
    # plt.close(fig=fig)
    print("I went through close command")

    with open(results_p.category_names, 'r') as f:
        cat_to_name = json.load(f)



    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    # Predict the most likely classes and the probabilities associated
    top_p, top_classes = iu.predict(img_path, model,device,cat_to_name,results_p.top_k)

    print("\n")
    print("Below are the predictions obtained with the model in",chkp_path,":\n")
    h.pprint(top_classes,top_p)
    print("\n")

    


if __name__=='__main__':

    # img_path = "flowers/test/5/image_05166.jpg"
    # main(img_path = img_path)
    # run like this: python predict.py flowers/test/5/image_05166.jpg checkpoint_vgg16_class1.pth --top_k 4
    main()