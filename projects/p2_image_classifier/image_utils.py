from PIL import Image
import torch
import numpy as np


# Function to process an input image the same way than for training the neural network
# The
def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    # Load image
    im = Image.open(image)    
    
    # Resize the image
    w,h = im.size
    if w>=h:
        new_size = (int(round(256*w/h)), 256)
    else:
        new_size = (256, int(round(256*h/w)))
    im.thumbnail(new_size)    
    
    # Crop the center of the image
    new_width = new_height = 224
    width , height = new_size[0], new_size[1]
    left = int(np.ceil((width - new_width) / 2))
    right = width - int(np.floor((width - new_width) / 2))
    top = int(np.ceil((height - new_height) / 2))
    bottom = height - int(np.floor((height - new_height) / 2))
    im = im.crop((left, top, right, bottom))    
       
    # Convert image to numpy
    np_image = np.array(im)
    im.close()
    
    # Convert range to [0-1]
    np_image = np_image/255
    
    # Normalize array
    means_norm = np.array([0.485, 0.456, 0.406]).reshape((1,1,3))
    std_norm = np.array([0.229, 0.224, 0.225]).reshape((1,1,3))
    np_image = (np_image-means_norm)/std_norm
    np_image = np_image.transpose(2,0,1)
    
    # Transform to torch tensor
    torch_img = torch.from_numpy(np_image)
    
    return torch_img

# Function for predicting the top_k classes with their probabilities
def predict(image_path, model, device,cat_to_name,topk):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    # TODO: Implement the code to predict the class from an image file
    torch_img = process_image(image_path)
    model.eval()
    torch_img = torch_img.view((1,*torch_img.shape)).float()
#     print(torch_img.shape)
#     print("Type at first",type(torch_img))
    torch_img = torch_img.to(device)
#     print("Type after",type(torch_img))
    logps = model.forward(torch_img)
    ps = torch.exp(logps)
    top_p, top_classes = ps.topk(topk,dim=1)
    top_p = top_p.tolist()[0]
    top_classes = top_classes.tolist()[0]
#     print("Top classes are",top_classes)
#     print("Top prob are",top_p)
    top_classes = [model.idx_to_class[i] for i in top_classes]
    print("Predicted top class is number ",top_classes[0])
    top_classes = [cat_to_name[i] for i in top_classes]
    # top_classes = cat_to_name[top_classes]
    
    return top_p, top_classes    
