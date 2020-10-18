#import all libraries and configure matplotlib to display larger plots
#Here I use pytorch
import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as dset
import glob
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
import time
import albumentations as A

from torch.utils.data import DataLoader, Dataset
from PIL import Image

#check if have gpu in your laptop
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

mpl.rcParams['axes.grid'] = False
mpl.rcParams['image.interpolation'] = 'nearest'
mpl.rcParams['figure.figsize'] = 15, 25

#train_withMask & train_withoutMask stores all the image names
train_withMask = glob.glob('Dataset/Train/WithMask/*.png')
train_withoutMask = glob.glob('Dataset/Train/WithoutMask/*.png')
print("Train with mask: " + str(len(train_withMask)))
print("Train without mask: " + str(len(train_withoutMask)))

#define pytorch transforms
#PyTorch transforms module will help define all the image augmentation 
#and transforms that we need to apply to the images. Can apply different transforms -> check for documentation. at 
transform = transforms.Compose([
     transforms.ToPILImage(), #converting the image into PIL format
     transforms.Resize((224, 224)), #apply the transforms for the followings, 
     transforms.CenterCrop((100, 100)),
     transforms.RandomCrop((80, 80)),
     transforms.RandomHorizontalFlip(p=0.5),
     transforms.RandomRotation(degrees=(-90, 90)),
     transforms.RandomVerticalFlip(p=0.5),
     transforms.ToTensor(), #convert the images to tensors
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), #normalize the images
     ])


# PyTorch image augmentation module
class PyTorchImageDataset(Dataset):
    def __init__(self, train_list, transforms=None):
        self.train_data = train_list
        self.transforms = transforms
         
    def __len__(self):
        return (len(self.train_data))
    
    #reading an image from the list based on the index value. 
    #PIL Image converts the image into 3-channels RGB format. 
    #np converts the image into NumPy array and uint8 data type.
    def __getitem__(self, i):
        image = plt.imread(self.train_data[i])
        #image = Image.fromarray(image).convert('RGB')        
        image = np.asarray(image * 225).astype(np.uint8)
        if self.transforms is not None:
            image = self.transforms(image)
            
        return torch.tensor(image, dtype=torch.float)

#initialize the dataset class and prepare the data loader
#train_withMask
pytorch_dataset_train_withMask = PyTorchImageDataset(train_list=train_withMask, transforms=transform)
pytorch_dataloader_train_withMask = DataLoader(dataset=pytorch_dataset_train_withMask, batch_size=16, shuffle=True)
#train_withoutMask
pytorch_dataset_train_withoutMask = PyTorchImageDataset(train_list=train_withoutMask, transforms=transform)
pytorch_dataloader_train_withoutMask = DataLoader(dataset=pytorch_dataset_train_withoutMask, batch_size=16, shuffle=True)

# visualizing a single batch of image
def show_img(img):
    plt.figure(figsize=(18,15))
    # unnormalize
    img = img / 2 + 0.5  
    npimg = img.numpy()
    npimg = np.clip(npimg, 0., 1.)
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

#train_withMask
data_train_withMask = iter(pytorch_dataloader_train_withMask)
images_train_withMask = data_train_withMask.next()
#train_withoutMask
data_train_withoutMask = iter(pytorch_dataloader_train_withoutMask)
images_train_withoutMask = data_train_withoutMask.next()

#show images -> pythorch performed images
show_img(torchvision.utils.make_grid(images_train_withMask))
show_img(torchvision.utils.make_grid(images_train_withoutMask))


#custom dataset class for albumentations library
#different data augmwntation approach from pytroch library
class AlbumentationImageDataset(Dataset):
    def __init__(self, train_list):
        self.train_list = train_list
        self.aug = A.Compose({
        A.Resize(224, 224),
        A.HorizontalFlip(p=0.5),
        #A.RandomBrightnessContrast(always_apply=True),
        #A.RandomGamma(gamma_limit=(100, 100), always_apply=True),
        #A.RGBShift(p=0.75),
        #A.GaussNoise(p=0.25),
        A.Flip(p=0.25),
        A.Rotate(limit=(-90, 90)),
        A.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        })
         
    def __len__(self):
        return (len(self.train_list))
    
    def __getitem__(self, i):
        image = plt.imread(self.train_list[i])
        #image = Image.fromarray(image).convert('RGB')
        image = self.aug(image=np.array(image * 255))['image']
        image = np.transpose(image, (2, 0, 1)).astype(np.float32)
            
        return torch.tensor(image, dtype=torch.float)

#train_withMask
alb_dataset_train_withMask = AlbumentationImageDataset(train_list=train_withMask)
alb_dataloader_train_withMask = DataLoader(dataset=alb_dataset_train_withMask, batch_size=16, shuffle=True)
#train_withoutMask
alb_dataset_train_withoutMask = AlbumentationImageDataset(train_list=train_withoutMask)
alb_dataloader_train_withoutMask = DataLoader(dataset=alb_dataset_train_withoutMask, batch_size=16, shuffle=True)

#train_withMask
alb_data_train_withMask = iter(alb_dataloader_train_withMask)
alb_images_train_withMask = alb_data_train_withMask.next()
#train_withoutMask
alb_data_train_withoutMask = iter(alb_dataloader_train_withoutMask)
alb_images_train_withoutMask = alb_data_train_withoutMask.next()

# show images
show_img(torchvision.utils.make_grid(alb_images_train_withMask))
show_img(torchvision.utils.make_grid(alb_images_train_withoutMask))


#references link: https://debuggercafe.com/image-augmentation-using-pytorch-and-albumentations/
#references link: https://colab.research.google.com/drive/109vu3F1LTzD1gdVV6cho9fKGx7lzbFll#scrollTo=9NRlYXKQy3Kx
#references link: https://pytorch.org/docs/stable/torchvision/datasets.html

#Tensorflow: https://www.tensorflow.org/tutorials/images/data_augmentation

#Some more improvements:
#A.RandomBrightnessContrast(always_apply=True)
#A.RandomGamma(gamma_limit=(400, 500), always_apply=True)
# A.RGBShift(p=0.75)
# A.GaussNoise(p=0.25)
# A.Flip(p=0.25)
