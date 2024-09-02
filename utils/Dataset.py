import numpy as np
import os
from PIL import Image, ImageOps

from torch.utils.data import Dataset
from torchvision.transforms import Resize, ToTensor

import torch
import random

EXTENSIONS = ['.jpg', '.png']

class cityscapes(Dataset):

    def __init__(self, root : str, co_transform=None, subset='train') -> None:
        self.images_root = os.path.join(root, 'leftImg8bit/')
        self.labels_root = os.path.join(root, 'gtFine/')
        
        self.images_root += subset
        self.labels_root += subset

        print (self.images_root)
        #self.filenames = [image_basename(f) for f in os.listdir(self.images_root) if is_image(f)]
        self.filenames = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(self.images_root)) for f in fn if self.__is_image(f)]
        self.filenames.sort()

        #[os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(".")) for f in fn]
        #self.filenamesGt = [image_basename(f) for f in os.listdir(self.labels_root) if is_image(f)]
        self.filenamesGt = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(self.labels_root)) for f in fn if self.__is_label(f)]
        self.filenamesGt.sort()

        assert len(self.filenames) == len(self.filenamesGt), "Mismatch between number of images and labels."

        self.co_transform = co_transform # ADDED THIS

    
    def __is_label(self, filename : str) -> bool:
        return filename.endswith("_labelTrainIds.png")


    def __is_image(self, filename : str) -> bool:
        return any(filename.endswith(ext) for ext in EXTENSIONS)


    def __getitem__(self, index : int):
        filename = self.filenames[index]
        filenameGt = self.filenamesGt[index]

        image = Image.open(filename).convert('RGB')
        label = Image.open(filenameGt).convert('P')

        if self.co_transform is not None:
            image, label = self.co_transform(image, label)

        return image, label

    def __len__(self):
        return len(self.filenames)
    

class MyCoTransform(object):
    def __init__(self, augment=True, size=(512, 256)):
        self.augment = augment
        self.size = size

    def __call__(self, input, target):
        # do something to both images
        input =  Resize(self.size, Image.BILINEAR)(input)
        target = Resize(self.size, Image.NEAREST)(target)

        if(self.augment):
            # Random hflip
            hflip = random.random()
            if (hflip < 0.5):
                input = input.transpose(Image.FLIP_LEFT_RIGHT)
                target = target.transpose(Image.FLIP_LEFT_RIGHT)
            
            #Random translation 0-2 pixels (fill rest with padding
            transX = random.randint(-2, 2) 
            transY = random.randint(-2, 2)

            input = ImageOps.expand(input, border=(transX,transY,0,0), fill=0)
            target = ImageOps.expand(target, border=(transX,transY,0,0), fill=255) #pad label filling with 255
            input = input.crop((0, 0, input.size[0]-transX, input.size[1]-transY))
            target = target.crop((0, 0, target.size[0]-transX, target.size[1]-transY))   

        input = ToTensor()(input)
        target = torch.tensor(np.array(target), dtype=torch.long)

        return input, target