import numpy as np
import os

from PIL import Image

from torch.utils.data import Dataset

from torchvision import transforms

from torchvision.transforms import Resize

EXTENSIONS = ['.jpg', '.png']

class cityscapes(Dataset):

    def __init__(self, root : str, co_transform = False, subset = 'train'):
        self.images_root = os.path.join(root, 'leftImg8bit/')
        self.labels_root = os.path.join(root, 'gtFine/')
        size = (512, 512)  #Transform image from 1024x2048 to 512x1024
        self.transform = True
        
        self.images_root += subset
        self.labels_root += subset

        print (self.images_root)
        self.filenames = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(self.images_root)) for f in fn if self.is_image(f)]
        self.filenames.sort()

        self.filenamesGt = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(self.labels_root)) for f in fn if self.is_label(f)]
        self.filenamesGt.sort()

        
        # Define the default transformation pipeline
        self.transform = transforms.Compose([
            Resize(size),
            transforms.ToTensor()  # Convert image to tensor
        ])

        self.label_transform = transforms.Compose([
            Resize(size),
            transforms.ToTensor() #Label to tensor
        ])
    
    def is_label(self, filename : str):
        return filename.endswith("_labelTrainIds.png")
    
    def is_image(self, filename : str):
        return any(filename.endswith(ext) for ext in EXTENSIONS)
    
    def load_image(self, file):
        return Image.open(file)

    def __getitem__(self, index : int):
        filename = self.filenames[index]
        filenameGt = self.filenamesGt[index]

        with open(filename, 'rb') as f:
            image = self.load_image(f).convert('RGB')
        with open(filenameGt, 'rb') as f:
            label = self.load_image(f).convert('P')

        image = self.transform(image)
        label = self.transform(label)

        return image, label

    def __len__(self) -> int:
        return len(self.filenames)