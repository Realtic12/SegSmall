import torch
import numpy as np
import argparse
import random #Random to flip an image
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'model')))  #import folder

from SegSmall import SegSmall

class TrainNet(): #look if in the future it is necessary
    def __init__(self, args, device) -> None:
       #Check device in use
        self.device = device
        self.epochs = args.epochs

        self.net = SegSmall(args).to(device) #placed the job on the available device, preferibably in gpu

        self.train()

    
    def train(self):
        """
            Interesting: https://stackoverflow.com/questions/51433378/what-does-model-train-do-in-pytorch
        """
        self.net.train() #inform model we are in training. Some layers behave different from training an evaluation 
        #self.net.eval() 

        init_epoch = 0
        step_epochs = 1
        print("Start training....!!!!")
        for epoch in range(init_epoch, self.epochs, step_epochs):
            print(f"Training epoch: {epoch}")
            pass


def main():

    #Parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", help="Dataset to train", default='../dataset/cityscapes/data')
    parser.add_argument("--batch_size", help = "Batch size", type = int, default = 2)
    parser.add_argument("--epochs", help="Number of epochs to train", type = int, default = 3)
    parser.add_argument("--num-classes", help='SegSmall classes. Required for loading the base network', default=1000, type=int)
    '''parser.add_argument("--in_channels", help = "Number of input channels for the network", default = 3, type = int)
    parser.add_argument("--out_channels", help = "Number of output channels for the network", default = 64, type = int)
    parser.add_argument("--kernel_size", help = "#size of the convolutional kernel, is cuadratic", default = 3, type = int)
    parser.add_argument("--stride", help = "Strides (cuadratic)", default = 3, type = int)'''
    args = parser.parse_args()

    num_gpus = torch.cuda.device_count()
    device = 'cuda' if num_gpus > 0 else 'cpu'
    if(device == 'cuda'):
        print(f"You are using cuda with: {num_gpus} gpu")
    print(f"You are using: {device}")

    #Check dataset
    assert os.path.exists(args.dataset), "Error: dataset directory does not exist"

    train_net = TrainNet(args,device)

if __name__ == "__main__":
    main()