import torch
import numpy as np
import argparse
import random #Random to flip an image
import os
import sys
import time
import matplotlib.pyplot as plt
from dataset import cityscapes
from torch.utils.data import DataLoader
from PIL import Image
from torch.nn import CrossEntropyLoss
from torch.optim import Adam, lr_scheduler

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'model')))  #import folder

from SegSmall import SegSmall

class Utils:
    def __init__(self) -> None:
        pass

    """
    def __call__(self, input, target):
        print("Call Method")
        print(f"Input type: {type(input)}, Target type: {type(target)}")
        #do something to both images
        flip_ = random.random()  #Genarate random number between 0 and 1
        if(flip_ > 0.5):
            input = input.transpose(Image.FLIP_LEFT_RIGHT)
            target = target.transpose(Image.FLIP_LEFT_RIGHT)
        return input, target"""

    #Check if the folder results exists
    def __exists_results_folder(self):
        return os.path.isdir('results')
    
    def create_results_plot(self, train_loss, val_loss, name_training) -> None:
        if (not self.__exists_results_folder()):
            os.mkdir('results')
        plt.figure(figsize=(10, 5))
        epochs = range(1, len(train_loss) + 1)
        plt.plot(epochs, train_loss, label='Training Loss')
        plt.plot(epochs, val_loss,   label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Epochs vs Loss')
        plt.legend()
        plt.grid(True)
    
        # Set x-ticks to be every epoch, ensuring they are integers
        plt.xticks(ticks=list(epochs))
    
        # Save the plot
        plt.savefig(f'results/training_validation_loss_{name_training}.png')
        plt.close()
    

class TrainNet(): #look if in the future it is necessary
    def __init__(self, args, device) -> None:
        #Necessary variables
        self.device = device
        self.epochs = args.epochs
        self.num_classes = args.num_classes
        self.device = device
        self.dataset_path = args.dataset_path
        self.save_dir = "./train_data"
        self.batch_size = args.batch_size
        self.weight = torch.ones(self.num_classes)
        self.is_resume_train = args.resume_train
        self.best_val_loss = 1 #dummy value to save the weights of the network
        self.name_training = args.train_name
        self.utils = Utils()

        torch.autograd.set_detect_anomaly(True)  #Detailer of error messages

        #Create folder to save the data after training
        if not os.path.exists(self.save_dir):
            os.makedirs("train_data")

        self.__check_empty_directories(args.dataset_path)
        self.net = SegSmall(args.num_classes).to(self.device) #placed the job on the available device, preferibably in gpu

    #Private method
    def __transform_images_randomly(self):
        dataset_train = cityscapes(self.dataset_path, subset = 'train')
        dataset_val = cityscapes(self.dataset_path, subset = 'val')
        return dataset_train, dataset_val

    #Private method
    def __check_empty_directories(self, dataset_path) -> None:
        for root, dirs, files in os.walk(dataset_path):
            for name in dirs:
                dir_path = os.path.join(root, name)
                if not os.listdir(dir_path):  # Check if the directory is empty
                    raise Exception(f"There's one empty directory {dir_path}")
                
    
    def resume_training(self, optimizer, checkpoint_file) -> None:
        filenameCheckpoint = self.save_dir + '/' + checkpoint_file
        
        #Check if the file exists, although the user wants to reignite the training
        if not os.path.exists(filenameCheckpoint):
            raise Exception("There's no checkpoint")
        
        filenameCheckpoint = self.save_dir + checkpoint_file
        checkpoint = torch.load(filenameCheckpoint)
        start_epoch = checkpoint['epoch']
        self.net.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        best_acc = checkpoint['best_acc']
        print("=> Loaded checkpoint at epoch {})".format(checkpoint['epoch']))

    
    def save_checkpoint(self, epoch, optimizer, val_loss, is_best=False) -> None:
        checkpoint = {
            'epoch': epoch,
            'state_dict': self.net.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_val_loss': val_loss
        }
        torch.save(checkpoint, os.path.join(self.save_dir, 'checkpoint.pth.tar'))
        if is_best:
            torch.save(self.net.state_dict(), os.path.join(self.save_dir, 'best_weights.pth'))
    
    def __empty_cuda_memory(self):
        torch.cuda.empty_cache()
    
    def train(self) -> None:
        """
            Interesting: https://stackoverflow.com/questions/51433378/what-does-model-train-do-in-pytorch
        """
        data_train, data_val = self.__transform_images_randomly()
        loader_train = DataLoader(data_train, num_workers=4, batch_size=self.batch_size, shuffle=True)
        loader_val = DataLoader(data_val, num_workers=4, batch_size=self.batch_size, shuffle=False)
        
        self.weight = self.weight.to(self.device)
        criterion = CrossEntropyLoss(weight = self.weight)

        optimizer = Adam(self.net.parameters(), lr=5e-4, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-4)

        checkpoint_file = 'checkpoint.pth.tar'
        if self.is_resume_train:
            self.resume_training(optimizer, checkpoint_file)

        lambda1 = lambda epoch: pow((1 - ((epoch - 1) / self.epochs)), 0.9)
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)

        init_epoch = 1
        step_epochs = 1

        #Values to store loss in training and validation
        train_loss = []
        val_loss = []

        self.time_train_sec = [] #Record start time

        self.__empty_cuda_memory()
        print(f"Start training using: {self.epochs} epochs, {self.batch_size} batch size!!!!!")
        for epoch in range(init_epoch, self.epochs + 1, step_epochs):
            print(f"Training epoch: {epoch}")

            self.net.train()  # Set the model to training mode
            
            init_time_sec = time.time()
            epoch_loss_train = []
            for batch in loader_train:
                inputs, targets = batch
                targets = targets.squeeze(1) #erase the first value

                #print(f"Inputs size {inputs.size()}")
                #print(f"Target size {targets.size()}")
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                # - Input [batch_size, channels, heigh, weight]
                # - Target [batch_size, height, weight]
                assert inputs.size(2) == targets.size(1), "Input and target height mismatch"
                assert inputs.size(3) == targets.size(2), "Input and target width mismatch"
                
                outputs = self.net(inputs)
                
                optimizer.zero_grad()
                loss = criterion(outputs, targets.long())
                loss.backward()
                optimizer.step() 

                epoch_loss_train.append(loss.item())

            avg_train_loss = np.mean(epoch_loss_train)
            train_loss.append(avg_train_loss)
            print(f"Epoch [{epoch}/{self.epochs}], Loss: {avg_train_loss:.4f}")

            #Calculate time for each epoch
            time_train_epoch_sec = time.time() - init_time_sec
            print(f"Time Train epoch: {time_train_epoch_sec:.2f} s")
            init_time_sec = time_train_epoch_sec #Now the iniial time is the end of the last epoch
            self.time_train_sec.append(time_train_epoch_sec)  #Save it for further evaluation

            # Validation step
            print(f"Validating epoch: {epoch}")
            self.net.eval()
            self.__empty_cuda_memory()
            with torch.no_grad():
                val_loss_epoch = []
                for batch in loader_val:   
                    inputs, targets = batch
                    targets = targets.squeeze(1) #erase the first value
                    inputs, targets = inputs.to(self.device), targets.to(self.device)
                    outputs = self.net(inputs)
                    loss = criterion(outputs, targets.long())
                    val_loss_epoch.append(loss.item())

                avg_val_loss = np.mean(val_loss_epoch)
                val_loss.append(avg_val_loss)
                print(f"Validation Loss: {avg_val_loss:.4f}")
            
            # Save checkpoint and weights
            is_best = avg_val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = avg_val_loss
            self.save_checkpoint(epoch, optimizer, avg_val_loss, is_best)

            scheduler.step()

        self.utils.create_results_plot(train_loss, val_loss, self.name_training)

    #Return training time in seconds
    def get_training_time(self):
        train_sec = 0

        #Iterate across the array to sum the total time of epochs
        for i in range(len(self.time_train_sec)):
            train_sec = train_sec + self.time_train_sec[i]
        return train_sec

def main():

    #Parser
    default_path_dataset = '../dataset/cityscapes/data'

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path",  type = str,   default = default_path_dataset, help="Dataset to train")
    parser.add_argument("--batch_size",    type = int,   default = 1,                    help = "Batch size") #after everything works, change it to 12, Warning with CUDA running out of memory
    parser.add_argument("--epochs",        type = int,   default = 2,                    help="Number of epochs to train")
    parser.add_argument("--num-classes",   type = int,   default = 5,                   help='SegSmall classes. Required for loading the base network')
    #parser.add_argument("--in_channels",   type = int,   default = 3,                   help = "Number of input channels for the network")
    #parser.add_argument("--out_channels",  type = int,   default = 64,                  help = "Number of output channels for the network")
    parser.add_argument("--kernel_size",   type = int,   default = 3,                    help = "Size of the convolutional kernel, shoudl be cuadratic")
    parser.add_argument("--stride",        type = int,   default = 3,                    help = "Strides (cuadratic)")
    parser.add_argument("--learning_rate", type = float, default = 0.001,                help = "Learning rate for the training")
    parser.add_argument("--resume_train",  type = bool,  default = False,                help = "Resume training from a checkpoint")
    parser.add_argument("--train_name",    type = str,   required = True,                help = "Training name to differentiate among them")
    args = parser.parse_args()

    num_gpus = torch.cuda.device_count()
    device = 'cuda' if num_gpus > 0 else 'cpu'
    if(device == 'cuda'):
        print(f"You are using cuda with: {num_gpus} gpu")
    else:
        print(f"You are using: {device}")

    #Check dataset
    assert os.path.exists(args.dataset_path), "Error: dataset directory does not exist"

    try: 
        train_net = TrainNet(args, device)
        train_net.train()
        print(f"Training time {train_net.get_training_time():.2f}")
        print("------------------------")
        print("Train finished!!!")
        
    except Exception as e:
        print(e)

if __name__ == "__main__":
    main()