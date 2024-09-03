import torch
import numpy as np
import argparse
import os
import sys
import time
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from torch.optim import Adam, lr_scheduler
from torchsummary import summary

sys.path.append("..")  #In order to use our classes

from model.SegSmall import SegSmall
from model.erfnet import Net
from utils.Utils import CheckDevice, Weights
from utils.Plot import Plot
from utils.IoU import iouCalc
from utils.Dataset import cityscapes, MyCoTransform

class TrainNet():
    def __init__(self, args, device : str) -> None:
        #Necessary variables for the implementation
        self.device = device
        self.epochs = args.epochs
        self.num_classes = args.num_classes
        self.device = device
        self.dataset_path = args.dataset_path
        self.name_training = args.train_name
        self.save_dir = f'./results/{self.name_training}'
        self.batch_size = args.batch_size
        self.model = args.model
        self.learning_rate = args.learning_rate
        self.is_resume_train = args.resume_train
        self.best_iou = 0
        self.time_train_sec = [] #Array to store training time by epoch
        self.plot = Plot(args.epochs, self.name_training)
        self.utils_weights = Weights(self.device)

        torch.autograd.set_detect_anomaly(True)  #Detailer of error messages

        #Create folder to save the weights, biases and checkpoint after training
        if not os.path.exists(self.save_dir):
            os.makedirs("train_data")

        self.__check_empty_directories(args.dataset_path)
        if(self.model == "SegSmall"):
            print("Using SegSmall model")
            self.net = SegSmall(args.num_classes).to(self.device) #placed the job on the available device, preferibably in gpu
        elif(self.model == "erfnet"):
            print("Using erfnet model")
            self.net = Net(args.num_classes).to(self.device) #placed the job on the available device, preferibably in gpu
        else:
            raise Exception("Model not found")

        #Interesting to show info about the model 
        #summary(self.net, input_size=(3, 512, 512), batch_size=self.batch_size)

    """
        Transform images randomly and create the dataset for further training and validation phase
    """
    def __transform_images_randomly(self):
        co_transform = MyCoTransform(augment=True)
        co_transform_val = MyCoTransform(augment=False)
        dataset_train = cityscapes(self.dataset_path, co_transform = co_transform, subset = 'train')
        dataset_val = cityscapes(self.dataset_path, co_transform = co_transform_val, subset = 'val')
        return dataset_train, dataset_val

    """ Check if there's any empty directory """
    def __check_empty_directories(self, dataset_path : str) -> None:
        for root, dirs, files in os.walk(dataset_path):
            for name in dirs:
                dir_path = os.path.join(root, name)
                if not os.listdir(dir_path):  # Check if the directory is empty
                    raise Exception(f"There's one empty directory {dir_path}")
    
    """
        Resume training from a checkpoint
    """
    def resume_training(self, optimizer, checkpoint_file : str) -> int:
        filenameCheckpoint = self.save_dir + '/' + checkpoint_file
        
        #Check if the file exists, although the user wants to reignite the training
        if not os.path.exists(filenameCheckpoint):
            raise Exception("There's no checkpoint")
        
        filenameCheckpoint = self.save_dir + checkpoint_file
        checkpoint = torch.load(filenameCheckpoint)
        self.net.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        self.best_iou = checkpoint['best_val_iou']
        epoch = checkpoint['epoch']
        print("=> Loaded checkpoint at epoch {})".format(checkpoint['epoch']))
        return epoch

    """
        Save the checkpoint
    """
    def save_checkpoint(self, epoch : int, optimizer, iou : float, is_best = False) -> None:
        checkpoint = {
            'epoch': epoch,
            'state_dict': self.net.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_val_iou': iou
        }
        torch.save(checkpoint, os.path.join(self.save_dir, f'{self.name_training}_checkpoint.pth.tar'))
        if is_best:
            print(f"Saving model as best with {iou:.4f} %")
            torch.save(self.net.state_dict(), os.path.join(self.save_dir, f'{self.name_training}_best_weights.pth'))
    
    """
        It is used to count the amount of learnable parameters in the model
    """
    def count_param_model(self) -> int:
        return sum(p.numel() for p in self.net.parameters() if p.requires_grad)
    
    def __empty_cuda_memory(self) -> None:
        torch.cuda.empty_cache()
    
    def get_training_time(self) -> float:
        """
            Return the amount of time needed for the training
        """
        train_sec = 0

        if (len(self.time_train_sec) == 0):
            print("Training hasn't been started")
            return 0

        #Iterate across the array to sum the total time of epochs
        for i in range(len(self.time_train_sec)):
            train_sec = train_sec + self.time_train_sec[i]
        return train_sec
    
    """
        Show the percentage of the training or validation
    """
    def show_percentatge(self, type : str , progress : int, total_images : int) -> None:
        percentage = min(progress * 100 / total_images, 100)
        print(f"\r{type} Progress: {percentage:.0f}%", end='', flush=True)

    """
        Train and validate the network
    """    
    def train_and_validation(self) -> None:
        """
            Interesting: https://stackoverflow.com/questions/51433378/what-does-model-train-do-in-pytorch
        """
        data_train, data_val = self.__transform_images_randomly()
        loader_train = DataLoader(data_train, num_workers = 4, batch_size = self.batch_size, shuffle = True)
        loader_val = DataLoader(data_val, num_workers = 4, batch_size = self.batch_size, shuffle = False)
        
        path_weights = f'{os.getcwd()}/class_weights.json'
        self.weights = self.utils_weights.calculate_class_weights(data_train, self.num_classes, path_weights)
        self.weights = self.weights.to(self.device)
        criterion = CrossEntropyLoss(weight = self.weights, ignore_index=255)

        optimizer = Adam(self.net.parameters(), lr = self.learning_rate, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-4)

        checkpoint_file = f'checkpoint_{self.name_training}.pth.tar'
        if self.is_resume_train:
            init_epoch = self.resume_training(optimizer, checkpoint_file)

        lambda1 = lambda epoch: pow((1 - ((epoch - 1) / self.epochs)), 0.9)
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda = lambda1)

        if(self.model == "SegSmall"):
            path_text_file = f'results/{self.name_training}/results_{self.name_training}_SegSmall.txt'
        elif(self.model == "erfnet"):
            path_text_file = f'results/{self.name_training}/results_{self.name_training}_erfnet.txt'
        else:
            raise Exception("Model not found")
        
        if (not os.path.exists(path_text_file)):    #dont add first line if it exists 
            with open(path_text_file, "a") as myfile:
                myfile.write(f'Using {self.model} model. Batch size: {self.batch_size} and epochs: {self.epochs} \n')
                myfile.write("Epoch\t\tTrain-loss\t\tTest-loss\t\t\Test-IoU(%)\t\tlearningRate")

        init_epoch = 1
        step_epochs = 1

        #Array to store the amount of iou
        iou_val_list = []
        best_epoch = 0

        #Values to store loss in training and validation
        train_loss = []
        val_loss = []
        total_images_train = len(loader_train.dataset)
        total_images_val = len(loader_val.dataset)


        print(f"Start training using: {self.epochs} epochs, {self.batch_size} batch size, Dataset train {total_images_train} images, Dataset val {total_images_val}!!!!!")
        for epoch in range(init_epoch, self.epochs + 1, step_epochs):
            
            self.net.train()  # Set the model to training mode
            print(f"Start training epoch: {epoch}")

            for param_group in optimizer.param_groups:
                print("LEARNING RATE: ", param_group['lr'])
                usedLr = float(param_group['lr'])

            self.__empty_cuda_memory()
            
            init_time_sec = time.time()
            epoch_loss_train = []
            progress = 0
            for batch in loader_train:
                inputs, targets = batch
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                # - Input [batch_size, channels, heigh, weight]
                # - Target [batch_size, height, weight]
                assert inputs.size(2) == targets.size(1), "Input and target height mismatch"
                assert inputs.size(3) == targets.size(2), "Input and target width mismatch"
                
                outputs = self.net(inputs)
                
                loss = criterion(outputs, targets.squeeze(1).long())
                optimizer.zero_grad() #grad to zero

                #Backpropragate to compute gradients
                loss.backward()

                #Update model parameters
                optimizer.step() 

                epoch_loss_train.append(loss.item())

                # Make sure not to exceed 100% if the last batch has fewer images
                progress+=self.batch_size
                self.show_percentatge("Training", progress, total_images_train)

            avg_train_loss = np.mean(epoch_loss_train)
            train_loss.append(avg_train_loss)

            #Calculate time for each epoch
            time_train_epoch_sec = time.time() - init_time_sec
            print(f"\nTime Train epoch: {time_train_epoch_sec:.2f} s")
            init_time_sec = time_train_epoch_sec #Now the iniial time is the end of the last epoch
            self.time_train_sec.append(time_train_epoch_sec)  #Save it for further evaluation
            
            # Validation step
            print(f"Start validation epoch: {epoch}")

            iouEval = iouCalc()

            self.net.eval()
            self.__empty_cuda_memory()
            progress = 0 #reset progress to show the percentage
            with torch.no_grad():
                val_loss_epoch = []
                val_iou_epoch = []
                val_mean_iou_epoch = []
                for batch in loader_val:   
                    inputs, targets = batch

                    inputs, targets = inputs.to(self.device), targets.to(self.device)
                    outputs = self.net(inputs)

                    #print(f"outputs shape: {outputs.shape}")
                    #print(f"targets shape: {targets.shape}")

                    #Compute loss
                    loss = criterion(outputs.squeeze(1), targets.squeeze(1).long())
                    val_loss_epoch.append(loss.item())

                    #Compute IoU
                    iou_per_class, mean_iou = iouEval.calculate_batch_iou(outputs, targets, self.num_classes)
                    val_iou_epoch.append(iou_per_class)
                    val_mean_iou_epoch.append(mean_iou)

                    progress+=self.batch_size
                    self.show_percentatge("Validation", progress, total_images_val)

            # Calculate average metrics for the entire epoch
            avg_val_loss = np.mean(val_loss_epoch)
            avg_val_iou = torch.stack(val_iou_epoch).mean(dim=0)
            avg_val_mean_iou = np.mean(val_mean_iou_epoch)

            val_loss.append(avg_val_loss)
            iou_val_list.append(avg_val_mean_iou)
            
            print(f"\nEpoch [{epoch}/{self.epochs}]")
            print(f"Average Epoch Training Loss: {avg_train_loss:.4f}, Average Epoch Validation Loss: {avg_val_loss:.4f}")
            print(f"Validation IoU per class: {avg_val_iou}")
            print(f"Validation Mean IoU: {avg_val_mean_iou:.4f}")

            #Write results to a text file
            with open(path_text_file, "a") as myfile:
                myfile.write("\n%d\t\t|%.4f\t\t|%.4f\t\t|%.4f\t\t|%.4f" % (
                epoch, avg_train_loss, avg_val_loss, avg_val_mean_iou, usedLr
            ))
                
            # Save the best weight for a checkpoint
            is_best = avg_val_mean_iou > self.best_iou
            if is_best:
                self.best_iou = avg_val_mean_iou
                best_epoch = epoch
            #save every epoch
            self.save_checkpoint(epoch, optimizer, avg_val_mean_iou, is_best)

            scheduler.step() #Update learning rate regardless any condition
            print("###############################################")
            
        #Create the plots for evaluating the network
        print(f"Best epoch {best_epoch} with IOU: {self.best_iou:.4f} %")
        self.plot.create_val_loss_plot(train_loss, val_loss)
        self.plot.create_iou_plot(iou_val_list)

def main():

    """
        Information about the dataset and classes of it:
            https://www.cityscapes-dataset.com/dataset-overview/
    """
    
    default_path_dataset = '../dataset/cityscapes/data'

    #Parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path",  type = str,   default = default_path_dataset, help = "Dataset to train")
    parser.add_argument("--batch_size",    type = int,   default = 5,                    help = "Batch size") 
    parser.add_argument("--epochs",        type = int,   default = 15,                   help = "Number of epochs to train")
    parser.add_argument("--num-classes",   type = int,   default = 19,                   help = "SegSmall classes. Required for loading the base network. Classes that are only evaluated in eval mode")
    parser.add_argument("--learning_rate", type = float, default = 0.005,                help = "Learning rate for the training")
    parser.add_argument("--resume_train",  type = bool,  default = False,                help = "Resume training from a checkpoint")
    parser.add_argument("--train_name",    type = str,   required = True,                help = "Training name to differentiate among them")
    parser.add_argument("--model",         type = str,   required = True,                help = "Model to use")
    args = parser.parse_args()

    check_device = CheckDevice() #Obtain the available device to use

    #Check dataset
    assert os.path.exists(args.dataset_path), "Error: dataset directory does not exist"

    try: 
        train_net = TrainNet(args, check_device())
        print(f"Model parameters: {train_net.count_param_model()}")
        train_net.train_and_validation()
        print(f"Training time {train_net.get_training_time():.2f} seconds")
        print("------------------------")
        print("Train finished!!!")

    except Exception as e:
        print(e)

if __name__ == "__main__":
    main()