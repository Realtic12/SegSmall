import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import os
import torch.cuda
import json
from torch.utils.data import DataLoader
from collections import Counter
"""
    @brief
    Class which is used to know the available resources in our system, GPU or CPU
"""
class CheckDevice:
    def __call__(self) -> str:
        num_gpus = torch.cuda.device_count()
        device = 'cuda' if num_gpus > 0 else 'cpu'
        if(device == 'cuda'):
            print(f"You are using cuda with: {num_gpus} gpu")
        else:
            print(f"You are using: {device}")
        return device

class Utils:
    def __init__(self, epochs : int, name_training : str) -> None:
        self.epochs = list(range(1, epochs + 1))  # Create a list of epoch numbers
        self.name_training = name_training
        self.size_fig = (10,5)

        self.results_folder_path = 'results/' + self.name_training        
        self.exists_results_folder()

    """
        Check if the results folder exists, if not, create it
    """
    def exists_results_folder(self) -> None:
        if not os.path.isdir('results'):
            os.mkdir('results')
        
        #Create folder according to the name of the training to store all the metrics
        if(not os.path.isdir('results/' + self.name_training)):
            os.mkdir('results/' + self.name_training)
        else:
            print("There's a training with the same name, do you want to override the results? (1: continue, 0: skip)")
            while True:
                answer = input()
                if answer in ["0", "1"]:
                    break
                else:
                    print("Invalid input. Please enter '1' to continue or '0' to skip.")
            if(answer == "0"):
                exit() #finish program
    
    """
        Create a figure with the size of the plot
    """
    def __create_figure(self) -> plt.figure:
        plt.figure(figsize = self.size_fig)
    
    """
        Save a plot in the results folder
    """
    def __save_fig(self, plot_type : str) -> None:
        plt.savefig(f'{self.results_folder_path}/{plot_type}_{self.name_training}.png')
        plt.close()

    """
        Create a plot of the loss of the training and validation process
    """
    def create_val_loss_plot(self, train_loss: list, val_loss: list) -> None:
        self.__create_figure()

        plt.plot(self.epochs, train_loss, label='Training Loss')
        plt.plot(self.epochs, val_loss, label='Validation Loss')

        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')

        #Add legend and grid
        plt.legend()
        plt.grid(True)
    
        # Set x-ticks to be every 5 epochs
        plt.xticks(range(0, len(self.epochs) + 1, 5))
    
        # Save the plot
        self.__save_fig("train_validation_loss")
    
    """
        Create a plot of the accuracy of the training and validation process
    """
    def create_accuracy_plot(self, train_acc: list, val_acc: list) -> None:

        self.__create_figure()

        plt.plot(self.epochs, train_acc, label='Training Accuracy')
        plt.plot(self.epochs, val_acc, label='Validation Accuracy')

        plt.xlabel('Epochs')
        plt.ylabel('Accuracy(%)')
        plt.title('Training and Validation Accuracy')

        #Add legend and grid
        plt.legend()
        plt.grid(True)  # Add grid for better readability

        # Set x-ticks to be every 5 epochs
        plt.xticks(range(0, len(self.epochs) + 1, 5))

        # Set y-axis to display percentages
        plt.gca().yaxis.set_major_formatter(PercentFormatter(xmax=1))
        
        # Save the plot
        self.__save_fig("train_validation_accuracy")
    
    """
        Create a plot of the precision of the training and validation process
    """ 
    def create_precision_plot(self, train_precision: list, val_precision: list) -> None:
        
        self.__create_figure()

        plt.plot(self.epochs, train_precision, label='Training Precision')
        plt.plot(self.epochs, val_precision, label='Validation Precision')

        plt.xlabel('Epochs')
        plt.ylabel('Precision(%)')
        plt.title('Training and Validation Precision')

        #Add legend and grid
        plt.legend()
        plt.grid(True)

        # Set x-ticks to be every 5 epochs
        plt.xticks(range(0, len(self.epochs) + 1, 5))

        # Set y-axis to display percentages
        plt.gca().yaxis.set_major_formatter(PercentFormatter(xmax=1))
        
        self.__save_fig("train_validation_precision")

    """
        Create a plot of the IoU of the training and validation process
    """
    def create_iou_plot(self, iou_val : list) -> None:        
        self.__create_figure()
        plt.plot(self.epochs, iou_val, 'b-', linewidth=2, markersize=8, marker='o')
        plt.title('IoU (%) vs Epochs')
        plt.xlabel('Epochs')
        plt.ylabel('IoU (%)')
        plt.grid(True, linestyle='--', alpha=0.7)

        # Annotate the highest IoU
        max_iou = max(iou_val)
        max_epoch = iou_val.index(max_iou) + 1
        plt.annotate(f'Max IoU: {max_iou:.2f}%',
                     xy=(max_epoch, max_iou),
                     xytext=(max_epoch, max_iou + 5),
                     arrowprops=dict(facecolor='black', shrink=0.05),
                     horizontalalignment='center')
        
        plt.ylim(0, 100)  # IoU percentage is between 0 and 100

        # Set y-axis to display integers
        plt.yticks(range(0, 101, 5))

        plt.tight_layout()
        self.__save_fig("iou_vs_epochs")


class Weights:
    def __init__(self, device : str) -> None:
        self.device = device

    """
        Load the class weigths from a JSON file
    """
    def load_class_weights(self, save_path : str) -> torch.tensor:
        with open(save_path, 'r') as f:
            weights_list = json.load(f)
        return torch.tensor(weights_list, device=self.device)

    """
        Calculate class weights for handling class imbalance. Load precocumpted weights if they exist
    """
    def calculate_class_weights(self, loader : DataLoader, num_classes : int, weights_file : str) -> torch.tensor:
        #Load weights if they are already calculated, in order to save time

        ignore_index = 255 #Cityscapes dataset ignores labels with id 255
        
        #print(f'Path weights: {weights_file}')
        if os.path.exists(weights_file):
            print(f"Weights file already exists at {weights_file}. Loading...")
            return self.load_class_weights(weights_file)
        
        print("Calculating class weights...")

        # Initialize a counter for all classes
        label_counts = Counter()
        
        # Iterate through the dataset
        for _, labels in loader:
        # Convert to NumPy, flatten, and then convert to a list of integers
            labels_np = labels.numpy().astype(int).flatten()
            # Filter out ignore_index
            valid_labels = labels_np[labels_np != ignore_index]
            label_counts.update(valid_labels)

        # Calculate weights
        total_samples = sum(label_counts.values())
        class_weights = torch.ones(num_classes, device = self.device)

        for class_label, count in label_counts.items():
            if count > 0:
                class_weights[class_label] = total_samples / (num_classes * count)
        
        # Normalize weights
        class_weights /= class_weights.sum()
        
        print(f"Calculated weights: {class_weights}")
        print(f"Label counts: {label_counts}")

        print("Saving weights...")
        weights_list = class_weights.cpu().tolist()
        with open(weights_file, 'w') as f:
            json.dump(weights_list, f)
            
        return class_weights

        