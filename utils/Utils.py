import random
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import os
import torch.cuda

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
    
    def create_val_loss_plot(self, train_loss: list, val_loss: list) -> None:
        plt.figure(figsize=self.size_fig)

        plt.plot(self.epochs, train_loss, label='Training Loss')
        plt.plot(self.epochs, val_loss, label='Validation Loss')

        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')

        #Add legend and grid
        plt.legend()
        plt.grid(True)
    
        # Set x-ticks to be every epoch, ensuring they are integers
        plt.xticks(self.epochs)
    
        # Save the plot
        plt.savefig(f'{self.results_folder_path}/training_validation_loss_{self.name_training}.png')
        plt.close()
    
    def create_accuracy_plot(self, train_acc: list, val_acc: list) -> None:
        plt.figure(figsize=self.size_fig)  # Set figure size for consistency
        plt.plot(self.epochs, train_acc, label='Training Accuracy')
        plt.plot(self.epochs, val_acc, label='Validation Accuracy')

        plt.xlabel('Epochs')
        plt.ylabel('Accuracy(%)')
        plt.title('Training and Validation Accuracy')

        #Add legend and grid
        plt.legend()
        plt.grid(True)  # Add grid for better readability

        # Set x-ticks to be every epoch, ensuring they are integers
        plt.xticks(self.epochs)

        # Set y-axis to display percentages
        plt.gca().yaxis.set_major_formatter(PercentFormatter(xmax=1))
        
        # Save the plot
        plt.savefig(f'{self.results_folder_path}/training_validation_accuracy_{self.name_training}.png')
        plt.close()
    
    def create_precision_plot(self, train_precision: list, val_precision: list) -> None:
        plt.figure(figsize=self.size_fig)
        plt.plot(self.epochs, train_precision, label='Training Precision')
        plt.plot(self.epochs, val_precision, label='Validation Precision')

        plt.xlabel('Epochs')
        plt.ylabel('Precision(%)')
        plt.title('Training and Validation Precision')

        #Add legend and grid
        plt.legend()
        plt.grid(True)

        plt.xticks(self.epochs)

        # Set y-axis to display percentages
        plt.gca().yaxis.set_major_formatter(PercentFormatter(xmax=1))
        
        plt.savefig(f'{self.results_folder_path}/training_validation_precision_{self.name_training}.png')
        plt.close()
        