import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import os

"""
    Class used to draw the necessary plots to evaluate visually the results from the model
"""
class Plot:
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