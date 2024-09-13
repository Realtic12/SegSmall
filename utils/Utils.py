
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


"""
    Assign different weights to an imbalanced dataset, in order to diminish the classes less representated.
    If they are already calculated, it retrieves them from a json fil. Otherwise, it will calculate them.
"""
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

        