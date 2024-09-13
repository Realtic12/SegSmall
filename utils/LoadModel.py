import torch
import os

"""
    Class to load weights from a finished model or to resume a training from a checkpoint
"""
class LoadModel:
    def __init__(self, model):
        self.model = model

    """
        Load weights from a model
    """
    def load_model_weights(self, path_to_weights : str):
        model = self.model.load_state_dict(torch.load(path_to_weights), weights_only=True)
        return model

    """ 
        Resume training from a checkpoint
    """
    def resume_training(self, optimizer, checkpoint_file : str, 
                         save_dir : str) -> int:
        filenameCheckpoint = save_dir + '/' + checkpoint_file
        
        #Check if the file exists, although the user wants to reignite the training
        if not os.path.exists(filenameCheckpoint):
            raise Exception("There's no checkpoint")
        
        filenameCheckpoint = save_dir + checkpoint_file
        checkpoint = torch.load(filenameCheckpoint)
        self.model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        self.best_iou = checkpoint['best_val_iou']
        epoch = checkpoint['epoch']
        print("=> Loaded checkpoint at epoch {})".format(checkpoint['epoch']))
        return epoch