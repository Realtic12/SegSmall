import torch
import os

"""
    Class to save a model, two options:
        -After each epoch
        -When the model reaches a higher IoU val
"""
class SaveModel:
    def __call__(self, epoch : int, optimizer, iou : float,  
                  name_training : str, save_dir : str, 
                  model, is_best = False) -> None:
        self.model = model
        self.save_dir = save_dir
        self.name_training = name_training

        checkpoint = {
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_val_iou': iou
        }

        torch.save(checkpoint, os.path.join(self.save_dir, f'{self.name_training}_checkpoint.pth.tar'))

        if is_best:
            print(f"Saving model as best with {iou:.4f} %")
            torch.save(self.model.state_dict(), os.path.join(self.save_dir, f'{self.name_training}_best_weights.pth'))