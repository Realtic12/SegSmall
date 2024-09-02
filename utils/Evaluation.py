import torch
import numpy as np

class iouCalc:
    """
        Calculate IoU for a batch of predictions and targets.
    """
    def calculate_batch_iou(self, predictions : torch.tensor , targets : torch.tensor, num_classes : int, smooth=1e-6) -> torch.tensor:
        # Move tensors to CPU for numpy operations
        predictions = predictions.detach().cpu().numpy()  #numpy only on cpu tensors
        targets = targets.detach().cpu().numpy()
        
        # Convert predictions to class indices
        predictions = np.argmax(predictions, axis=1)
        
        intersection = np.zeros(num_classes)
        union = np.zeros(num_classes)
        
        for class_id in range(num_classes):
            # Create masks for predictions and targets for the current class
            pred_mask = (predictions == class_id)
            target_mask = (targets == class_id)

            # Compute intersection and union for the current class
            intersection[class_id] += np.sum(np.logical_and(pred_mask, target_mask))
            union[class_id] += np.sum(np.logical_or(pred_mask, target_mask))
        
        iou_classes = (intersection + smooth) / (union + smooth)

        # Convert to percentages
        iou_classes_percentage = iou_classes * 100
        mean_iou_percentatge = np.mean(iou_classes_percentage)
        
        return torch.from_numpy(iou_classes_percentage), mean_iou_percentatge

class PrecisionCalc:
    """
        Calculate the precision score for classification.

        Parameters:
        - outputs: Model's output tensor (logits or probabilities).
        - targets: Ground truth labels (tensor).

        Returns the average precision score across the batch, between 0 an 1"""
    def __call__(self, outputs : torch.tensor , targets : torch.tensor) -> float:
       
        #print(f'Batch size {targets.size(0)}')
        #print(f'Num classes {outputs.size(1)}')
        batch_size = targets.size(0)
        precision_scores = []
        for i in range(batch_size):
            output = outputs[i]
            target = targets[i]

            _, predicted = torch.max(output, 0)
            true_positive = ((predicted == target) & (target != 0)).float().sum()
            predicted_positive = (predicted != 0).float().sum()
            precision = true_positive / (predicted_positive + 1e-8)  # Adding small epsilon to avoid division by zero
            
            precision_scores.append(precision.item())

        return sum(precision_scores) / batch_size  # Return the average precision across the batch

