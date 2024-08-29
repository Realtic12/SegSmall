import torch

class iouCalc:
    def __init__(self, device : str, nClasses : int, ignoreIndex = 20):
        self.nClasses = nClasses
        self.ignoreIndex = ignoreIndex if nClasses > ignoreIndex else -1  # No ignoreIndex if it's out of range
        self.device = device
        self.reset()

    def reset(self):
        num_classes = self.nClasses if self.ignoreIndex == -1 else self.nClasses - 1
        self.tp = torch.zeros(num_classes).double()
        self.fp = torch.zeros(num_classes).double()
        self.fn = torch.zeros(num_classes).double()

    def addBatch(self, x, y):
        # Ensure x and y are on the same device
        y = y.to(self.device)
        x = x.to(self.device)

        # Convert predictions to one-hot encoding if needed
        if x.size(1) == 1:
            x_onehot = torch.zeros(x.size(0), self.nClasses, x.size(2), x.size(3), device = self.device)
            x_onehot.scatter_(1, x.long(), 1).float()
        else:
            x_onehot = x.float()

        if y.size(1) == 1:
            y_onehot = torch.zeros(y.size(0), self.nClasses, y.size(2), y.size(3), device = self.device)
            y_onehot.scatter_(1, y.long(), 1).float()
        else:
            y_onehot = y.float()

        # Handle ignoreIndex
        if self.ignoreIndex != -1:
            ignores = y_onehot[:, self.ignoreIndex].unsqueeze(1)
            x_onehot = x_onehot[:, :self.ignoreIndex]
            y_onehot = y_onehot[:, :self.ignoreIndex]
        else:
            ignores = torch.zeros_like(y_onehot[:, 0:1])

        # Calculate TP, FP, FN
        tpmult = x_onehot * y_onehot
        tp = tpmult.sum(dim=[0, 2, 3])
        fpmult = x_onehot * (1 - y_onehot - ignores)
        fp = fpmult.sum(dim=[0, 2, 3])
        fnmult = (1 - x_onehot) * y_onehot
        fn = fnmult.sum(dim=[0, 2, 3])

        # Accumulate
        self.tp += tp.double().cpu()
        self.fp += fp.double().cpu()
        self.fn += fn.double().cpu()

    def getIoU(self):
        """
        Compute the Intersection over Union (IoU) for each class and the mean IoU.

        Returns:
            - mean_iou: Average IoU across all classes (float).
            - iou_per_class: IoU for each class (tensor).
        """

        num = self.tp
        den = self.tp + self.fp + self.fn + 1e-15
        iou = num / den
        iou = torch.clamp(iou, min=0.0, max=1.0) #Ensure values between 0 and 1
        return iou.mean().item(), iou
    
class AccuracyCalc:
    def __call__(self, outputs, targets) -> float:
        _, predicted = outputs.max(1)
        total = (outputs == targets).float().sum().item()
        acc = 100 * total / targets.size(0)
        #return correct/total

class PrecisionCalc:

    def __call__(self, outputs : torch.tensor , targets : torch.tensor) -> float:
        """
            Calculate the precision score for classification.

            Parameters:
            - outputs: Model's output tensor (logits or probabilities).
            - targets: Ground truth labels (tensor).

            Returns the average precision score across the batch, between 0 an 1
        """
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

