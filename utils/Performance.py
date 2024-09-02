import time
from typing import Tuple

import torch
import sys
sys.path.append("..")

from model.SegSmall import SegSmall
from model.erfnet import Net
from utils.Utils import CheckDevice

class PerformanceEvaluator:
    def __init__(self, num_classes: int, device: str, input_shape: Tuple[int, int, int, int]) -> None:
        """
        Initialize the PerformanceEvaluator.

        Args:
            num_classes (int): Number of classes for the segmentation model.
            device (str): Device to run the model on ('cuda' or 'cpu').
            input_shape (tuple): Shape of the input tensor (batch_size, channels, height, width).
        """
        self.num_classes = num_classes
        self.input_shape = input_shape
        self.device = device
        self.model = SegSmall(num_classes=self.num_classes).to(self.device)
        self.input_tensor = torch.randn(*self.input_shape).to(self.device)

    def __warm_up(self, num_runs: int = 10) -> None:
        """
        Warm up the model to avoid cold start issues.

        Args:
            num_runs (int): Number of warm-up runs. Defaults to 10.
        """
        with torch.no_grad():
            for _ in range(num_runs):
                _ = self.model(self.input_tensor)

    def __measure_inference_time(self, n_runs: int = 100) -> float:
        """
        Measure the inference time of the model.

        Args:
            n_runs (int): Number of inference runs. Defaults to 100.

        Returns:
            float: Average inference time in seconds.
        """
        torch.cuda.synchronize()
        start_time = time.time()

        with torch.no_grad():
            for _ in range(n_runs):
                _ = self.model(self.input_tensor)

        torch.cuda.synchronize()
        end_time = time.time()
        total_inference_time = end_time - start_time
        return total_inference_time / n_runs

    def __calculate_fps(self, average_inference_time: float) -> float:
        """
        Calculate the FPS of the model.

        Args:
            average_inference_time (float): Average inference time in seconds.

        Returns:
            float: Frames Per Second (FPS).
        """
        return 1 / average_inference_time

    def evaluate(self, warm_up_runs: int = 10, measurement_runs: int = 100) -> None:
        """
        Evaluate the performance of the model.

        Args:
            warm_up_runs (int): Number of warm-up runs. Defaults to 10.
            measurement_runs (int): Number of measurement runs. Defaults to 100.
        """
        try:
            self.__warm_up(num_runs=warm_up_runs)
            average_inference_time = self.__measure_inference_time(n_runs=measurement_runs)
            fps = self.__calculate_fps(average_inference_time)

            print(f'Average Inference Time: {average_inference_time:.6f} seconds')
            print(f'Frames Per Second (FPS): {fps:.2f}')
        except RuntimeError as e:
            print(f"An error occurred during evaluation: {e}")
            if 'CUDA' in str(e):
                print("This might be due to insufficient GPU memory. Try reducing the input size or using CPU.")

def main():
    NUM_CLASSES = 19
    check_device = CheckDevice()
    shape = (1, 3, 512, 256)
    
    try:
        evaluator = PerformanceEvaluator(num_classes=NUM_CLASSES,
                                         device=check_device(),
                                         input_shape=shape)
        evaluator.evaluate()
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()