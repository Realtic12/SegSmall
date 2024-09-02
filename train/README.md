# Training

The training have been done only using one gpu, otherwise the code is not prepared for that.

If you want to train the network, you can use the following command:

```
python train_net.py --train_name SegSmall_Cityscapes --model SegSmall --epochs 100 --batch_size 16 --learning_rate 0.0005
```
Parameters can be modified, passing the arguments when calling the previous python file.

## Results

The results folder contains the results of the training process, with the best model weights and a checkpoint after each poch, in case you want to resume the training process for whatever reason. Furthermore, you can find a text file where you can find the performance of the model according to our chosen metrics (loss, precision, Iou). Finally, according to that metrics, plots are generated to show the performance of the model during the training process.


